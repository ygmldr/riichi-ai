#!/usr/bin/env python3
"""t2 tests: EX1 oracle label extraction — stream pairing, decision alignment, label files.

Covers PREREG §3 truth definitions (`deal_in`, `exp_loss = -deltas[us]`, multi-ron, chankan,
34-class tiles) and the t2 acceptance criteria (frozen-fixture decode vs `verify_and_label`,
`extract_labels` identity pinning for the train manifest and t3's corpus index).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from exp_ex1_labels import (CANON_KEY, LABEL_SCHEMA, TRAIN_YEARS, RecordError, _bootstrap_mortal,
                            align_decisions, deaka, extract_labels, find_rons, label_file,
                            load_labels, load_targets, parse_events, sparse_ron_labels, tile_id)

FIXTURE = REPO_ROOT / "runs/1v3/10219_579411421376059130_d.json.gz"
FIXTURE_SEAT = 3  # split d -> seat 3
PREREG_PATH = REPO_ROOT / "scripts/exp_ex1_prereg.md"
TRAIN_MANIFEST = REPO_ROOT / "runs/exp/EX1/train_manifest.json"
HOLDOUT_LABELS = REPO_ROOT / "runs/exp/EX1/holdout_labels.json"
CORPUS_INDEX = REPO_ROOT / "runs/exp/EX1/corpus_index.json"


def hora(actor: int, target: int, deltas=None, **kw) -> dict:
    ev = {"type": "hora", "actor": actor, "target": target}
    if deltas is not None:
        ev["deltas"] = deltas
    return {**ev, **kw}


class TileTest(unittest.TestCase):
    def test_deaka_folds_aka_onto_base_fives(self):
        given = {"5mr": 34, "5pr": 35, "5sr": 36}
        when = {name: deaka(tile_id(name)) for name in given}
        then = {"5mr": 4, "5pr": 13, "5sr": 22}
        self.assertEqual(when, then)

    def test_tile_table_matches_jiangpu_tiles(self):
        from jiangpu.tiles import TILE_STRINGS as JIANGPU_STRINGS
        from jiangpu.tiles import deaka as jiangpu_deaka
        from jiangpu.tiles import tile_id as jiangpu_tile_id
        for i, name in enumerate(JIANGPU_STRINGS[:37]):
            self.assertEqual(tile_id(name), jiangpu_tile_id(name), name)
            self.assertEqual(deaka(i), jiangpu_deaka(i), name)

    def test_tile_ids_are_34_class(self):
        base = [tile_id(f"{n}{s}") for s in "mps" for n in range(1, 10)]
        honors = [tile_id(x) for x in ("E", "S", "W", "N", "P", "F", "C")]
        self.assertEqual(sorted(base + honors), list(range(34)))

    def test_unknown_tile_string_rejected(self):
        with self.assertRaises(RecordError):
            tile_id("10m")


class FindRonsTest(unittest.TestCase):
    def test_single_ron_pairs_with_preceding_dahai_and_aka_folds(self):
        events = [
            {"type": "dahai", "actor": 3, "pai": "5pr", "tsumogiri": False},
            hora(1, 3, [0, 3900, 0, -3900]),
        ]
        rons = find_rons(events)
        self.assertEqual([(r["actor"], r["target"], r["tile"], r["loss"]) for r in rons],
                         [(1, 3, 13, 3900)])

    def test_multi_ron_counts_once_per_hora_event(self):
        events = [
            {"type": "dahai", "actor": 3, "pai": "2m", "tsumogiri": True},
            hora(0, 3, [2900, 0, 0, -2900]),
            hora(1, 3, [0, 2000, 0, -2000]),
        ]
        rons = find_rons(events)
        self.assertEqual([(r["actor"], r["tile"], r["loss"]) for r in rons],
                         [(0, 1, 2900), (1, 1, 2000)])

    def test_chankan_pairs_with_preceding_kakan(self):
        events = [
            {"type": "tsumo", "actor": 0, "pai": "7s"},
            {"type": "kakan", "actor": 0, "pai": "7s", "consumed": ["7s", "7s", "7s"]},
            hora(1, 0, [-4200, 4200, 0, 0]),
        ]
        rons = find_rons(events)
        self.assertEqual([(r["actor"], r["tile"], r["loss"]) for r in rons], [(1, 24, 4200)])

    def test_chankan_survives_interleaved_dora_event(self):
        events = [
            {"type": "kakan", "actor": 0, "pai": "2m", "consumed": ["2m", "2m", "2m"]},
            {"type": "dora", "dora_marker": "9p"},
            hora(1, 0, [-8900, 8900, 0, 0]),
        ]
        self.assertEqual([r["tile"] for r in find_rons(events)], [1])

    def test_tsumo_is_not_a_deal_in(self):
        events = [hora(2, 2, [0, 0, 8000, -8000])]
        self.assertEqual(find_rons(events), [])

    def test_missing_deltas_fails_loud(self):
        events = [
            {"type": "dahai", "actor": 3, "pai": "1m", "tsumogiri": False},
            hora(1, 3),
        ]
        with self.assertRaises(RecordError):
            find_rons(events)

    def test_bad_delta_sign_fails_loud(self):
        events = [
            {"type": "dahai", "actor": 3, "pai": "1m", "tsumogiri": False},
            hora(1, 3, [0, 1000, 0, 0]),
        ]
        with self.assertRaises(RecordError):
            find_rons(events)

    def test_ron_without_preceding_discard_fails_loud(self):
        with self.assertRaises(RecordError):
            find_rons([hora(1, 3, [0, 1000, 0, -1000])])

    def test_ankan_chankan_is_rejected_not_mispaird(self):
        events = [
            {"type": "dahai", "actor": 3, "pai": "1m", "tsumogiri": False},
            {"type": "ankan", "actor": 0, "consumed": ["9m", "9m", "9m", "9m"]},
            hora(1, 0, [-4200, 4200, 0, 0]),
        ]
        with self.assertRaises(RecordError):
            find_rons(events)


class AlignDecisionsTest(unittest.TestCase):
    US = 3
    ACTIONS = np.asarray([41, 32, 45, 37, 0, 42, 13, 43, 44, 45], dtype=np.int64)
    EVENTS = [
        {"type": "pon", "actor": US, "pai": "F", "consumed": ["F", "F"]},
        {"type": "dahai", "actor": US, "pai": "F", "tsumogiri": False},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "reach", "actor": US},
        {"type": "dahai", "actor": US, "pai": "1m", "tsumogiri": False},
        {"type": "kakan", "actor": US, "pai": "5p", "consumed": ["5p", "5p", "5p"]},
        hora(US, 1, [0, -1000, 0, 1000]),
    ]

    def test_clean_stream_maps_every_decision_and_marks_kan_select(self):
        al = align_decisions(self.ACTIONS, self.EVENTS, self.US)
        self.assertEqual(al["dec_event"].tolist(), [0, 1, -1, 3, 4, 5, -1, 6, -1, -1])
        self.assertEqual(al["is_kan_select"].tolist(), [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(al["kinds"], {"pon": 1, "dahai": 2, "reach": 1, "kakan": 1, "hora": 1})

    def test_wrong_discard_tile_fails_loud(self):
        bad = self.ACTIONS.copy()
        bad[1] = 0  # says 1m, event says F
        with self.assertRaises(RecordError):
            align_decisions(bad, self.EVENTS, self.US)

    def test_kan_without_select_pseudo_decision_fails_loud(self):
        bad = np.asarray([42], dtype=np.int64)
        with self.assertRaises(RecordError):
            align_decisions(bad, [self.EVENTS[5]], self.US)

    def test_unclaimed_action_event_fails_loud(self):
        with self.assertRaises(RecordError):
            align_decisions(np.asarray([45], dtype=np.int64), self.EVENTS, self.US)


class SparseRonLabelsTest(unittest.TestCase):
    def test_known_ron_lands_on_exact_cell_and_sign_is_positive_loss(self):
        deal_in, exp_loss = sparse_ron_labels([(3, 1, 13, 3900)], n_decisions=5)
        self.assertEqual(deal_in.shape, (5, 3, 34))
        self.assertEqual(deal_in.sum(), 1.0)
        self.assertEqual(deal_in[3, 1, 13], 1.0)
        self.assertEqual(exp_loss[3, 1, 13], 3900.0)
        self.assertEqual(exp_loss.sum(), 3900.0)
        other = exp_loss.copy()
        other[3, 1, 13] = 0.0
        self.assertEqual(other.sum(), 0.0)

    def test_multi_ron_sums_per_opponent_on_same_discard(self):
        _, exp_loss = sparse_ron_labels([(1, 0, 1, 2900), (1, 1, 1, 2000)], n_decisions=2)
        self.assertEqual(float(exp_loss[1, :, 1].sum()), 4900.0)
        self.assertEqual(float(exp_loss[1, 2, 1]), 0.0)

    def test_negative_or_out_of_range_rows_rejected(self):
        with self.assertRaises(RecordError):
            sparse_ron_labels([(0, 0, 0, -100)], 1)
        with self.assertRaises(RecordError):
            sparse_ron_labels([(1, 0, 0, 100)], 1)


class HandcraftedExpLossTest(unittest.TestCase):
    """Acceptance: one handcrafted ron on our discard -> one exact positive `exp_loss` cell."""

    def test_ron_on_us_maps_to_positive_exp_loss_cell_and_zeros_elsewhere(self):
        us = 3
        events = [
            {"type": "tsumo", "actor": us, "pai": "1m"},
            {"type": "dahai", "actor": us, "pai": "3s", "tsumogiri": True},
            hora(1, us, [0, 3900, 0, -3900]),
        ]
        actions = np.asarray([20], dtype=np.int64)  # 3s
        al = align_decisions(actions, events, us)
        rows = []
        for ron in find_rons(events):
            if ron["target"] != us:
                continue
            j = (ron["actor"] - us - 1) % 4
            rows.append((al["event_to_dec"][ron["discard_index"]], j, ron["tile"], ron["loss"]))
        deal_in, exp_loss = sparse_ron_labels(rows, n_decisions=1)
        self.assertEqual(j, 1)
        self.assertEqual(deal_in[0, 1, 20], 1.0)
        self.assertEqual(exp_loss[0, 1, 20], 3900.0)
        self.assertEqual(float(exp_loss.sum()), 3900.0)
        self.assertEqual(float(deal_in.sum()), 1.0)


class FrozenFixtureDecodeTest(unittest.TestCase):
    """Acceptance: per-(decision,opponent) decode must agree with `verify_and_label` (0 critical)."""

    def test_decode_matches_verify_and_label(self):
        self.assertTrue(FIXTURE.is_file(), FIXTURE)
        import gzip
        from libriichi.dataset import GameplayLoader
        _bootstrap_mortal()
        from exp_a2_probe import CRITICAL_COUNTERS, verify_and_label

        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "labels.npz"
            info = label_file(FIXTURE, FIXTURE_SEAT, out, prereg_sha256="test")
            raw = load_labels(out)
            targets = load_targets(out)

        # independent decode of the same fixture for the same POV
        text = gzip.decompress(FIXTURE.read_bytes()).decode("utf-8")
        games = GameplayLoader(version=4, oracle=True).load_log(text)
        game = games[FIXTURE_SEAT]
        self.assertEqual(game.take_player_id(), FIXTURE_SEAT)
        actions = game.take_actions()
        obs = np.stack([np.asarray(o) for o in game.take_obs()])
        inv = np.stack([np.asarray(o) for o in game.take_invisible_obs()])
        own_sh = np.asarray(game.take_shantens(), dtype=np.int8)
        labels, counters = verify_and_label(obs, inv, own_sh)

        self.assertEqual(info["counters"], {k: int(v) for k, v in counters.items()})
        self.assertFalse({k: v for k, v in counters.items() if k in CRITICAL_COUNTERS and v > 0})
        self.assertEqual(info["n_decisions"], len(actions))
        self.assertEqual(raw["schema"], LABEL_SCHEMA)
        self.assertEqual(raw["meta"]["player_id"], FIXTURE_SEAT)
        self.assertEqual(int(raw["actions"].shape[0]), len(actions))
        self.assertEqual(raw["opp_shanten"].shape, (len(actions), 3))
        self.assertEqual(raw["opp_waits_packed"].shape, (len(actions), 3, 5))

        waits_any = np.unpackbits(raw["opp_waits_packed"], axis=-1, bitorder="little")[..., :34].any(axis=2)
        self.assertEqual(int((waits_any != (raw["opp_shanten"] == 0)).sum()),
                         counters["waits_vs_shanten"])
        self.assertTrue(np.array_equal(labels["opp_any_tenpai"],
                                       raw["opp_shanten"].min(axis=1) <= 0))

        # dense targets: PREREG shapes, sparse realized ron rows, continuous exp_loss
        self.assertEqual(targets["opp_tenpai_prob"].shape, (len(actions), 3))
        self.assertEqual(targets["wait_membership"].shape, (len(actions), 3, 34))
        self.assertEqual(targets["deal_in"].shape, (len(actions), 3, 34))
        self.assertEqual(targets["exp_loss"].shape, (len(actions), 3, 34))
        self.assertEqual(float(targets["deal_in"].sum()), float(info["n_rons"]))
        self.assertEqual(float(targets["exp_loss"].sum()), float(info["exp_loss_sum"]))
        self.assertEqual(float(targets["opp_tenpai_prob"].sum()), float(info["n_opp_tenpai"]))
        self.assertTrue(np.all(targets["exp_loss"] >= 0.0))
        self.assertTrue(np.all((targets["deal_in"] == 0.0) | (targets["deal_in"] == 1.0)))
        self.assertEqual(int(targets["real_mask"].sum()),
                         len(actions) - int(raw["is_kan_select"].sum()))


class ExtractLabelsTest(unittest.TestCase):
    def test_identity_pinning_and_byte_determinism(self):
        if not FIXTURE.is_file():
            self.skipTest(f"missing fixture {FIXTURE}")
        identities = [(900_000_000, CANON_KEY, "a"), (900_000_000, CANON_KEY, "b")]
        with tempfile.TemporaryDirectory() as td:
            first = extract_labels([FIXTURE, FIXTURE], td, identities=identities,
                                   subset="calib_sub", progress_every=0)
            hashes = {r["label_path"]: hashlib.sha256(pathlib.Path(r["label_path"]).read_bytes()).hexdigest()
                      for r in first["files"]}
            self.assertEqual(first["n_ok"], 2)
            self.assertEqual(len(hashes), 2)
            self.assertEqual({r["seed"] for r in first["files"]}, {900_000_000})
            self.assertEqual({r["split"] for r in first["files"]}, {"a", "b"})
            for r in first["files"]:
                self.assertTrue(pathlib.Path(r["label_path"]).is_file(), r["label_path"])
                self.assertEqual(pathlib.Path(r["label_path"]).name,
                                 f"{r['seed']}_{r['key']}_{r['split']}.npz")
            second = extract_labels([FIXTURE, FIXTURE], td, identities=identities,
                                    subset="calib_sub", progress_every=0)
            for r in second["files"]:
                again = hashlib.sha256(pathlib.Path(r["label_path"]).read_bytes()).hexdigest()
                self.assertEqual(hashes[r["label_path"]], again)

    def test_default_identity_and_seat_rotation(self):
        if not FIXTURE.is_file():
            self.skipTest(f"missing fixture {FIXTURE}")
        with tempfile.TemporaryDirectory() as td:
            summary = extract_labels([FIXTURE], td, subset="select_sub", progress_every=0)
        entry = summary["files"][0]
        self.assertEqual((entry["seed"], entry["key"], entry["split"]), (903_000_000, CANON_KEY, "a"))
        self.assertEqual(entry["player_id"], 0)
        self.assertTrue(pathlib.Path(entry["label_path"]).name == "903000000_4544561_a.npz")

    def test_missing_file_raises_record_error(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RecordError):
                extract_labels([REPO_ROOT / "runs/1v3/does_not_exist_a.json.gz"], td)


class TrainManifestArtifactTest(unittest.TestCase):
    @unittest.skipUnless(TRAIN_MANIFEST.is_file(), "run scripts/exp_ex1_labels.py train-manifest first")
    def test_manifest_is_bounded_and_label_pinned(self):
        manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "ex1-train-manifest/v1")
        self.assertEqual(manifest["seed"], 13)
        lo, hi = manifest["bounds"]
        self.assertTrue(lo <= manifest["counts"]["n_files"] <= hi, manifest["counts"])
        self.assertEqual(manifest["counts"]["n_error"], 0)
        self.assertEqual(manifest["years"], list(TRAIN_YEARS))
        files = manifest["files"]
        self.assertEqual(len(files), manifest["counts"]["n_files"])
        missing = []
        for i, entry in enumerate(files):
            self.assertEqual((entry["seed"], entry["key"], entry["split"]),
                             (903_000_000 + i // 4, CANON_KEY, "abcd"[i % 4]))
            self.assertIn(entry["year"], TRAIN_YEARS)
            self.assertEqual(entry["player_id"], "abcd".index(entry["split"]))
            if not (REPO_ROOT / entry["label_path"]).is_file():
                missing.append(entry["label_path"])
            self.assertEqual(pathlib.Path(entry["label_path"]).name,
                             f"{entry['seed']}_{entry['key']}_{entry['split']}.npz")
        self.assertEqual(missing, [])
        self.assertTrue((REPO_ROOT / files[0]["source_path"]).is_file(), files[0]["source_path"])
        raw = load_labels(REPO_ROOT / files[0]["label_path"])
        self.assertEqual(raw["schema"], LABEL_SCHEMA)
        self.assertEqual(raw["meta"]["seed"], files[0]["seed"])
        self.assertEqual(raw["meta"]["source_sha256"], files[0]["source_sha256"])

    @unittest.skipUnless(HOLDOUT_LABELS.is_file() and CORPUS_INDEX.is_file(),
                         "run scripts/exp_ex1_labels.py holdout first")
    def test_holdout_labels_match_corpus_index(self):
        index = json.loads(CORPUS_INDEX.read_text(encoding="utf-8"))
        holdout = json.loads(HOLDOUT_LABELS.read_text(encoding="utf-8"))
        self.assertEqual(holdout["schema"], "ex1-holdout-labels/v1")
        for name, block in index["subsets"].items():
            self.assertEqual(holdout["subsets"][name]["count"], len(block["files"]))
            for f in block["files"]:
                label = REPO_ROOT / holdout["subsets"][name]["label_dir"] / \
                    f"{f['seed']}_{f['key']}_{f['split']}.npz"
                self.assertTrue(label.is_file(), label)
            first = block["files"][0]
            raw = load_labels(REPO_ROOT / holdout["subsets"][name]["label_dir"] /
                              f"{first['seed']}_{first['key']}_{first['split']}.npz")
            self.assertEqual(raw["meta"]["source_sha256"], first["target_sha256"])
            self.assertEqual(raw["meta"]["player_id"], "abcd".index(first["split"]))


class PreregDisciplineTest(unittest.TestCase):
    def test_prereg_hash_matches_inherited_pin(self):
        from exp_ex1_audit import load_prereg
        self.assertEqual(load_prereg(PREREG_PATH),
                         "eb7d6bc99d20f2a3aacd42528f888f42e065ad09f5861d6b0b3cbe477aff0153")

    def test_parse_events_rejects_garbage(self):
        with self.assertRaises(RecordError):
            parse_events("")
        with self.assertRaises(RecordError):
            parse_events("{not json}")


if __name__ == "__main__":
    unittest.main()
