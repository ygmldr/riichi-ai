#!/usr/bin/env python3
"""t12: `context` provider — public state/turn context (free, no checkpoint).

Bit-exact vs obs slices on sampled `report_sub` decisions (PREREG §6.2 F2: 0 mismatches),
NO-REVEAL key allowlist (`doras_owned` excluded, M12/F6), `basis="public_info_predictor"` +
`no_hidden_info=true` (NB-8). Writes `runs/exp/EX1/context_validity.json`."""
from __future__ import annotations

import json
import pathlib
import sys
import unittest
from datetime import datetime, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from exp_ex1_audit import PREREG_PATH, assert_prereg_unchanged, load_prereg  # noqa: E402

CORPUS_INDEX = REPO_ROOT / "runs/exp/EX1/corpus_index.json"
OUT_PATH = REPO_ROOT / "runs/exp/EX1/context_validity.json"
PROVIDER_SRC = REPO_ROOT / "scripts/jiangpu/providers/context.py"
SAMPLE_FILES = 12
TILES = 34
OBS_ROWS = 1012

BLOCK_NAMES = ("scores", "honba", "kyotaku", "round_index", "tiles_left", "tiles_seen")
FIELD_NAMES = ("tiles_left", "honba", "kyotaku", "scores", "round_index", "tiles_seen", "at_turn")
META_KEYS = ("provider", "provider_version", "checkpoint_sha256", "unit", "no_hidden_info", "basis")
ALLOWED_KEYS = set(META_KEYS) | set(FIELD_NAMES) | {"unit_by_field"}
CAPS = {"scores": 100000, "honba": 10, "kyotaku": 10, "round_index": 7, "tiles_left": 69}
FORBIDDEN_FIELD = "doras_owned"  # M12/F6: self face = hand+melds ⇒ self_hand, never public


def _spans():
    from jiangpu import layout
    top = layout.load_layout()
    return {name: layout.span(top, name) for name in BLOCK_NAMES}


def _expected_fields(dec, spans):
    """Recompute the decoded public fields straight from the obs slices (the frozen spec)."""
    obs = dec.obs
    score_off = spans["scores"][0]
    return {
        "tiles_left": int(round(float(obs[spans["tiles_left"][0]][0]) * CAPS["tiles_left"])),
        "honba": int(round(float(obs[spans["honba"][0]][0]) * CAPS["honba"])),
        "kyotaku": int(round(float(obs[spans["kyotaku"][0]][0]) * CAPS["kyotaku"])),
        "scores": [int(round(float(obs[score_off + 2 * k][0]) * CAPS["scores"])) for k in range(4)],
        "round_index": int(round(float(obs[spans["round_index"][0]][0]) * CAPS["round_index"])),
        "tiles_seen": [int(round(float(obs[spans["tiles_seen"][0]][t]) * 4)) for t in range(TILES)],
        "at_turn": int(dec.at_turn),
    }


def _synthetic_decisions(spans):
    """One crafted decision whose obs slices encode known values (no fixture needed)."""
    import numpy as np
    from jiangpu.replay import Decision
    obs = np.zeros((OBS_ROWS, TILES), dtype=np.float32)
    for block, value, cap in (("tiles_left", 57, 69), ("honba", 3, 10),
                              ("kyotaku", 2, 10), ("round_index", 5, 7)):
        obs[spans[block][0]][0] = np.float32(value / cap)
    score_off = spans["scores"][0]
    for k, score in enumerate((25000, 24000, 12000, 30000)):
        obs[score_off + 2 * k][0] = np.float32(score / 100000)
        obs[score_off + 2 * k + 1][0] = np.float32(score / 30000)
    seen_off = spans["tiles_seen"][0]
    for t in range(TILES):
        obs[seen_off][t] = np.float32((t % 5) / 4)
    return [Decision(index=7, player_id=1, obs=obs, mask=np.zeros(46, dtype=np.bool_),
                     legal_actions=[], logged_action=0, at_kyoku=0, at_turn=4,
                     shanten=2, done=False, public={})]


def _load_sample():
    """Deterministic first-`SAMPLE_FILES` of `report_sub`; [] when the corpus is absent."""
    if not CORPUS_INDEX.exists():
        return []
    index = json.loads(CORPUS_INDEX.read_text(encoding="utf-8"))
    entries = index["subsets"]["report_sub"]["files"][:SAMPLE_FILES]
    from eval_report import SEAT_OF_SPLIT
    from jiangpu.record import read_record
    from jiangpu.replay import iter_decisions
    sample = []
    for entry in entries:
        record = read_record(str((REPO_ROOT / entry["target_path"]).resolve()))
        sample.append((entry, iter_decisions(record, pov=int(SEAT_OF_SPLIT[record.split]))))
    return sample


def _run_validity(sample, prereg_sha):
    """Sample-wide bit-exactness + event cross-check; assembles the t12 artifact."""
    from jiangpu.providers import context
    spans = _spans()
    obs_slices = {field: {"n": 0, "mismatches": 0} for field in FIELD_NAMES}
    events = {f: {"n": 0, "mismatches": 0} for f in ("scores", "honba", "kyotaku", "round_index")}
    n_decisions = n_unavailable = 0
    files = []
    for entry, decisions in sample:
        files.append(entry["target_path"])
        payloads = context.compute(decisions)
        for dec in decisions:
            n_decisions += 1
            payload = payloads.get(dec.index)
            if payload is None or payload.get("unavailable"):
                n_unavailable += 1
                continue
            expected = _expected_fields(dec, spans)
            for field in FIELD_NAMES:
                obs_slices[field]["n"] += 1
                obs_slices[field]["mismatches"] += int(payload[field] != expected[field])
        seen = set()
        for dec in decisions:
            if dec.at_kyoku in seen:
                continue
            seen.add(dec.at_kyoku)
            expected = _expected_fields(dec, spans)
            rel_scores = [int(dec.public["scores"][(dec.player_id + k) % 4]) for k in range(4)]
            ev_round = (0 if dec.public["bakaze"] == "E" else 1) * 4 + int(dec.public["kyoku"]) - 1
            for field, ok in (("scores", expected["scores"] == rel_scores),
                              ("honba", expected["honba"] == dec.public["honba"]),
                              ("kyotaku", expected["kyotaku"] == dec.public["kyotaku"]),
                              ("round_index", expected["round_index"] == ev_round)):
                events[field]["n"] += 1
                events[field]["mismatches"] += int(not ok)
    clean = n_decisions > 0 and all(c["mismatches"] == 0 for c in obs_slices.values())
    return {
        "schema": "ex1-context-validity/v1",
        "todo": "t12",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": {"name": context.PROVIDER_NAME, "version": context.PROVIDER_VERSION,
                     "unit": context.UNIT, "unit_by_field": context.UNIT_BY_FIELD,
                     "no_hidden_info": context.NO_HIDDEN_INFO, "basis": context.BASIS,
                     "checkpoint_sha256": None},
        "prereg": {"path": "scripts/exp_ex1_prereg.md", "sha256": prereg_sha},
        "sample": {"subset": "report_sub", "n_files": len(sample), "n_decisions": n_decisions,
                   "n_unavailable": n_unavailable, "files": files},
        "coverage": (n_decisions - n_unavailable) / n_decisions if n_decisions else None,
        "obs_slice_mismatches": obs_slices,
        "event_cross_check": {"subset": "report_sub", "fields": events},
        "no_concealed_hand_fields": True,
        "verdict": "delivery" if clean else "not_estimable",
    }


class ContextProviderTest(unittest.TestCase):
    """t12: public context provider — bit-exact, NO-REVEAL, contract v2 metadata."""

    @classmethod
    def setUpClass(cls):
        cls.prereg_sha = load_prereg(PREREG_PATH)
        cls.spans = _spans()
        from jiangpu.providers import context
        cls.synth = _synthetic_decisions(cls.spans)
        cls.synth_payloads = context.compute(cls.synth)
        cls.sample = _load_sample()
        cls.artifact = None
        if cls.sample:
            cls.artifact = _run_validity(cls.sample, cls.prereg_sha)
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_text(json.dumps(cls.artifact, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        assert_prereg_unchanged(PREREG_PATH, cls.prereg_sha)

    def _compute_with_layout(self, mutate):
        """Run `compute` against a mutated layout copy; restores the real loader after."""
        from jiangpu import layout
        original = layout.load_layout
        top = dict(original())
        mutate(top)
        layout.load_layout = lambda *a, **k: top
        try:
            from jiangpu.providers import context
            return context.compute(self.synth)
        finally:
            layout.load_layout = original

    def test_synthetic_payload_is_exactly_the_frozen_decode(self):
        payload = self.synth_payloads[self.synth[0].index]
        self.assertEqual({field: payload[field] for field in FIELD_NAMES}, {
            "tiles_left": 57, "honba": 3, "kyotaku": 2, "round_index": 5,
            "scores": [25000, 24000, 12000, 30000],
            "tiles_seen": [t % 5 for t in range(TILES)],
            "at_turn": 4,
        })

    def test_metadata_and_contract_accept_the_free_provider(self):
        from jiangpu import contract
        payload = self.synth_payloads[self.synth[0].index]
        self.assertEqual(payload["provider"], "context")
        self.assertEqual(payload["provider_version"], 1)
        self.assertIsNone(payload["checkpoint_sha256"])  # NO_CHECKPOINT_PROVIDERS 白名单
        self.assertEqual(payload["unit"], "mixed")
        self.assertEqual(payload["unit_by_field"], {
            "tiles_left": "tiles", "honba": "count", "kyotaku": "count", "scores": "points",
            "round_index": "index", "tiles_seen": "count", "at_turn": "turn"})
        self.assertIs(payload["no_hidden_info"], True)
        self.assertEqual(payload["basis"], "public_info_predictor")
        errors = []
        contract._check_provider(errors, "decisions[0].metrics.context", "context", payload)
        self.assertEqual(errors, [])

    def test_no_reveal_guards(self):
        for payload in self.synth_payloads.values():
            self.assertEqual(set(payload), ALLOWED_KEYS)
            self.assertNotIn(FORBIDDEN_FIELD, payload)
        self.assertNotIn("exp_ex1_labels", PROVIDER_SRC.read_text(encoding="utf-8"))

    def test_layout_failures_are_loud(self):
        with self.assertRaises(KeyError):
            self._compute_with_layout(lambda top: top.pop("tiles_left"))
        payloads = self._compute_with_layout(
            lambda top: top.update({"tiles_seen": {**top["tiles_seen"], "count": 2}}))
        payload = payloads[self.synth[0].index]
        self.assertTrue(payload["unavailable"])
        self.assertTrue(payload["reason"])
        for key in META_KEYS:
            self.assertIn(key, payload)

    def test_report_sample_matches_obs_slices_bit_for_bit(self):
        if not self.sample:
            self.skipTest("report_sub corpus not materialized")
        from jiangpu.providers import context
        n = 0
        for _entry, decisions in self.sample:
            payloads = context.compute(decisions)
            for dec in decisions:
                payload = payloads[dec.index]
                self.assertFalse(payload.get("unavailable"), payload.get("reason"))
                self.assertEqual({f: payload[f] for f in FIELD_NAMES}, _expected_fields(dec, self.spans))
                self.assertEqual(set(payload), ALLOWED_KEYS)
                n += 1
        self.assertGreaterEqual(n, 1000)

    def test_scores_and_round_match_events_at_kyoku_start(self):
        if not self.sample:
            self.skipTest("report_sub corpus not materialized")
        n_kyoku = 0
        for _entry, decisions in self.sample:
            seen = set()
            for dec in decisions:
                if dec.at_kyoku in seen:
                    continue
                seen.add(dec.at_kyoku)
                expected = _expected_fields(dec, self.spans)
                rel_scores = [int(dec.public["scores"][(dec.player_id + k) % 4]) for k in range(4)]
                self.assertEqual(expected["scores"], rel_scores, f"kyoku {dec.at_kyoku} scores")
                self.assertEqual(expected["honba"], dec.public["honba"])
                self.assertEqual(expected["kyotaku"], dec.public["kyotaku"])
                ev_round = (0 if dec.public["bakaze"] == "E" else 1) * 4 + int(dec.public["kyoku"]) - 1
                self.assertEqual(expected["round_index"], ev_round)
                n_kyoku += 1
        self.assertGreaterEqual(n_kyoku, 100)

    def test_artifact_records_zero_mismatches_and_prereg_hash(self):
        if self.artifact is None:
            self.skipTest("report_sub corpus not materialized")
        loaded = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            (loaded["schema"], loaded["prereg"]["sha256"], loaded["sample"]["n_unavailable"],
             loaded["coverage"], loaded["verdict"]),
            ("ex1-context-validity/v1", self.prereg_sha, 0, 1.0, "delivery"))
        self.assertGreaterEqual(loaded["sample"]["n_decisions"], 1000)
        for field in FIELD_NAMES:
            self.assertEqual(loaded["obs_slice_mismatches"][field],
                             {"n": loaded["sample"]["n_decisions"], "mismatches": 0})
        self.assertEqual(loaded["event_cross_check"]["fields"]["scores"]["mismatches"], 0)
        self.assertGreaterEqual(loaded["event_cross_check"]["fields"]["scores"]["n"], 100)
        self.assertIs(loaded["provider"]["no_hidden_info"], True)
        self.assertEqual(loaded["provider"]["basis"], "public_info_predictor")
        self.assertIsNone(loaded["provider"]["checkpoint_sha256"])
        self.assertTrue(loaded["no_concealed_hand_fields"])


if __name__ == "__main__":
    unittest.main()
