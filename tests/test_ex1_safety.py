#!/usr/bin/env python3
"""t13: tests for the public-info safety heuristic provider (`heuristic_score`, never a probability).

Covers: the mandated layout spans, the synthetic genbutsu/suji/visible/unseen/dora decode, the
`subject_tile`/`subject_score` contract, the naming guard (no `prob`/`deal_in`/`danger` keys),
fail-closed degradation on a malformed layout, the no-oracle import graph, and a real-record
smoke test on the frozen fixture. Writes `runs/exp/EX1/safety_provider.json` as the run artifact
and pins the PREREG hash (start + end, drift is fatal).
"""
from __future__ import annotations

# allow: SIZE_OK — the plan pins this exact single test filename (t13 acceptance runs
# `-p 'test_ex1_safety.py'`) and the commit is restricted to it; splitting would break both.

import hashlib
import json
import math
import pathlib
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from exp_ex1_audit import PREREG_PATH, assert_prereg_unchanged, load_prereg  # noqa: E402
from jiangpu import layout  # noqa: E402
from jiangpu.providers import safety  # noqa: E402
from jiangpu.record import FROZEN_FIXTURE  # noqa: E402

LAYOUT_PATH = REPO_ROOT / "runs/analysis/obs_layout.json"
OUT_PATH = REPO_ROOT / "runs/exp/EX1/safety_provider.json"
FROZEN_RECORD = pathlib.Path(FROZEN_FIXTURE)  # single source of truth: jiangpu.record
BANNED_KEY_SUBSTRINGS = ("prob", "deal_in", "danger")
N_TILES = 34
N_PLANES = 1012
DISCARD_MAX = 36
EXPECTED_SPANS = {
    "kawa_overview_self": (723, 7), "kawa_overview_shimocha": (730, 7),
    "kawa_overview_toimen": (737, 7), "kawa_overview_kamicha": (744, 7),
    "riichi_sutehai_shimocha": (845, 3), "riichi_sutehai_toimen": (848, 3),
    "riichi_sutehai_kamicha": (851, 3), "riichi_declared_opponents": (854, 3),
    "riichi_accepted_opponents": (857, 3), "tiles_seen": (835, 1),
    "dora_indicators": (28, 7),
}


def banned_key_hits(obj, where="payload"):
    """Recursively collect the paths of dict keys containing a banned substring."""
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if any(s in str(key) for s in BANNED_KEY_SUBSTRINGS):
                hits.append(f"{where}.{key}")
            hits.extend(banned_key_hits(value, f"{where}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            hits.extend(banned_key_hits(value, f"{where}[{i}]"))
    return hits


def blank_obs():
    return np.zeros((N_PLANES, N_TILES), dtype=np.float32)


def span(name):
    return layout.span(layout.load_layout(), name)


def set_river(obs, block, tile, count=1):
    off, _ = span(block)
    for c in range(count):
        obs[off + c][tile] = 1.0


def set_sutehai(obs, block, tile):
    obs[span(block)[0]][tile] = 1.0


def set_flag(obs, block, opponent):
    obs[span(block)[0] + opponent][0] = 1.0


def set_seen(obs, tile, count):
    obs[span("tiles_seen")[0]][tile] = count / 4.0


def set_indicator(obs, tile, count=1):
    off, _ = span("dora_indicators")
    for c in range(count):
        obs[off + c][tile] = 1.0


def fake_decision(index, obs, logged_action):
    return SimpleNamespace(index=index, obs=obs, logged_action=logged_action)


def synthetic_decision(index=0):
    """shimocha riichi; river {4m, 5m}; declaration 4m; dora indicator 1m (=> 2m is dora)."""
    obs = blank_obs()
    set_flag(obs, "riichi_declared_opponents", 0)
    set_flag(obs, "riichi_accepted_opponents", 0)
    set_river(obs, "kawa_overview_shimocha", 3)   # 4m: genbutsu, suji {1m, 7m}
    set_river(obs, "kawa_overview_shimocha", 4)   # 5m: genbutsu, suji {2m, 8m}
    set_sutehai(obs, "riichi_sutehai_shimocha", 3)
    set_seen(obs, 1, 1)                           # 2m visible x1
    set_seen(obs, 3, 2)                           # 4m visible x2
    set_seen(obs, 4, 1)                           # 5m visible x1
    set_seen(obs, 5, 4)                           # 6m visible x4 (all copies located)
    set_indicator(obs, 0)                         # 1m indicator => 2m dora
    return fake_decision(index, obs, 3)


def load_frozen_decisions():
    from eval_report import SEAT_OF_SPLIT
    from jiangpu.record import read_record
    from jiangpu.replay import iter_decisions
    record = read_record(str(FROZEN_RECORD))
    return iter_decisions(record, pov=SEAT_OF_SPLIT[record.split])


def payload_digest(payloads):
    blob = json.dumps(payloads, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class SafetyProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prereg_sha = load_prereg(PREREG_PATH)
        missing = [p for p in (LAYOUT_PATH, FROZEN_RECORD) if not p.exists()]
        if missing:
            raise unittest.SkipTest("safety fixtures missing: " + ", ".join(str(p) for p in missing))
        cls.decisions = load_frozen_decisions()
        cls.payloads = safety.compute(cls.decisions)
        cls.artifact = cls._build_artifact()
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(cls.artifact, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        assert_prereg_unchanged(PREREG_PATH, cls.prereg_sha)

    @classmethod
    def _build_artifact(cls):
        available = [p for p in cls.payloads.values() if not p.get("unavailable")]
        sample_index = cls.decisions[0].index
        return {
            "schema": "ex1-safety-provider/v1",
            "todo": "t13",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prereg": {"path": str(PREREG_PATH.relative_to(REPO_ROOT)), "sha256": cls.prereg_sha},
            "layout": {
                "path": "runs/analysis/obs_layout.json",
                "spans": {name: list(span(name)) for name in EXPECTED_SPANS},
            },
            "record": {
                "path": str(FROZEN_RECORD.relative_to(REPO_ROOT)),
                "sha256": hashlib.sha256(FROZEN_RECORD.read_bytes()).hexdigest(),
                "n_decisions": len(cls.decisions),
                "n_available": len(available),
                "coverage": len(available) / len(cls.decisions),
            },
            "naming_guard": {
                "banned_key_substrings": list(BANNED_KEY_SUBSTRINGS),
                "n_payloads_scanned": len(cls.payloads),
                "violations": sum(len(banned_key_hits(p)) for p in cls.payloads.values()),
            },
            "payloads_sha256": payload_digest(cls.payloads),
            "subject_convention": {
                "discard_action_max": DISCARD_MAX,
                "subject_tile": "deaka(logged_action) for discards 0..36; null for riichi/melds",
            },
            "score_scheme": {
                "formula": ("1000*[visible==4] + 100*genbutsu_count + 10*suji_count "
                            "+ visible - dora"),
                "unit": "heuristic_score",
                "interpretation": "unitless tiers, higher = safer; never a probability",
            },
            "sample": cls.payloads[sample_index],
        }

    def test_mandated_layout_spans(self):
        for name, expected in EXPECTED_SPANS.items():
            self.assertEqual(span(name), expected, name)
        for name, count in safety.REQUIRED_CHANNELS.items():
            self.assertEqual(span(name)[1], count, name)

    def test_synthetic_genbutsu_suji_visible_unseen_dora(self):
        payload = safety.compute([synthetic_decision()])[0]
        comp = payload["components"]
        self.assertEqual(payload["subject_tile"], 3)
        self.assertEqual(payload["subject_score"], payload["tile_score"][3])
        self.assertEqual(comp["genbutsu_opponents"][3], 1)
        self.assertEqual(comp["genbutsu_opponents"][4], 1)
        self.assertEqual(comp["genbutsu_opponents"][5], 0)
        for t in (0, 1, 6, 7):                        # 1m, 2m, 7m, 8m are suji of {4m, 5m}
            self.assertEqual(comp["suji_opponents"][t], 1, t)
        for t in (2, 5, 8):                           # 3m, 6m, 9m are not
            self.assertEqual(comp["suji_opponents"][t], 0, t)
        self.assertEqual(comp["visible_copies"][3:6], [2, 1, 4])
        self.assertEqual(comp["unseen_copies"][5], 0)
        self.assertTrue(comp["dora_flag"][1])
        self.assertEqual(comp["dora_flag"][2], False)
        scores = payload["tile_score"]
        for t, expected in ((0, 10.0), (1, 10.0), (2, 0.0), (3, 102.0), (4, 101.0),
                            (5, 1004.0), (6, 10.0), (7, 10.0), (8, 0.0)):
            self.assertEqual(scores[t], expected, t)
        self.assertEqual(payload["threat"]["n_riichi"], 1)
        self.assertEqual(payload["threat"]["declaration_tiles"], [3, None, None])

    def test_genbutsu_bits_are_per_opponent(self):
        obs = blank_obs()
        set_flag(obs, "riichi_declared_opponents", 2)   # kamicha only
        set_river(obs, "kawa_overview_kamicha", 5)      # 6m: genbutsu for bit 2
        out = safety.compute([fake_decision(0, obs, 5)])[0]
        comp = out["components"]
        self.assertEqual(comp["genbutsu_opponents"][5], 4)
        self.assertEqual(comp["suji_opponents"][2], 4)  # 3m suji of 6m
        self.assertEqual(comp["suji_opponents"][8], 4)  # 9m suji of 6m
        self.assertEqual(comp["genbutsu_opponents"][2], 0)
        self.assertEqual(out["tile_score"][5], 100.0)
        self.assertEqual(out["tile_score"][2], 10.0)

    def test_visible_four_dominates_and_no_riichi_scores_are_visible_only(self):
        payload = safety.compute([synthetic_decision()])[0]
        scores = payload["tile_score"]
        self.assertEqual(scores[5], safety.SCORE_VISIBLE_FOUR + 4.0)
        self.assertEqual(max(scores), scores[5])
        quiet = blank_obs()
        set_seen(quiet, 5, 4)
        out = safety.compute([fake_decision(0, quiet, 5)])[0]
        self.assertEqual(out["tile_score"][5], safety.SCORE_VISIBLE_FOUR + 4.0)
        self.assertEqual(out["tile_score"][4], 0.0)
        self.assertEqual(out["threat"]["n_riichi"], 0)

    def test_subject_tile_follows_logged_action(self):
        decisions = [
            synthetic_decision(0),
            fake_decision(1, blank_obs(), 35),          # aka 5p -> deaka 13
            fake_decision(2, blank_obs(), 37),          # riichi -> no subject tile
            fake_decision(3, blank_obs(), 40),          # meld -> no subject tile
        ]
        out = safety.compute(decisions)
        self.assertEqual(out[0]["subject_tile"], 3)
        self.assertEqual(out[1]["subject_tile"], 13)
        self.assertEqual(out[1]["subject_score"], out[1]["tile_score"][13])
        self.assertIsNone(out[2]["subject_tile"])
        self.assertIsNone(out[2]["subject_score"])
        self.assertIsNone(out[3]["subject_tile"])

    def test_payload_contract_on_the_frozen_record(self):
        self.assertEqual(len(self.payloads), len(self.decisions))
        self.assertTrue(self.decisions)
        for dec in self.decisions:
            payload = self.payloads[dec.index]
            self.assertIsNotNone(payload, dec.index)
            self.assertNotIn("unavailable", payload, dec.index)
            self.assertEqual(payload["provider"], "safety")
            self.assertEqual(payload["unit"], "heuristic_score")
            self.assertEqual(payload["basis"], "public_info_predictor")
            self.assertIs(payload["no_hidden_info"], True)
            self.assertIsNone(payload["checkpoint_sha256"])
            self.assertEqual(len(payload["tile_score"]), N_TILES)
            self.assertEqual(banned_key_hits(payload), [])
            subject = payload["subject_tile"]
            if dec.logged_action <= DISCARD_MAX:
                self.assertEqual(subject, safety.deaka(dec.logged_action), dec.index)
                self.assertEqual(payload["subject_score"], payload["tile_score"][subject])
            else:
                self.assertIsNone(subject, dec.index)
                self.assertIsNone(payload["subject_score"], dec.index)

    def test_frozen_record_invariants(self):
        self.assertGreater(len(self.decisions), 0)
        n_riichi_decisions = 0
        max_score = -math.inf
        for payload in self.payloads.values():
            comp = payload["components"]
            self.assertEqual(len(comp["genbutsu_opponents"]), N_TILES)
            for t in range(N_TILES):
                gen = bin(comp["genbutsu_opponents"][t]).count("1")
                suji = bin(comp["suji_opponents"][t]).count("1")
                visible = comp["visible_copies"][t]
                self.assertEqual(comp["genbutsu_opponents"][t] & comp["suji_opponents"][t], 0, t)
                self.assertLessEqual(gen, visible, t)              # each genbutsu is a witnessed tile
                self.assertLessEqual(suji, 3 - gen, t)
                self.assertEqual(visible + comp["unseen_copies"][t], 4, t)
                self.assertIsInstance(comp["dora_flag"][t], bool)
                self.assertTrue(math.isfinite(payload["tile_score"][t]))
                max_score = max(max_score, payload["tile_score"][t])
            n_riichi_decisions += payload["threat"]["n_riichi"] > 0
        self.assertGreater(n_riichi_decisions, 0, "frozen record must contain riichi opponents")
        self.assertGreaterEqual(max_score, 100.0)                  # genbutsu tier reached
        self.assertGreater(max_score, 1.0)                         # not a [0,1] probability scale

    def test_naming_guard_rejects_probability_style_keys(self):
        for bad_key in ("opp_win_prob", "deal_in", "dangerous_tiles", "deal_in_prob"):
            self.assertNotEqual(banned_key_hits({bad_key: 1.0}), [], bad_key)
        self.assertNotEqual(banned_key_hits({"components": {"nested_deal_in": 1}}), [])
        self.assertEqual(banned_key_hits({"score": [{"genbutsu_opponents": 1}]}), [])

    def test_malformed_layout_fails_closed(self):
        decision = synthetic_decision(7)
        bad = {name: dict(block) for name, block in layout.load_layout().items()}
        bad["kawa_overview_self"]["count"] = 6
        with mock.patch.object(layout, "load_layout", lambda: bad):
            out = safety.compute([decision])
        self.assertIs(out[7]["unavailable"], True)
        self.assertIn("通道数应为 7", out[7]["reason"])
        self.assertEqual(out[7]["unit"], "heuristic_score")

    def test_missing_block_makes_span_lookup_fail_loud(self):
        decision = synthetic_decision()
        with mock.patch.object(layout, "load_layout", lambda: {}):
            with self.assertRaises(KeyError):
                safety.compute([decision])

    def test_provider_imports_without_oracle_tensors(self):
        code = (
            "import sys; sys.path.insert(0, 'scripts');"
            "import jiangpu.providers.safety as s;"
            "leaked = sorted(m for m in sys.modules"
            " if m in ('libriichi', 'torch', 'exp_ex1_labels')"
            " or m.startswith(('libriichi.', 'torch.')));"
            "print(leaked); assert not leaked, leaked;"
            "assert s.UNIT == 'heuristic_score'"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_artifact_round_trips_with_prereg_hash(self):
        loaded = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(loaded["prereg"]["sha256"], self.prereg_sha)
        self.assertEqual(loaded["record"]["n_decisions"], len(self.decisions))
        self.assertEqual(loaded["record"]["n_available"], len(self.payloads))
        self.assertEqual(loaded["record"]["coverage"], 1.0)
        self.assertEqual(loaded["naming_guard"]["violations"], 0)
        self.assertEqual(loaded["payloads_sha256"], payload_digest(self.payloads))
        self.assertEqual(loaded["layout"]["spans"], {k: list(v) for k, v in EXPECTED_SPANS.items()})
        self.assertEqual(loaded["sample"]["unit"], "heuristic_score")
        self.assertEqual(loaded["score_scheme"]["unit"], "heuristic_score")
        self.assertTrue(loaded["score_scheme"]["formula"])


if __name__ == "__main__":
    unittest.main()
