#!/usr/bin/env python3
"""PR-46B arena tests: output guard, POV safety, native context, real CPU smoke.

The synthetic tests are fast and always run.  The native tests load the real M5D-R0
checkpoint on CPU, run four complete sets of anchor self-control (16 games), one
repeat-determinism set and one one-layer RF bridge set.  They write only under
temporary directories; no frozen run/dataset/checkpoint path is touched.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.eval import m5d_arena as A  # noqa: E402
from riichiformer.eval import mortal_anchor as M  # noqa: E402
from riichiformer.eval import relative_pt as R  # noqa: E402
from riichiformer.schemas.action_v3 import ActionPhase  # noqa: E402
from riichiformer.schemas.event_v3 import (  # noqa: E402
    AnkanEvent,
    StartKyokuEvent,
    TileRef,
    TsumoEvent,
    UNKNOWN_TILE_KIND,
    tile_from_text,
)
from riichiformer.selfplay.tiles import tile_text  # noqa: E402

CHECKPOINT = Path(M.DEFAULT_CHECKPOINT_PATH)
NATIVE = CHECKPOINT.is_file()
SMOKE_KEY = 0x524636410000A001
SMOKE_WINDOW = 220_000
DETERMINISM_KEY = 0x524636410000D001
DETERMINISM_WINDOW = 221_000
BRIDGE_KEY = 0x524636410000F001
BRIDGE_WINDOW = 222_000


def _native_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    release = str(REPO_ROOT / "vendor" / "mortal" / "target" / "release")
    env["PYTHONPATH"] = release + (os.pathsep + env["PYTHONPATH"]
                                   if env.get("PYTHONPATH") else "")
    return env


# ---------------------------------------------------------------------------
# output guard
# ---------------------------------------------------------------------------
class OutputGuardTests(unittest.TestCase):
    def test_no_old_frozen_run_write(self):
        for path in (
            REPO_ROOT / "runs" / "1v3" / "x",
            REPO_ROOT / "runs" / "variance" / "x",
            REPO_ROOT / "checkpoints" / "x",
            REPO_ROOT / "datasets" / "x",
            REPO_ROOT / "vendor" / "x",
            REPO_ROOT,
        ):
            with self.subTest(path=path), self.assertRaises(A.M5DArenaError):
                A.guard_output_dir(path)
        with self.assertRaises(A.M5DArenaError):
            A.run_arena(seed_key=1, seed_count=1, candidate="mortal",
                        out_dir=REPO_ROOT / "runs" / "1v3" / "forbidden")
        allowed = Path(tempfile.mkdtemp(prefix="rf_m5d_guard_")) / "run"
        self.assertEqual(A.guard_output_dir(allowed), allowed.resolve())

    def test_log_directory_must_be_fresh(self):
        base = Path(tempfile.mkdtemp(prefix="rf_m5d_log_"))
        fresh = base / "games"
        fresh.mkdir()
        (fresh / "stale.json.gz").write_bytes(b"x")
        with self.assertRaises(A.M5DArenaError):
            A.run_arena(seed_key=1, seed_count=1, candidate="mortal",
                        out_dir=base / "out", log_dir=fresh)


# ---------------------------------------------------------------------------
# POV safety
# ---------------------------------------------------------------------------
def _synthetic_hands() -> list[list[TileRef]]:
    # Seat 0 and seat 1 have disjoint kind ranges so a concealed opponent tile is
    # provably absent from the POV stream.
    ranges = (
        range(0, 13),
        range(13, 26),
        tuple(range(26, 34)) + tuple(range(0, 5)),
        tuple(range(5, 18)),
    )
    return [[TileRef(kind) for kind in kinds] for kinds in ranges]


def _synthetic_events(hands) -> list[dict]:
    texts = [[tile_text(tile.kind, tile.aka) for tile in hand] for hand in hands]
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"], "seed": [1, 2]},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1m",
         "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
         "scores": [25000, 25000, 25000, 25000], "tehais": texts},
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "tsumo", "actor": 2, "pai": "7m"},
        {"type": "dahai", "actor": 2, "pai": "7m", "tsumogiri": True},
    ]


class PovSafetyTests(unittest.TestCase):
    def test_no_hidden_tile_access(self):
        hands = _synthetic_hands()
        events = _synthetic_events(hands)
        serialized = json.dumps(events, ensure_ascii=False)
        stream = A.PovDecisionStream(0)
        projected, meta = stream.update(serialized)
        start = next(event for event in projected if isinstance(event, StartKyokuEvent))
        self.assertEqual(start.own_haipai, tuple(hands[0]))
        public_tiles = {"1m", "5m", "6m", "7m"}  # dora/tsumo/dahai are public
        opponent_only = [tile for tile in hands[1]
                         if tile not in hands[0]
                         and tile_text(tile.kind, tile.aka) not in public_tiles]
        self.assertTrue(opponent_only)
        opponent_text = tile_text(opponent_only[0].kind, opponent_only[0].aka)
        # The opponent's concealed hand must not survive projection.
        payload = json.dumps([event.to_dict() for event in projected], ensure_ascii=False)
        self.assertNotIn(opponent_text, payload)
        for event in projected:
            if isinstance(event, TsumoEvent) and event.actor_rel != 0:
                self.assertEqual(event.tile.kind, UNKNOWN_TILE_KIND)
            if isinstance(event, AnkanEvent) and event.actor_rel != 0:
                self.assertTrue(all(tile.kind == UNKNOWN_TILE_KIND
                                    for tile in event.consumed))
        self.assertEqual(meta.pov, 0)

        mask = [False] * 46
        mask[0] = True
        decision, _ = stream.build_decision(
            serialized, legal_mask=mask, phase=ActionPhase.SELF_TURN)
        decision.assert_forward_safe()
        self.assertEqual(decision.before_event_idx, len(decision.events))
        self.assertNotIn(opponent_text, json.dumps(decision.to_forward_dict(),
                                                   ensure_ascii=False))

    def test_pov_stream_detects_rewritten_prefix(self):
        hands = _synthetic_hands()
        events = _synthetic_events(hands)
        stream = A.PovDecisionStream(0)
        stream.update(json.dumps(events))
        broken = json.loads(json.dumps(events))
        broken[1]["scores"] = [24000, 25000, 25000, 26000]
        with self.assertRaises(A.BridgeError):
            stream.update(json.dumps(broken))


# ---------------------------------------------------------------------------
# native action context (mocked states)
# ---------------------------------------------------------------------------
class _Cans:
    def __init__(self, **values):
        defaults = {
            "can_discard": True, "can_pass": False, "can_daiminkan": False,
            "can_ankan": False, "can_kakan": False, "can_riichi": False,
            "can_tsumo_agari": False, "can_ron_agari": False,
            "can_ryukyoku": False, "target_actor": 3,
        }
        defaults.update(values)
        self.__dict__.update(defaults)


class _MockState:
    def __init__(self, tiles, *, mask, cans=None, last_kawa="5m", last_tsumo=None,
                 ankan=(), kakan=()):
        counts = [0] * 34
        akas = [0, 0, 0]
        for tile in tiles:
            counts[tile.kind] += 1
            if tile.aka is not None:
                akas[tile.aka] += 1
        self.tehai = counts
        self.akas_in_hand = akas
        self.last_cans = cans or _Cans()
        self._mask = tuple(bool(value) for value in mask)
        self._last_kawa = last_kawa
        self._last_tsumo = last_tsumo
        self._ankan = list(ankan)
        self._kakan = list(kakan)
        self.validated = []

    def encode_obs(self, version, kan):
        return None, self._mask

    def last_kawa_tile(self):
        return self._last_kawa

    def last_self_tsumo(self):
        return self._last_tsumo

    def ankan_candidates(self):
        return list(self._ankan)

    def kakan_candidates(self):
        return list(self._kakan)

    def validate_reaction(self, text):
        json.loads(text)
        self.validated.append(text)


def _mask_with(*actions: int) -> list[bool]:
    mask = [False] * 46
    for action in actions:
        mask[action] = True
    return mask


def _hand(*texts: str) -> list[TileRef]:
    return [tile_from_text(text) for text in texts]


class NativeContextTests(unittest.TestCase):
    def test_riichi_kan(self):
        base = _hand("1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m",
                     "1p", "2p", "3p", "4p", "6p", "7p")
        riichi_state = _MockState(base, mask=_mask_with(37))
        event, context = A.decode_native_action(37, riichi_state, seat=0)
        self.assertEqual(event, {"type": "reach", "actor": 0})
        self.assertIs(context.phase, ActionPhase.SELF_TURN)
        self.assertEqual(len(riichi_state.validated), 1)

        ankan_hand = _hand("5m", "5m", "5m", "5m", "1m", "2m", "3m", "4m",
                           "6m", "7m", "8m", "9m", "1p", "2p")
        ankan_state = _MockState(
            ankan_hand, mask=_mask_with(42), ankan=("5m",),
            cans=_Cans(can_ankan=True))
        event, _ = A.decode_native_action(42, ankan_state, seat=0)
        self.assertEqual(event["type"], "ankan")
        self.assertEqual(event["consumed"], ["5m"] * 4)
        self.assertEqual(len(ankan_state.validated), 1)

        kakan_hand = _hand("5m", "1m", "2m", "3m", "4m", "6m", "7m", "8m",
                           "9m", "1p", "2p", "3p", "4p", "6p")
        kakan_state = _MockState(
            kakan_hand, mask=_mask_with(42), kakan=("5m",),
            cans=_Cans(can_kakan=True))
        event, _ = A.decode_native_action(42, kakan_state, seat=0)
        self.assertEqual(event["type"], "kakan")
        self.assertIn("consumed", event)
        self.assertEqual(len(event["consumed"]), 3)

        daiminkan_hand = _hand("5m", "5m", "5m", "1m", "2m", "3m", "4m",
                               "6m", "7m", "8m", "9m", "1p", "2p", "3p")
        daiminkan_state = _MockState(
            daiminkan_hand, mask=_mask_with(42), last_kawa="5m",
            cans=_Cans(can_discard=False, can_pass=True, can_daiminkan=True,
                       target_actor=3))
        event, context = A.decode_native_action(42, daiminkan_state, seat=0)
        self.assertEqual(event["type"], "daiminkan")
        self.assertEqual(event["target"], 3)
        self.assertIs(context.phase, ActionPhase.CALL_RESPONSE)

    def test_multi_kan_picks_first_candidate(self):
        hand = _hand("5m", "5m", "5m", "5m", "6p", "6p", "6p", "6p",
                     "1m", "2m", "3m", "4m", "6m", "7m")
        state = _MockState(hand, mask=_mask_with(42), ankan=("5m", "6p"),
                           cans=_Cans(can_ankan=True))
        event, _ = A.decode_native_action(42, state, seat=0)
        self.assertEqual(event["consumed"], ["5m"] * 4)

    def test_false_tsumogiri_regression(self):
        hand = _hand("1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m",
                     "1p", "2p", "3p", "4p", "6p", "7p")
        state = _MockState(hand, mask=_mask_with(0), last_tsumo=None)
        event, _ = A.decode_native_action(0, state, seat=0)
        self.assertEqual(event["type"], "dahai")
        self.assertFalse(event["tsumogiri"])
        state = _MockState(hand, mask=_mask_with(0), last_tsumo="1m")
        event, _ = A.decode_native_action(0, state, seat=0)
        self.assertTrue(event["tsumogiri"])

    def test_illegal_action_is_not_retried(self):
        hand = _hand("1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m",
                     "1p", "2p", "3p", "4p", "6p", "7p")
        state = _MockState(hand, mask=_mask_with(0))
        with self.assertRaises(A.BridgeError):
            A.decode_native_action(37, state, seat=0)
        self.assertEqual(state.validated, [])


# ---------------------------------------------------------------------------
# RF bridge with a tiny deterministic policy (no external checkpoint)
# ---------------------------------------------------------------------------
class _TinyBackend:
    name = "tiny-rf"

    def __init__(self):
        import torch
        from riichiformer.models.policy import MVPConfig, Policy
        from riichiformer.train.checkpoint import config_hash

        config = MVPConfig(layers=1, hidden=64, heads=4, ffn=64,
                           state_hidden=64, max_positions=4096, name="tiny_bridge")
        torch.manual_seed(46)
        self.policy = Policy(config).eval()
        self.model_config = {field: getattr(config, field) for field in
                             config.field_names()}
        self.model_config_hash = config_hash(self.model_config)

    def act(self, decision, meta):
        import torch
        from riichiformer.data.encode import encode_decision

        encoded = encode_decision(decision, meta, max_events=4096)
        with torch.inference_mode():
            output = self.policy(encoded.to_model_batch())
        return int(output.masked_logits.argmax(dim=-1)[0])

    def as_dict(self):
        return {
            "mode": "rf",
            "checkpoint_path": "<tiny-in-memory>",
            "checkpoint_sha256": None,
            "steps": 0,
            "model_config": self.model_config,
            "model_config_hash": self.model_config_hash,
            "parameter_count": sum(p.numel() for p in self.policy.parameters()),
            "use_ema": False,
            "inference_mode": "deterministic_argmax",
            "device": "cpu",
            "max_events": 4096,
            "test_only": True,
        }


# ---------------------------------------------------------------------------
# real native CPU smoke
# ---------------------------------------------------------------------------
@unittest.skipUnless(NATIVE, f"M5D-R0 checkpoint not found at {CHECKPOINT}")
class NativeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="rf_m5d_native_")
        cls.run4 = A.run_arena(
            seed_key=SMOKE_KEY, seed_start=SMOKE_WINDOW, seed_count=4,
            candidate="mortal", out_dir=Path(cls.tmp) / "selfcontrol4",
        )

    def test_4seat_rotation(self):
        run = self.run4
        self.assertEqual(run.seed_count, 4)
        self.assertEqual(run.games, 16)
        self.assertEqual(len(run.sets), 4)
        for index, item in enumerate(run.sets):
            self.assertEqual(item.wall_seed, run.seed_start + index)
            self.assertEqual([game.rotation for game in item.games], [0, 1, 2, 3])
            self.assertEqual(sorted(game.candidate_seat for game in item.games),
                             [0, 1, 2, 3])
            self.assertEqual(len({game.game_id for game in item.games}), 4)
        self.assertEqual(sum(run.rank_counts), 16)
        self.assertEqual(sum(run.report["rank_hist"]), 16)

    def test_anchor_self_play_is_balanced(self):
        run = self.run4
        report = run.report
        self.assertEqual(report["n_complete_sets"], 4)
        self.assertEqual(report["n_games"], 16)
        self.assertLessEqual(abs(report["delta_pt5"]), 5.0,
                             f"self-control delta drifted: {report['delta_pt5']}")
        self.assertAlmostEqual(report["avg_rank"], 2.5, delta=0.1)
        self.assertEqual(report["score_policy_id"], "TOKUJOU_5D_HANCHAN")
        self.assertTrue(run.self_control_only)
        self.assertEqual(run.candidate["rf_candidate"], "NOT_RUN")
        # Every complete set has exactly the four rotations and four games.
        self.assertEqual(report["seed_key"], A.seed_key_hex(SMOKE_KEY))
        self.assertEqual(report["seed_window"], [SMOKE_WINDOW, SMOKE_WINDOW + 4])
        # Replay/manifest files exist and the log hashes match.
        out = Path(self.tmp) / "selfcontrol4"
        manifest = json.loads((out / "run.json").read_text())
        self.assertEqual(manifest["games"], 16)
        self.assertEqual(manifest["seed_key"], A.seed_key_hex(SMOKE_KEY))
        rows = [json.loads(line) for line in
                (out / "games.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(sum(row["scores"]), 100000)
            digest = hashlib.sha256(Path(row["log_path"]).read_bytes()).hexdigest()
            self.assertEqual(digest, row["log_sha256"])
            with gzip.open(row["log_path"], "rt", encoding="utf-8") as handle:
                self.assertIn("end_game", handle.read())

    @staticmethod
    def _action_trace_sha(path: str) -> str:
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                event.pop("meta", None)  # eval timings/q metadata are not decisions
                rows.append(event)
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def test_repeated_seed_determinism(self):
        out = Path(self.tmp)
        first = A.run_arena(
            seed_key=DETERMINISM_KEY, seed_start=DETERMINISM_WINDOW, seed_count=1,
            candidate="mortal", out_dir=out / "repeat_a")
        second = A.run_arena(
            seed_key=DETERMINISM_KEY, seed_start=DETERMINISM_WINDOW, seed_count=1,
            candidate="mortal", out_dir=out / "repeat_b")
        self.assertEqual(first.challenger_scores, second.challenger_scores)
        self.assertEqual(first.rank_counts, second.rank_counts)
        self.assertEqual(first.report["delta_pt5"], second.report["delta_pt5"])
        # The gzip files embed per-decision eval-time metadata, so the byte hashes
        # legitimately differ; the decision/action trace must not.
        self.assertEqual(
            [self._action_trace_sha(item["path"]) for item in first.log_files],
            [self._action_trace_sha(item["path"]) for item in second.log_files])

    def test_native_action_roundtrip(self):
        run = A.run_arena(
            seed_key=BRIDGE_KEY, seed_start=BRIDGE_WINDOW, seed_count=1,
            candidate="rf", candidate_backend=_TinyBackend(),
            out_dir=Path(self.tmp) / "tiny_rf",
        )
        self.assertFalse(run.self_control_only)
        self.assertEqual(run.games, 4)
        self.assertGreater(run.candidate.get("decisions", 0), 0)
        self.assertEqual(run.candidate.get("illegal"), 0)
        self.assertEqual(run.candidate.get("fallback"), 0)
        for row in run.game_rows:
            self.assertEqual(sum(row["scores"]), 100000)
            self.assertTrue(Path(row["log_path"]).is_file())
        self.assertEqual(run.report["n_complete_sets"], 1)


# ---------------------------------------------------------------------------
# CLI / manifest
# ---------------------------------------------------------------------------
class CliTests(unittest.TestCase):
    def test_cli_plan_and_manifest(self):
        out = Path(tempfile.mkdtemp(prefix="rf_m5d_cli_")) / "plan"
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "rf_eval_m5d.py"),
             "--plan-only", "--sets", "4", "--seed-key", "0x5246364100009999",
             "--seed-start", "230000", "--out", str(out)],
            cwd=str(REPO_ROOT), env=_native_env(), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["plan"])
        self.assertEqual(payload["seed_key"], "5246364100009999")
        self.assertEqual(payload["seed_window"], [230000, 230004])
        self.assertEqual(payload["games"], 16)
        self.assertEqual(payload["candidate"]["mode"], "mortal_self_control")
        self.assertEqual(payload["anchor_sha256"], M.DEFAULT_CHECKPOINT_SHA256)
        self.assertEqual(payload["bootstrap"]["resamples"], 2000)

    def test_cli_rejects_bad_output(self):
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "rf_eval_m5d.py"),
             "--plan-only", "--out", str(REPO_ROOT / "runs" / "1v3" / "x")],
            cwd=str(REPO_ROOT), env=_native_env(), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 3)
        self.assertIn("frozen path", completed.stderr)


if __name__ == "__main__":
    unittest.main()
