#!/usr/bin/env python3
"""PR-41 self-play tests: shanten, bot legality, provenance, seed pools, rollouts.

The procedural-fallback tests need no engine and no GPU.  The real-rules test uses the
pinned local ``libriichi`` arena when it is importable (see the acceptance command's
``PYTHONPATH``) and is skipped otherwise; it never touches the network, a dataset or a
checkpoint.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.selfplay import rollout as R  # noqa: E402
from riichiformer.selfplay import shanten as S  # noqa: E402
from riichiformer.selfplay.bots import (  # noqa: E402
    BOT_FACTORIES,
    DefenseBot,
    IllegalBotAction,
    LegalRandomBot,
    SimpleShantenUkeireBot,
    make_bot,
)
from riichiformer.selfplay.provenance import (  # noqa: E402
    RULESET_LIBRIICHI,
    RULESET_SYNTHETIC,
    ProvenanceError,
    generator_fingerprint,
    guard_out,
    ruleset_metadata,
    sha256_file,
    sha256_text,
    source_metadata,
)
from riichiformer.selfplay.seeds import (  # noqa: E402
    SeedConflict,
    SeedError,
    SeedLedger,
    seed_range,
)
from riichiformer.selfplay.tiles import tile_from_text  # noqa: E402
from riichiformer.selfplay.view import SeatView, ViewError  # noqa: E402
from riichiformer.schemas.action_v3 import ACTION_SPACE, ActionPhase  # noqa: E402
from riichiformer.schemas.event_v3 import TileRef  # noqa: E402

LIBRIICHI = R.libriichi_status()["available"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def counts_of(*texts: str) -> list[int]:
    counts = [0] * 34
    for text in texts:
        counts[tile_from_text(text).kind] += 1
    return counts


def akas_of(*texts: str) -> tuple[int, int, int]:
    akas = [0, 0, 0]
    for text in texts:
        tile = tile_from_text(text)
        if tile.aka is not None:
            akas[tile.aka] += 1
    return tuple(akas)


def mask_of(*actions: int) -> tuple[bool, ...]:
    mask = [False] * ACTION_SPACE
    for action in actions:
        mask[action] = True
    return tuple(mask)


def random_hand(seed: int, size: int = 14) -> list[int]:
    import random

    rng = random.Random(seed)
    pool = [kind for kind in range(34) for _ in range(4)]
    rng.shuffle(pool)
    counts = [0] * 34
    for kind in pool[:size]:
        counts[kind] += 1
    return counts


def view_for(counts, *, seat=0, mask=None, phase=ActionPhase.SELF_TURN,
             rivers=None, riichi=None, drawn=None, meld_count=0,
             ankan=(), kakan=(), daiminkan=None, scores=(25000,) * 4,
             source="synthetic", ruleset=RULESET_SYNTHETIC) -> SeatView:
    if mask is None:
        mask = mask_of(*range(34))
    return SeatView(
        seat=seat,
        hand_counts=tuple(counts),
        hand_akas=(0, 0, 0),
        drawn=TileRef(int(drawn)) if drawn is not None else None,
        legal_mask=tuple(mask),
        phase=phase,
        meld_count=meld_count,
        rivers=tuple(tuple(rivers) for rivers in (rivers or ((), (), (), ()))),
        riichi=tuple(riichi or (False, False, False, False)),
        scores=tuple(scores),
        ankan_candidates=tuple(ankan),
        kakan_candidates=tuple(kakan),
        daiminkan_tile=daiminkan,
        source=source,
        ruleset_id=ruleset,
    )


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="rf41-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# shanten
# ---------------------------------------------------------------------------
class ShantenTests(unittest.TestCase):
    def test_known_shanten(self):
        cases = [
            (("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
              "1p", "2p", "3p", "1s"), 0),
            (("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
              "1p", "2p", "4p", "1s"), 1),
            (("1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"), 0),
            (("1m", "1m", "2m", "2m", "3m", "3m", "4p", "4p", "5p", "5p",
              "6s", "6s", "7s"), 0),
            (("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
              "1p", "2p", "3p", "9s"), 0),
        ]
        for texts, expected in cases:
            with self.subTest(hand=texts):
                self.assertEqual(S.shanten(counts_of(*texts)), expected)

    def test_open_hand_shanten(self):
        # one open meld: 10 concealed tiles
        texts = ("5p", "6p", "7p", "5s", "6s", "7s", "2m", "3m", "4m", "9p")
        self.assertEqual(S.shanten(counts_of(*texts), open_melds=1), 0)

    def test_win_and_wait(self):
        win = counts_of("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                        "1p", "2p", "3p", "1s", "1s")
        self.assertTrue(S.is_winning_hand(win))
        tenpai = counts_of("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                           "1p", "2p", "3p", "1s")
        self.assertFalse(S.is_winning_hand(tenpai))
        self.assertEqual(S.waits(tenpai), (18,))
        self.assertEqual(S.shanten(tenpai), 0)

    def test_chiitoi_and_kokushi(self):
        chiitoi = counts_of("1m", "1m", "2m", "2m", "3m", "3m", "4p", "4p",
                            "5p", "5p", "6s", "6s", "7s")
        self.assertEqual(S.chiitoi_shanten(chiitoi), 0)
        self.assertEqual(S.shanten(chiitoi), 0)
        kokushi = counts_of("1m", "9m", "1p", "9p", "1s", "9s",
                            "E", "S", "W", "N", "P", "F", "C")
        self.assertEqual(S.kokushi_shanten(kokushi), 0)
        self.assertEqual(S.shanten(kokushi), 0)
        self.assertTrue(S.is_winning_hand(counts_of(
            "1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C", "C")))

    def test_invalid_shapes_fail_closed(self):
        with self.assertRaises(ValueError):
            S.shanten([0] * 34)
        with self.assertRaises(ValueError):
            S.shanten([5] + [0] * 33)
        with self.assertRaises(TypeError):
            S.shanten("not-a-hand")
        with self.assertRaises(ValueError):
            S.standard_shanten(counts_of("1m", "2m", "3m"), open_melds=0)

    def test_ukeire_counts_remaining_copies(self):
        hand = counts_of("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                         "1p", "2p", "3p", "1s")
        visible = list(hand)
        visible[18] = 2  # one 1s already visible publicly
        self.assertEqual(S.ukeire(hand, 0, visible), 2)
        self.assertEqual(S.shanten_14(counts_of(
            "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
            "1p", "2p", "3p", "1s", "1s")), -1)
        with self.assertRaises(ValueError):
            S.shanten_14(hand)  # 13 tiles is not a 3n+2 hand


# ---------------------------------------------------------------------------
# bots
# ---------------------------------------------------------------------------
class BotLegalityTests(unittest.TestCase):
    def test_every_bot_only_returns_masked_actions(self):
        for seed in range(40):
            counts = random_hand(seed)
            discards = [kind for kind in range(34) if counts[kind] > 0]
            mask = mask_of(*discards, 37, 44)
            view = view_for(counts, mask=mask, drawn=discards[0])
            for name, factory in BOT_FACTORIES.items():
                bot = factory(seed=seed)
                action = bot.act(view)
                self.assertIn(action.index, range(ACTION_SPACE))
                self.assertTrue(view.legal_mask[action.index],
                                f"{name} chose {action.index} outside the mask")

    def test_illegal_action_raises(self):
        class Broken(LegalRandomBot):
            name = "broken"

            def _decide(self, view):
                from riichiformer.selfplay.bots import BotAction

                return BotAction(45)  # 45 is not legal on a self turn

        counts = random_hand(1)
        view = view_for(counts, mask=mask_of(*[k for k in range(34) if counts[k] > 0]))
        with self.assertRaises(IllegalBotAction):
            Broken().act(view)

    def test_riichi_forced_discard(self):
        # after riichi acceptance only the drawn tile is legal
        counts = random_hand(3)
        drawn = max(range(34), key=lambda kind: counts[kind])
        viewed = view_for(counts, mask=mask_of(drawn), drawn=drawn)
        for bot in (SimpleShantenUkeireBot(0), DefenseBot(0), LegalRandomBot(0)):
            self.assertEqual(bot.act(viewed).index, drawn)

    def test_call_response_passes_or_wins(self):
        counts = random_hand(4, 13)
        pass_view = view_for(counts, mask=mask_of(45, 41), phase=ActionPhase.CALL_RESPONSE)
        self.assertEqual(SimpleShantenUkeireBot(0).act(pass_view).index, 45)
        self.assertEqual(DefenseBot(0).act(pass_view).index, 45)
        win_view = view_for(counts, mask=mask_of(45, 43), phase=ActionPhase.CALL_RESPONSE,
                            drawn=None)
        for bot in (SimpleShantenUkeireBot(0), DefenseBot(0)):
            self.assertEqual(bot.act(win_view).index, 43)

    def test_defense_prefers_genbutsu(self):
        counts = counts_of("1m", "1m", "5p", "9p", "3s", "9s", "E", "S", "W",
                           "N", "P", "F", "C", "2m")
        self.assertGreater(S.shanten_14(list(counts)), 0)
        rivers = ((), (TileRef(17),), (), ())  # opponent 1 discarded 9p
        mask = mask_of(*[kind for kind in range(34) if counts[kind] > 0])
        view = view_for(counts, mask=mask, rivers=rivers,
                        riichi=(False, True, False, False))
        action = DefenseBot(0).act(view)
        self.assertEqual(action.index, 17)
        self.assertIn("defense", action.label)

    def test_shanten_bot_declares_riichi_when_tenpai(self):
        counts = counts_of("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                           "1p", "2p", "3p", "1s", "2s")
        mask = mask_of(*[kind for kind in range(34) if counts[kind] > 0], 37)
        view = view_for(counts, mask=mask)
        self.assertEqual(SimpleShantenUkeireBot(0).act(view).index, 37)

    def test_kan_choice_recorded(self):
        counts = random_hand(5)
        four = None
        for kind in range(34):
            if counts[kind] >= 1:
                four = kind
                break
        counts[four] += 3
        counts = [min(4, value) for value in counts]
        total = sum(counts)
        while total > 14:
            for kind in range(34):
                if counts[kind] and kind != four:
                    counts[kind] -= 1
                    total -= 1
                    break
        mask = mask_of(*[kind for kind in range(34) if counts[kind] > 0], 42)
        view = view_for(counts, mask=mask, ankan=(four,))
        # legal-random may pick 42; verify the recorded kan choice is well formed
        bot = LegalRandomBot(seed=0)
        seen = set()
        for seed in range(30):
            bot.reset(seed)
            result = bot.act(view)
            self.assertIn(result.index, view.legal_actions())
            if result.index == 42:
                self.assertIsNotNone(result.kan)
                seen.add(result.kan)
        for choice in seen:
            self.assertEqual(choice[0], "ankan")
            self.assertEqual(choice[1], four)

    def test_view_rejects_inconsistent_mask(self):
        counts = random_hand(6)
        with self.assertRaises(ViewError):
            view_for(counts, mask=mask_of(45))  # 45 is not legal on a self turn
        with self.assertRaises(ViewError):
            view_for(counts, mask=mask_of(*range(34), 42), ankan=())  # 42 w/o candidate


# ---------------------------------------------------------------------------
# provenance / output guard
# ---------------------------------------------------------------------------
class ProvenanceTests(TempDirTest):
    def test_guard_out_rejects_frozen_paths(self):
        for relative in ("datasets/x", "runs/x", "checkpoints/x", "vendor/x"):
            with self.subTest(path=relative):
                with self.assertRaises(ProvenanceError):
                    guard_out(REPO_ROOT / relative)
        with self.assertRaises(ProvenanceError):
            guard_out(REPO_ROOT)
        with self.assertRaises(ProvenanceError):
            guard_out("")
        self.assertEqual(guard_out(self.tmp / "ok"), (self.tmp / "ok").resolve())

    def test_generator_fingerprint_is_content_addressed(self):
        fingerprint = generator_fingerprint()
        self.assertRegex(fingerprint["code_sha256"], r"^[0-9a-f]{64}$")
        self.assertGreaterEqual(len(fingerprint["modules"]), 6)
        for name in ("bots.py", "rollout.py", "provenance.py", "seeds.py"):
            self.assertIn(name, fingerprint["modules"])
        copy_dir = self.tmp / "pkg"
        copy_dir.mkdir()
        for path in (REPO_ROOT / "riichiformer" / "selfplay").glob("*.py"):
            shutil.copy(path, copy_dir / path.name)
        first = generator_fingerprint(package_dir=copy_dir)
        (copy_dir / "bots.py").write_text("# changed\n", encoding="utf-8")
        second = generator_fingerprint(package_dir=copy_dir)
        self.assertNotEqual(first["code_sha256"], second["code_sha256"])

    def test_ruleset_labels_never_claim_tenhou(self):
        for ruleset in (RULESET_LIBRIICHI, RULESET_SYNTHETIC):
            metadata = ruleset_metadata(ruleset)
            self.assertFalse(metadata["tenhou_equivalent"])
            self.assertEqual(metadata["strength_claim"], "none")
        self.assertEqual(ruleset_metadata(RULESET_SYNTHETIC)["parity_status"],
                         "not_mahjong_equivalent")
        self.assertEqual(ruleset_metadata(RULESET_LIBRIICHI)["parity_status"],
                         "tenhou_approximate")
        with self.assertRaises(ProvenanceError):
            ruleset_metadata("tenhou_equivalent_official")

    def test_source_metadata_claims_no_third_party(self):
        source = source_metadata(ruleset_id=RULESET_SYNTHETIC)
        row = source["source_rows"][0]
        self.assertEqual(row["source_kind"], "selfplay_own")
        self.assertFalse(row["third_party_logs"])
        self.assertFalse(row["third_party_weights"])
        self.assertFalse(row["network_access"])
        self.assertEqual(row["external_license"], "none")

    def test_engine_resolution_is_explicit(self):
        self.assertEqual(R.resolve_engine(RULESET_SYNTHETIC), R.ENGINE_SYNTHETIC)
        if LIBRIICHI:
            self.assertEqual(R.resolve_engine(RULESET_LIBRIICHI), R.ENGINE_LIBRIICHI)
        else:
            with self.assertRaises(R.RolloutError):
                R.resolve_engine(RULESET_LIBRIICHI)


# ---------------------------------------------------------------------------
# seed pools
# ---------------------------------------------------------------------------
class SeedPoolTests(TempDirTest):
    def test_seed_range_is_unique_walls(self):
        self.assertEqual(seed_range(10, 8), (10, 11))
        with self.assertRaises(SeedError):
            seed_range(0, 6)
        with self.assertRaises(SeedError):
            seed_range(-1, 4)

    def test_cross_pool_reuse_is_refused(self):
        ledger = SeedLedger()
        ledger.reserve_range("train", 100, 8)
        with self.assertRaises(SeedConflict):
            ledger.reserve_range("select", 101, 4)
        with self.assertRaises(SeedConflict):
            ledger.reserve_range("train", 100, 8)
        with self.assertRaises(SeedConflict):
            ledger.reserve("confirm", [100])
        self.assertEqual(ledger.pool_of(100), "train")

    def test_consumed_seed_cannot_be_reused(self):
        ledger = SeedLedger()
        ledger.record_run("train", 7, 4)
        with self.assertRaises(SeedConflict):
            ledger.reserve("confirm", [7])
        with self.assertRaises(SeedConflict):
            ledger.record_run("train", 7, 4)

    def test_consume_requires_matching_pool(self):
        ledger = SeedLedger()
        ledger.reserve_range("train", 20, 4)
        with self.assertRaises(SeedConflict):
            ledger.consume("select", [20])

    def test_ledger_round_trip_and_corruption(self):
        path = self.tmp / "ledger.json"
        ledger = SeedLedger.load(path)
        ledger.record_run("train", 50, 4, tag="unit")
        reloaded = SeedLedger.load(path)
        self.assertEqual(reloaded.reserved("train"), (50,))
        self.assertTrue(reloaded.is_consumed(50))
        self.assertEqual(len(reloaded.runs), 1)
        with self.assertRaises(SeedConflict):
            reloaded.reserve_range("confirm", 50, 4)
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(SeedError):
            SeedLedger.load(path)
        path.write_text(json.dumps({"format": "other"}), encoding="utf-8")
        with self.assertRaises(SeedError):
            SeedLedger.load(path)


# ---------------------------------------------------------------------------
# POV safety
# ---------------------------------------------------------------------------
class PovSafetyTests(unittest.TestCase):
    def test_public_log_never_reads_concealed_fields(self):
        rows = [
            {"type": "start_kyoku", "bakaze": "E", "dora_marker": "4p", "kyoku": 1,
             "honba": 0, "kyotaku": 0, "oya": 0, "scores": [25000] * 4,
             "tehais": [["1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m",
                         "9m", "1p", "2p", "3p"]] * 4},
            {"type": "tsumo", "actor": 0, "pai": "9s"},
            {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": False},
            {"type": "hora", "actor": 0, "target": 1, "deltas": [8000, -8000, 0, 0],
             "ura_markers": ["1m"]},
        ]
        payload = json.dumps(rows)
        parsed = R._public_events(payload)
        for event in parsed:
            self.assertNotIn("tehais", event)
            self.assertNotIn("ura_markers", event)
        log = R._parse_public_log(payload)
        visible = log.visible_counts([0] * 34)
        self.assertEqual(visible[17], 1)   # 9p discarded by seat 1 (kind 17)
        self.assertEqual(visible[11], 0)   # 2p never appeared publicly
        self.assertEqual(visible[12], 1)   # dora indicator 4p (kind 12)
        self.assertEqual(visible[0], 0)    # opponent haipai 1m stays hidden

    def test_events_json_must_be_a_list(self):
        with self.assertRaises(R.RolloutError):
            R._public_events(json.dumps({"type": "start_kyoku"}))


# ---------------------------------------------------------------------------
# synthetic rollout end to end
# ---------------------------------------------------------------------------
class SyntheticRolloutTests(TempDirTest):
    def _config(self, *, seed_start=11, games=4, pool="train", ledger=None,
                lineup=("legal-random", "legal-random", "legal-random", "legal-random")):
        return R.RolloutConfig(
            seed_start=seed_start, games=games, ruleset_id=RULESET_SYNTHETIC,
            out=self.tmp / "out", key=777, lineup=lineup, seed_pool=pool,
            ledger=ledger, write_logs=False)

    def test_synthetic_rollout_writes_ledger_and_summary(self):
        ledger_path = self.tmp / "seeds.json"
        ledger = SeedLedger.load(ledger_path)
        summary = R.run_selfplay(self._config(ledger=ledger))
        self.assertTrue(summary["audit"]["ok"], summary["audit"])
        self.assertEqual(summary["games"], 4)
        self.assertEqual(summary["seed_count"], 1)
        self.assertEqual(summary["ruleset_id"], RULESET_SYNTHETIC)
        self.assertEqual(summary["rules_engine"], R.ENGINE_SYNTHETIC)
        self.assertFalse(summary["tenhou_equivalent"])
        self.assertEqual(summary["parity_status"], "not_mahjong_equivalent")
        self.assertEqual(summary["strength_claim"], "none")
        self.assertEqual(summary["illegal_actions"], 0)
        self.assertGreater(summary["decisions"], 0)
        self.assertEqual(summary["score_conservation"]["violations"], 0)
        for name in ("legal-random",):
            self.assertIn(name, summary["per_bot"])
        out = self.tmp / "out"
        for name in ("summary.json", "games.jsonl", "provenance.json", "seeds.json"):
            self.assertTrue((out / name).is_file(), name)
        rows = [json.loads(line) for line in
                (out / "games.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(len(row["povs"]), 4)
            self.assertEqual(row["score_sum"], 100000)
            self.assertFalse(row["tenhou_equivalent"])
            self.assertRegex(row["generator_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual({pov["position"] for pov in row["povs"]}, {0, 1, 2, 3})
            self.assertFalse(row["source"]["third_party_logs"])
        provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        self.assertFalse(provenance["ruleset"]["tenhou_equivalent"])
        self.assertEqual(provenance["source"]["source_rows"][0]["external_license"], "none")
        self.assertEqual(ledger.reserved("train"), (11,))
        self.assertTrue(ledger.is_consumed(11))

    def test_seed_reuse_between_runs_is_refused(self):
        ledger_path = self.tmp / "seeds.json"
        R.run_selfplay(self._config(ledger=SeedLedger.load(ledger_path)))
        with self.assertRaises(SeedConflict):
            R.run_selfplay(self._config(ledger=SeedLedger.load(ledger_path)))
        # a different pool on a fresh ledger file is still disjoint
        other = SeedLedger.load(self.tmp / "other.json")
        other.reserve_range("select", 11, 4)
        with self.assertRaises(SeedConflict):
            other.reserve("confirm", [11])

    def test_four_seat_rotation_and_scores(self):
        records = R.run_seed(5, 777, ("defense", "legal-random", "shanten-ukeire",
                                      "legal-random"),
                             ruleset_id=RULESET_SYNTHETIC, out=self.tmp / "rot")
        self.assertEqual(len(records), 4)
        for record in records:
            self.assertEqual(sorted(record.seats), [0, 1, 2, 3])
            self.assertEqual(sorted(record.ranks), [1, 2, 3, 4])
            self.assertEqual(sum(record.scores), 100000)
            self.assertEqual(record.illegal, 0)
            self.assertEqual([pov.position for pov in record.povs], [0, 1, 2, 3])
        # seat rotation: position p is at seat (rotation + p) % 4
        for record in records:
            for pov in record.povs:
                self.assertEqual(pov.seat, (record.rotation + pov.position) % 4)
            self.assertEqual(record.lineup,
                             ("defense", "legal-random", "shanten-ukeire", "legal-random"))

    def test_ruleset_is_mandatory(self):
        with self.assertRaises(ProvenanceError):
            R.RolloutConfig(seed_start=1, games=4, ruleset_id="tenhou_equiv",
                            out=self.tmp / "x")
        with self.assertRaises(SeedError):
            R.RolloutConfig(seed_start=1, games=5, ruleset_id=RULESET_SYNTHETIC,
                            out=self.tmp / "x")
        with self.assertRaises(R.RolloutError):
            R.RolloutConfig(seed_start=1, games=4, ruleset_id=RULESET_SYNTHETIC,
                            out=self.tmp / "x", lineup=("nope",) * 4)

    def test_cli_refuses_frozen_output_path(self):
        from scripts import rf_selfplay

        code = rf_selfplay.main([
            "--out", str(REPO_ROOT / "datasets" / "rf41"),
            "--seed-start", "1", "--games", "4",
            "--ruleset", RULESET_SYNTHETIC,
        ])
        self.assertEqual(code, 2)

    def test_cli_runs_and_reports(self):
        from scripts import rf_selfplay

        code = rf_selfplay.main([
            "--out", str(self.tmp / "cli"), "--seed-start", "21", "--games", "4",
            "--ruleset", RULESET_SYNTHETIC, "--seed-pool", "select",
            "--seed-ledger", str(self.tmp / "cli-seeds.json"), "--quiet",
        ])
        self.assertEqual(code, 0)
        summary = json.loads((self.tmp / "cli" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["games"], 4)
        self.assertEqual(summary["ruleset_id"], RULESET_SYNTHETIC)


# ---------------------------------------------------------------------------
# real local rules engine
# ---------------------------------------------------------------------------
@unittest.skipUnless(LIBRIICHI, "pinned libriichi arena is not importable")
class LibriichiRolloutTests(TempDirTest):
    def test_real_four_seat_rotation(self):
        records = R._run_seed_libriichi(
            23, 20261121, ("shanten-ukeire", "defense", "legal-random", "defense"),
            out=self.tmp / "real", write_logs=True, verify_bots=True)
        self.assertEqual(len(records), 4)
        for record in records:
            self.assertEqual(record.ruleset_id, RULESET_LIBRIICHI)
            self.assertEqual(record.rules_engine, R.ENGINE_LIBRIICHI)
            self.assertEqual(sum(record.scores), 100000)
            self.assertEqual(sorted(record.ranks), [1, 2, 3, 4])
            self.assertEqual(sorted(record.seats), [0, 1, 2, 3])
            self.assertEqual(len(record.povs), 4)
            self.assertEqual(record.illegal, 0)
            self.assertEqual(record.rejected, 0)
            self.assertEqual(record.fallbacks, 0)
            self.assertGreater(record.decisions, 0)
            self.assertTrue(record.game_log_path)
            log_path = self.tmp / "real" / record.game_log_path
            self.assertTrue(log_path.is_file())
            self.assertEqual(sha256_file(log_path), record.game_log_sha256)
            for key in ("R02_multi_ron", "R02_triple_ron", "R26_four_riichi",
                        "R29_ippatsu_proxy_unverified", "R25_end_kyotaku_carry_proxy"):
                self.assertIn(key, record.deviation_counts)
        notes = " ".join(records[0].notes)
        self.assertIn("mask_checks=", notes)
        mask_checks = int(notes.split("mask_checks=")[1].split()[0])
        self.assertGreater(mask_checks, 0)
        shanten_checks = int(notes.split("verify_shanten_checks=")[1].split()[0])
        self.assertGreater(shanten_checks, 0)
        dev = records[0].deviation_counts
        self.assertGreater(dev.get("kyoku", 0), 0)
        self.assertGreater(dev.get("reach", 0) + dev.get("hora", 0), 0)

    def test_mask_derivation_explains_the_native_mask(self):
        # The native mask is authoritative; the independent derivation must at least
        # contain every native action.  This runs the same bridge with verification.
        records = R._run_seed_libriichi(
            24, 20261121, ("defense", "legal-random", "defense", "legal-random"),
            out=self.tmp / "real2", write_logs=False, verify_bots=False)
        self.assertEqual(len(records), 4)
        for record in records:
            self.assertEqual(sum(record.scores), 100000)
            self.assertEqual(record.illegal, 0)


if __name__ == "__main__":
    unittest.main()
