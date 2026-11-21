#!/usr/bin/env python3
"""PR-42 tests: opponent registry, duplicate arena audit and seed-block bootstrap.

All tests are offline and CPU-only.  The integration test drives the procedural
fallback (no libriichi needed); no engine, dataset, checkpoint or network is touched.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.eval import arena as A  # noqa: E402
from riichiformer.eval.opponents import (  # noqa: E402
    RESERVED_ENGINE_TOKENS,
    ExternalOpponentRefused,
    ImpersonationError,
    OpponentError,
    OpponentKind,
    OpponentRegistry,
    OpponentSpec,
    RegistryConflict,
    UnknownOpponent,
    build_lineup,
    default_registry,
)
from riichiformer.selfplay import rollout as R  # noqa: E402
from riichiformer.selfplay.provenance import (  # noqa: E402
    RULESET_SYNTHETIC,
    generator_fingerprint,
)
from riichiformer.selfplay.seeds import SeedLedger  # noqa: E402

CAVEAT = "locally authored test heuristic; no weights and no third-party claim"


def spec(name: str, *, kind=OpponentKind.LOCAL_HEURISTIC, checkpoint=None,
         ruleset=RULESET_SYNTHETIC, caveat=CAVEAT, external=False,
         display_name="", version="v1", notes="") -> OpponentSpec:
    return OpponentSpec(name=name, version=version, kind=kind,
                        checkpoint_sha256=checkpoint, license_caveat=caveat,
                        ruleset=ruleset, display_name=display_name, external=external,
                        notes=notes)


def make_registry(*names: str) -> OpponentRegistry:
    registry = OpponentRegistry()
    for name in names:
        registry.register(spec(name))
    return registry


# ---------------------------------------------------------------------------
# fake game records (duck-typed like riichiformer.selfplay.rollout.GameRecord)
# ---------------------------------------------------------------------------
@dataclass
class FakePov:
    position: int
    seat: int
    bot: str


@dataclass
class FakeRecord:
    seed: int
    rotation: int
    lineup: tuple[str, ...]
    scores: tuple[int, int, int, int]
    ranks: tuple[int, int, int, int]
    seats: tuple[int, int, int, int]
    povs: tuple[FakePov, ...]
    ruleset_id: str = RULESET_SYNTHETIC
    rules_engine: str = R.ENGINE_SYNTHETIC
    illegal: int = 0
    rejected: int = 0
    fallbacks: int = 0
    deviation_counts: dict = field(default_factory=dict)

    @property
    def game_id(self) -> str:
        return f"{self.seed}:{self.rotation}"


def ranks_for(scores) -> tuple[int, int, int, int]:
    order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
    ranks = [0] * 4
    for rank, seat in enumerate(order, start=1):
        ranks[seat] = rank
    return tuple(ranks)


def fake_game(seed: int, rotation: int, lineup, scores, *, ruleset=RULESET_SYNTHETIC,
              engine=R.ENGINE_SYNTHETIC, illegal=0, rejected=0, fallbacks=0,
              deviation_counts=None) -> FakeRecord:
    seats = tuple((rotation + position) % 4 for position in range(4))
    return FakeRecord(
        seed=seed, rotation=rotation, lineup=tuple(lineup), scores=tuple(scores),
        ranks=ranks_for(scores), seats=seats,
        povs=tuple(FakePov(position, seats[position], lineup[position])
                   for position in range(4)),
        ruleset_id=ruleset, rules_engine=engine, illegal=illegal, rejected=rejected,
        fallbacks=fallbacks, deviation_counts=dict(deviation_counts or {}))


def constant_runner(scores_by_seed, lineup, **kwargs):
    def runner(seed: int):
        return [fake_game(seed, rotation, lineup, scores_by_seed[seed],
                          **kwargs) for rotation in range(4)]

    return runner


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
class RegistryTests(unittest.TestCase):
    def test_default_registry_only_has_weight_free_local_bots(self):
        registry = default_registry()
        self.assertEqual(set(registry.names()),
                         {"legal-random", "shanten-ukeire", "defense"})
        for name in registry.names():
            entry = registry.get(name)
            self.assertIs(entry.kind, OpponentKind.LOCAL_HEURISTIC)
            self.assertIsNone(entry.checkpoint_sha256)
            self.assertFalse(entry.external)
            lowered = f"{entry.name} {entry.display_name}".lower()
            for token in RESERVED_ENGINE_TOKENS:
                self.assertNotIn(token, lowered)
        self.assertIn("no Mortal", registry.as_dict()["note"])
        self.assertFalse(registry.as_dict()["allow_external"])

    def test_unknown_opponent_is_refused(self):
        registry = default_registry()
        with self.assertRaises(UnknownOpponent):
            registry.get("mortal")
        with self.assertRaises(UnknownOpponent):
            registry.resolve(["defense", "akochan"])

    def test_duplicate_registration_conflicts(self):
        registry = default_registry()
        identical = registry.get("defense")
        self.assertIs(registry.register(identical), identical)
        with self.assertRaises(RegistryConflict):
            registry.register(spec("defense", version="v9"))

    def test_local_bot_cannot_impersonate_mortal(self):
        registry = OpponentRegistry()
        for token in ("mortal", "akochan", "tenhou"):
            with self.subTest(token=token):
                with self.assertRaises(ImpersonationError):
                    registry.register(spec(token + "-bot"))
                with self.assertRaises(ImpersonationError):
                    registry.register(spec("defense-x", display_name=token.upper()))
        with self.assertRaises(ImpersonationError):
            registry.register(spec("defense", checkpoint="a" * 64))
        with self.assertRaises(ImpersonationError):
            registry.register(spec("defense", external=True))

    def test_external_engine_requires_explicit_opt_in(self):
        registry = OpponentRegistry()
        with self.assertRaises(ExternalOpponentRefused):
            registry.register(spec("mortal-1v3", kind=OpponentKind.EXTERNAL_ENGINE,
                                   checkpoint="b" * 64, external=True))
        opted_in = OpponentRegistry(allow_external=True)
        entry = opted_in.register(spec(
            "mortal-1v3", kind=OpponentKind.EXTERNAL_ENGINE, checkpoint="b" * 64,
            external=True, caveat="third-party engine used only for local research, "
                                  "no redistribution"))
        self.assertEqual(entry.kind, OpponentKind.EXTERNAL_ENGINE)
        self.assertEqual(entry.checkpoint_sha256, "b" * 64)
        with self.assertRaises(OpponentError):
            opted_in.register(spec("ext-no-hash", kind=OpponentKind.EXTERNAL_ENGINE,
                                   external=True))

    def test_spec_validation(self):
        with self.assertRaises(OpponentError):
            spec("Bad Name")
        with self.assertRaises(OpponentError):
            spec("ok-name", caveat="short")
        with self.assertRaises(OpponentError):
            spec("ok-name", ruleset="tenhou_equivalent")
        with self.assertRaises(OpponentError):
            spec("ok-name", checkpoint="not-a-sha")
        with self.assertRaises(OpponentError):
            spec("ok-name", kind=OpponentKind.LOCAL_MODEL, checkpoint=None)

    def test_build_lineup_modes(self):
        registry = default_registry()
        mode, lineup = build_lineup(["shanten-ukeire", "defense"], registry=registry)
        self.assertEqual(mode, "one_vs_three")
        self.assertEqual(lineup, ("shanten-ukeire", "defense", "defense", "defense"))
        mode, lineup = build_lineup(
            ["shanten-ukeire", "defense", "legal-random", "defense"], registry=registry)
        self.assertEqual(mode, "free_for_all")
        with self.assertRaises(OpponentError):
            build_lineup(["defense"], registry=registry)
        with self.assertRaises(OpponentError):
            build_lineup(["defense", "defense", "defense"], registry=registry)
        with self.assertRaises(UnknownOpponent):
            build_lineup(["defense", "mortal"], registry=registry)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
class GameAuditTests(unittest.TestCase):
    def test_clean_game_passes(self):
        audit = A.GameAudit()
        audit.check(fake_game(1, 0, ("a", "b", "b", "b"), (30000, 25000, 25000, 20000),
                              deviation_counts={"R02_multi_ron": 0}))
        self.assertTrue(audit.ok, audit.as_dict())
        self.assertEqual(audit.games, 1)

    def test_score_conservation_and_illegal_actions_flagged(self):
        audit = A.GameAudit()
        audit.check(fake_game(1, 0, ("a", "b", "b", "b"), (30000, 25000, 25000, 19000)))
        audit.check(fake_game(2, 0, ("a", "b", "b", "b"), (25000, 25000, 25000, 25000),
                              illegal=2))
        categories = audit.as_dict()["violation_categories"]
        self.assertIn("score_conservation", categories)
        self.assertIn("illegal_actions", categories)
        self.assertFalse(audit.ok)

    def test_bad_ranks_seats_and_povs_flagged(self):
        audit = A.GameAudit()
        record = fake_game(1, 0, ("a", "b", "b", "b"), (30000, 25000, 25000, 20000))
        record.ranks = (1, 1, 3, 4)
        record.seats = (0, 0, 2, 3)
        record.povs = record.povs[:3]
        audit.check(record)
        categories = set(audit.as_dict()["violation_categories"])
        self.assertIn("rank_permutation", categories)
        self.assertIn("seat_rotation", categories)
        self.assertIn("pov_count", categories)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
class BootstrapTests(unittest.TestCase):
    def test_block_bootstrap_does_not_treat_four_seats_as_independent(self):
        # Each seed has the same mean but a wide within-seed spread; the four rotated
        # games share a wall, so the seed-block interval must be (near) zero-width,
        # while an independent-games bootstrap reports a spurious interval.
        blocks = [[0.0, 0.0, 10.0, 10.0] for _ in range(3)]
        block_ci = A.bootstrap_seed_blocks(blocks, statistic=A._mean, resamples=400)
        naive_ci = A.bootstrap_independent_games(blocks, statistic=A._mean, resamples=400)
        self.assertEqual(block_ci["unit"], "seed")
        self.assertEqual(block_ci["blocks"], 3)
        self.assertAlmostEqual(block_ci["low"], 5.0)
        self.assertAlmostEqual(block_ci["high"], 5.0)
        self.assertEqual(naive_ci["unit"], "game")
        self.assertGreater(naive_ci["high"] - naive_ci["low"], 0.5)
        self.assertIn("invalid", naive_ci["warning"])

    def test_block_bootstrap_within_seed_variation_is_not_shrunk(self):
        rng = random.Random(0)
        blocks = [[rng.uniform(0, 10) for _ in range(4)] for _ in range(8)]
        block_ci = A.bootstrap_seed_blocks(blocks, statistic=A._mean, resamples=500,
                                           seed=3)
        naive_ci = A.bootstrap_independent_games(blocks, statistic=A._mean,
                                                 resamples=500, seed=3)
        block_width = block_ci["high"] - block_ci["low"]
        naive_width = naive_ci["high"] - naive_ci["low"]
        self.assertGreater(block_width, naive_width)
        self.assertLessEqual(block_ci["low"], block_ci["point"])
        self.assertGreaterEqual(block_ci["high"], block_ci["point"])

    def test_bootstrap_requires_two_blocks(self):
        outcome = A.bootstrap_seed_blocks([[1.0, 2.0]], statistic=A._mean, resamples=200)
        self.assertIsNone(outcome["low"])
        self.assertEqual(outcome["resamples"], 0)
        with self.assertRaises(A.ArenaError):
            A.bootstrap_seed_blocks([[1.0], [2.0]], statistic=A._mean, resamples=10)


# ---------------------------------------------------------------------------
# duplicate arena
# ---------------------------------------------------------------------------
class DuplicateArenaTests(unittest.TestCase):
    def _arena(self, runner, *, resamples=200, audit=None, registry=None):
        return A.DuplicateArena(
            registry=registry if registry is not None else make_registry("alpha", "beta"),
            runner=runner, ruleset_id=RULESET_SYNTHETIC,
            expected_engine=R.ENGINE_SYNTHETIC, key=1, resamples=resamples,
            audit=audit)

    def test_one_vs_three_rotation_and_per_opponent_stats(self):
        lineup = ("alpha", "beta", "beta", "beta")
        scores = (30000, 25000, 25000, 20000)
        report = self._arena(constant_runner({1: scores, 2: scores}, lineup)).run(
            (1, 2), opponents=("alpha", "beta"))
        self.assertEqual(report.mode, "one_vs_three")
        self.assertEqual(report.lineup, lineup)
        self.assertEqual(report.games, 8)
        alpha = report.per_opponent["alpha"]
        beta = report.per_opponent["beta"]
        self.assertEqual(alpha["games"], 8)          # one seat per game
        self.assertEqual(beta["games"], 24)          # three seats per game
        self.assertEqual(alpha["mean_rank"], 1.0)
        self.assertEqual(alpha["win_rate"], 1.0)
        self.assertEqual(alpha["rank_counts"]["1"], 8)
        self.assertEqual(beta["mean_rank"], 3.0)
        self.assertEqual(beta["win_rate"], 0.0)
        self.assertEqual(alpha["mean_rank_ci95"]["unit"], "seed")
        self.assertFalse(report.as_dict()["official_rank"])
        self.assertEqual(report.as_dict()["strength_claim"], "none")
        self.assertFalse(report.as_dict()["tenhou_equivalent"])
        self.assertTrue(report.audit["ok"])

    def test_arena_rejects_malformed_runs(self):
        lineup = ("alpha", "beta", "beta", "beta")
        scores = (30000, 25000, 25000, 20000)
        def three_games(seed):
            return [fake_game(seed, rotation, lineup, scores) for rotation in range(3)]
        with self.assertRaises(A.ArenaError):
            self._arena(three_games).run((1,), lineup=lineup)

        def wrong_seed(seed):
            return [fake_game(seed + 1, rotation, lineup, scores) for rotation in range(4)]
        with self.assertRaises(A.ArenaError):
            self._arena(wrong_seed).run((1,), lineup=lineup)

        def wrong_lineup(seed):
            return [fake_game(seed, rotation, ("alpha", "alpha", "alpha", "alpha"),
                              scores) for rotation in range(4)]
        with self.assertRaises(A.ArenaError):
            self._arena(wrong_lineup).run((1,), lineup=lineup)

        def wrong_ruleset(seed):
            return [fake_game(seed, rotation, lineup, scores, ruleset=R.RULESET_LIBRIICHI)
                    for rotation in range(4)]
        with self.assertRaises(A.ArenaError):
            self._arena(wrong_ruleset).run((1,), lineup=lineup)

        def duplicated_rotations(seed):
            return [fake_game(seed, 0, lineup, scores) for _ in range(4)]
        with self.assertRaises(A.ArenaError):
            self._arena(duplicated_rotations).run((1,), lineup=lineup)

        with self.assertRaises(A.ArenaError):
            self._arena(constant_runner({1: scores}, lineup)).run((1, 1), lineup=lineup)

    def test_audit_violation_propagates_to_the_report(self):
        lineup = ("alpha", "beta", "beta", "beta")
        bad = (30000, 25000, 25000, 19000)  # 99000 total
        report = self._arena(constant_runner({1: bad}, lineup)).run((1,), lineup=lineup)
        self.assertFalse(report.audit["ok"])
        self.assertIn("score_conservation", report.audit["violation_categories"])

    def test_unknown_opponent_is_refused(self):
        with self.assertRaises(UnknownOpponent):
            self._arena(constant_runner({1: (25000,) * 4}, ("alpha", "beta", "beta", "beta"))
                        ).run((1,), opponents=("alpha", "mortal"))

    def test_integration_with_procedural_rollout(self):
        tmp = Path(tempfile.mkdtemp(prefix="rf42-arena-"))
        try:
            registry = default_registry()
            lineup = ("shanten-ukeire", "defense", "legal-random", "defense")
            def runner(seed: int):
                return R.run_seed(seed, 42, lineup, ruleset_id=RULESET_SYNTHETIC,
                                  out=tmp / "games", write_logs=False)
            report = self._arena(runner, registry=registry).run((5,), lineup=lineup)
            payload = report.as_dict()
            self.assertTrue(payload["audit"]["ok"], payload["audit"])
            self.assertEqual(payload["games"], 4)
            self.assertEqual(payload["seed_count"], 1)
            self.assertEqual(len(report.game_rows), 4)
            self.assertEqual([row["rotation"] for row in report.game_rows], [0, 1, 2, 3])
            for name in lineup:
                self.assertIn(name, payload["per_opponent"])
                entry = payload["per_opponent"][name]
                self.assertEqual(entry["kind"], "local_heuristic")
                self.assertIsNone(entry["checkpoint_sha256"])
                self.assertEqual(entry["mean_rank_ci95"]["unit"], "seed")
            self.assertFalse(payload["official_rank"])
            self.assertEqual(payload["strength_claim"], "none")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_duplicate_arena_reports_generator_provenance(self):
        fingerprint = generator_fingerprint()
        self.assertRegex(fingerprint["code_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("rollout.py", fingerprint["modules"])


if __name__ == "__main__":
    unittest.main()
