#!/usr/bin/env python3
"""PR-45 validation corpus generation on top of the frozen self-play machinery.

A validation corpus is a set of **native** four-seat hanchan played by the pinned
local ``libriichi`` arena with the project's weak heuristic bots.  It uses the
existing :class:`~riichiformer.selfplay.rollout.RolloutConfig` /
:func:`~riichiformer.selfplay.rollout.run_selfplay` and therefore inherits the
four-seat rotation, the full-information engine log, the provenance file and the
hard score/legality audit.

What this module adds:

* a fresh 64-bit ``seed_key`` generated from the OS CSPRNG (never a constant);
* a wall window that is audited against B0 and every consumed evaluation window;
* the exact corpus manifest (game ids + log hashes + generator hash) used by the
  shard builder.

The bots are labelled weak local heuristics everywhere.  Nothing here is a strong
teacher, Tenhou truth or a strength measurement.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..selfplay.bots import default_lineup
from ..selfplay.provenance import (
    RULESET_LIBRIICHI,
    ProvenanceError,
    generator_fingerprint,
    guard_out,
    utc_now,
    write_json,
)
from ..selfplay.rollout import RolloutConfig, RolloutError, run_selfplay
from .disjoint import (
    B0_SAMPLE_SEED,
    RESERVED_SEED_KEYS,
    audit_seed_window,
)

__all__ = [
    "MAX_VALIDATION_GAMES",
    "MIN_VALIDATION_GAMES",
    "TARGET_VALIDATION_GAMES",
    "ValidationCorpusError",
    "ValidationWindow",
    "generate_seed_key",
    "run_validation_selfplay",
    "validation_game_ids",
]

MIN_VALIDATION_GAMES = 4
MAX_VALIDATION_GAMES = 200
#: PR-45 asks for 100-200 real games.  A smaller corpus is allowed but must be
#: reported as an explicit shortfall, never rounded up.
TARGET_VALIDATION_GAMES = (100, 200)

_ENGINE_LOG_RE = re.compile(r"^(?P<seed>\d+)_(?P<key>\d+)_(?P<rot>[abcd])\.json\.gz$")


class ValidationCorpusError(RuntimeError):
    """The validation corpus cannot be generated or is not disjoint."""


def generate_seed_key(*, exclude: Sequence[int] = ()) -> int:
    """A fresh non-zero 64-bit key from ``secrets.randbits``.

    Reserved historical keys and any caller-supplied exclusions are rejected
    (retry, not fallback to a constant).
    """
    blocked = set(RESERVED_SEED_KEYS) | set(exclude)
    for _ in range(64):
        key = secrets.randbits(64)
        if key and key not in blocked:
            return key
    raise ValidationCorpusError("could not generate a fresh 64-bit seed key")


@dataclass(frozen=True)
class ValidationWindow:
    """One fresh wall window: ``walls = games / 4`` seeds, four rotations each."""

    seed_start: int
    games: int
    seed_key: int
    ruleset_id: str = RULESET_LIBRIICHI

    def __post_init__(self):
        if not isinstance(self.games, int) or isinstance(self.games, bool) \
                or self.games < MIN_VALIDATION_GAMES or self.games > MAX_VALIDATION_GAMES \
                or self.games % 4 != 0:
            raise ValidationCorpusError(
                f"games must be a multiple of 4 in "
                f"[{MIN_VALIDATION_GAMES},{MAX_VALIDATION_GAMES}], got {self.games!r}")
        if not isinstance(self.seed_start, int) or isinstance(self.seed_start, bool) \
                or self.seed_start < 0:
            raise ValidationCorpusError(
                f"seed_start must be a non-negative int, got {self.seed_start!r}")
        if not isinstance(self.seed_key, int) or isinstance(self.seed_key, bool) \
                or not 0 < self.seed_key < 2 ** 64:
            raise ValidationCorpusError(
                f"seed_key must be a 64-bit non-zero int, got {self.seed_key!r}")
        if self.seed_key == B0_SAMPLE_SEED:
            raise ValidationCorpusError(
                "the B0 sample seed must not be reused as a validation wall key")
        if self.ruleset_id != RULESET_LIBRIICHI:
            raise ValidationCorpusError(
                "independent validation must be native self-play "
                f"({RULESET_LIBRIICHI}); the procedural fallback is not mahjong")
        audit = audit_seed_window(self.seed_start, self.games, self.seed_key)
        if not audit["window_ok"]:
            raise ValidationCorpusError(
                f"seed window {audit['window']} overlaps consumed windows "
                f"{audit['window_overlaps']}")
        if not audit["seed_key_ok"]:
            raise ValidationCorpusError(
                f"seed_key 0x{self.seed_key:016x} is reserved "
                f"({audit['seed_key_reserved']})")

    @property
    def walls(self) -> int:
        return self.games // 4

    @property
    def seed_stop(self) -> int:
        return self.seed_start + self.walls

    @property
    def window(self) -> str:
        return f"[{self.seed_start},{self.seed_stop})"

    def as_dict(self) -> dict:
        return {
            "seed_start": self.seed_start,
            "seed_stop": self.seed_stop,
            "walls": self.walls,
            "games": self.games,
            "seed_key": self.seed_key,
            "seed_key_hex": f"0x{self.seed_key:016x}",
            "window": self.window,
            "ruleset_id": self.ruleset_id,
        }


def validation_game_ids(games_dir: str | Path) -> list[str]:
    """Game ids of an existing corpus, parsed from the engine-log filenames."""
    ids = []
    for path in sorted(Path(games_dir).glob("*.json.gz")):
        match = _ENGINE_LOG_RE.match(path.name)
        if match is None:
            raise ValidationCorpusError(f"unexpected engine log {path.name!r}")
        rotation = "abcd".index(match.group("rot"))
        ids.append(f"{int(match.group('seed'))}:{int(match.group('key'))}:{rotation}")
    return ids


def run_validation_selfplay(out: str | Path, window: ValidationWindow, *,
                            lineup: Sequence[str] | None = None,
                            write_logs: bool = True,
                            progress: Callable[[str], None] | None = None
                            ) -> dict:
    """Play the window with the pinned arena and write the corpus artifacts."""
    try:
        out_dir = guard_out(out)
    except ProvenanceError as exc:
        raise ValidationCorpusError(str(exc)) from exc
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValidationCorpusError(
            f"validation output {out_dir} is not empty; use a fresh directory")
    config = RolloutConfig(
        seed_start=window.seed_start,
        games=window.games,
        ruleset_id=window.ruleset_id,
        out=out_dir,
        key=window.seed_key,
        lineup=tuple(lineup) if lineup is not None else default_lineup(),
        seed_pool="select",
        ledger=None,
        write_logs=write_logs,
    )
    generator = generator_fingerprint()
    try:
        summary = run_selfplay(config, progress=progress)
    except (RolloutError, ProvenanceError, ValueError) as exc:
        raise ValidationCorpusError(
            f"native self-play failed: {type(exc).__name__}: {exc}") from exc
    if summary["games"] != window.games:
        raise ValidationCorpusError(
            f"engine produced {summary['games']} games, expected {window.games}")
    if summary["key"] != window.seed_key:
        raise ValidationCorpusError("provenance key mismatch")
    audit = summary.get("audit") or {}
    if not audit.get("ok"):
        raise ValidationCorpusError(f"self-play audit failed: {audit.get('violations')}")
    game_ids = [row["game_id"] for row in _read_games_jsonl(out_dir / "games.jsonl")]
    if len(game_ids) != window.games or len(set(game_ids)) != len(game_ids):
        raise ValidationCorpusError("duplicate or missing game ids in games.jsonl")
    corpus = {
        "format": "rf-validation-corpus/1",
        "created_utc": utc_now(),
        "role": "independent_validation",
        "window": window.as_dict(),
        "lineup": list(config.lineup),
        "generator_sha256": generator["code_sha256"],
        "generator": generator,
        "ruleset_id": window.ruleset_id,
        "engine": summary["rules_engine"],
        "games": window.games,
        "decisions": summary["decisions"],
        "game_ids": game_ids,
        "score_conservation": summary["score_conservation"],
        "deviations": summary["deviations"],
        "illegal_actions": summary["illegal_actions"],
        "rejected_actions": summary["rejected_actions"],
        "fallbacks": summary["fallbacks"],
        "audit_ok": True,
        "strength_claim": "none",
        "teacher_claim": (
            "weak local heuristic bots only; not a strong teacher, not Tenhou "
            "truth; CE on this corpus measures imitation of the weak bots"),
        "r29_status": ("MATCH per PR-40R (Lead-relayed); shard-level provenance "
                       "flag only because the frozen per-decision enum has no R29"),
        "outputs": {
            "games_jsonl": str(out_dir / "games.jsonl"),
            "games_dir": str(out_dir / "games"),
            "summary": str(out_dir / "summary.json"),
            "provenance": str(out_dir / "provenance.json"),
        },
    }
    write_json(out_dir / "validation_corpus.json", corpus)
    return corpus


def _read_games_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise ValidationCorpusError(f"missing {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def corpus_artifact_sha256(path: str | Path) -> str:
    """SHA-256 of the corpus manifest file (evidence for the intersection report)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
