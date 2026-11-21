#!/usr/bin/env python3
"""PR-45 validation shard: self-play logs -> diagnostic ``rf-shard/3``.

The builder reuses the frozen production pieces and changes only the provenance:

* decisions come from :func:`riichiformer.data.decision.build_decisions` with
  ``RulesetId.MORTAL_ARENA_PINNED`` / ``SourceKind.ARENA_SELFPLAY`` (R02/R25/R26 as
  per-decision ruleset deviations);
* rows come from :func:`riichiformer.data.shard.make_game_record` (four POV
  streams, projected once, no concealed fields);
* writes go through :class:`~riichiformer.data.shard.ShardWriter` with
  ``purpose="diagnostic"`` and ``research_policy="operator_private_research"``.

R29 is **not** representable in the frozen ``RulesetDeviation`` enum, so it is
recorded at shard level only (the full known-deviation set required for this
corpus) and the per-decision snapshot carries R02/R25/R26.  PR-40R reports the
R29 ippatsu sequence as MATCH; the enum gap is still real, and inventing an enum
value in a frozen schema would change the normalizer contract.  The gap is
asserted by the validation tests.

``ShardMetadata`` has three mandatory 64-hex provenance fields that were named
for the H-02 human-log pipeline.  A self-play corpus has no canonical/split
manifest, so those fields carry explicit, documented derivations of the corpus
itself (:data:`SELFPLAY_HASH_DERIVATION`); the names are frozen by
``data/shard.py`` and are not rewritten here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..data.decision import DecisionBuildConfig, build_decisions
from ..data.replay import replay_scores
from ..data.shard import (
    ACTION_HASH,
    FROZEN_SCHEMA,
    NORMALIZER_HASH,
    SCHEMA_HASH,
    ShardMetadata,
    ShardWriter,
    make_game_record,
    sample_hashes,
)
from ..policy.config import PolicyConfig
from ..schemas.decision_v3 import RulesetDeviation, RulesetId, SourceKind
from .integrity import assert_visibility_contract, verify_shard_integrity

__all__ = [
    "SELFPLAY_HASH_DERIVATION",
    "ValidationShardError",
    "ValidationShardSpec",
    "build_validation_shard",
    "read_engine_log",
    "scan_engine_logs",
    "scenario_counts",
]

LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)_(?P<key>\d+)_(?P<rot>[abcd])\.json\.gz$")
ROTATIONS = "abcd"
#: Shard-level known deviations of ``mortal_arena_pinned``.  R29 (ippatsu timing)
#: is reported MATCH by PR-40R but stays a shard-level provenance flag because the
#: frozen per-decision enum has no R29 member.
ARENA_DEVIATIONS = ("R02", "R25", "R26", "R29")
#: Per-decision deviations representable by the frozen enum.
DECISION_DEVIATIONS = (RulesetDeviation.R02, RulesetDeviation.R25,
                       RulesetDeviation.R26)

SELFPLAY_HASH_DERIVATION = {
    "canonical_manifest_sha256": (
        "sha256(canonical JSON of the validation corpus manifest: format, "
        "ruleset, window, key, lineup, generator sha, and every "
        "(game_id, log_sha256) pair)"),
    "canonical_index_sha256": (
        "sha256(canonical JSON of the sorted [[game_id, log_sha256], ...] index)"),
    "split_manifest_sha256": (
        "sha256(canonical JSON of {seed_start, games, seed_key, ruleset_id, "
        "lineup, generator_sha256})"),
    "note": ("self-play corpus has no H-02 canonical/split manifest; the field "
             "names are frozen by data/shard.py and carry these corpus hashes"),
}


class ValidationShardError(RuntimeError):
    """The corpus cannot be sealed into a diagnostic shard."""


@dataclass(frozen=True)
class ValidationShardSpec:
    seed_start: int
    games: int
    seed_key: int
    ruleset_id: str = "libriichi_tenhou_approx/1"
    lineup: tuple[str, ...] = ()
    generator_sha256: str = ""

    def __post_init__(self):
        if self.games <= 0 or self.games % 4 != 0:
            raise ValidationShardError("games must be a positive multiple of 4")
        if not 0 < self.seed_key < 2 ** 64:
            raise ValidationShardError("seed_key must be a 64-bit non-zero int")
        if len(self.generator_sha256) != 64:
            raise ValidationShardError("generator_sha256 must be a 64-char sha256")

    @property
    def walls(self) -> int:
        return self.games // 4


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_engine_log(path: str | Path) -> list[dict]:
    """Read one gzip MJAI engine log (full-information raw events)."""
    target = Path(path)
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValidationShardError(f"empty engine log {target}")
    return rows


def scenario_counts(raw_events: Sequence[dict]) -> dict:
    """Public-event scenario coverage of one full game log (counts only)."""
    counts = {"kyoku": 0, "reach": 0, "chi": 0, "pon": 0, "daiminkan": 0,
              "ankan": 0, "kakan": 0, "hora": 0, "ryukyoku": 0, "dora": 0,
              "multi_ron": 0, "triple_ron": 0, "four_riichi": 0,
              "end_kyotaku": 0}
    reach_in_kyoku = 0
    ron_run = 0
    kyotaku = 0
    for event in raw_events:
        kind = event.get("type")
        if kind == "start_kyoku":
            counts["kyoku"] += 1
            reach_in_kyoku = 0
            kyotaku = int(event.get("kyotaku", 0))
        elif kind == "reach":
            counts["reach"] += 1
            reach_in_kyoku += 1
            kyotaku += 1
            if reach_in_kyoku == 4:
                counts["four_riichi"] += 1
        elif kind in ("chi", "pon", "daiminkan", "ankan", "kakan"):
            counts[kind] += 1
        elif kind == "dora":
            counts["dora"] += 1
        elif kind == "dahai":
            ron_run = 0
        elif kind == "tsumo":
            ron_run = 0
        elif kind == "hora":
            counts["hora"] += 1
            ron_run += 1
            kyotaku = 0
        elif kind == "ryukyoku":
            counts["ryukyoku"] += 1
            if ron_run >= 2:
                counts["multi_ron"] += 1
                if ron_run >= 3:
                    counts["triple_ron"] += 1
            ron_run = 0
        elif kind == "end_kyoku":
            if ron_run >= 2:
                counts["multi_ron"] += 1
                if ron_run >= 3:
                    counts["triple_ron"] += 1
            ron_run = 0
    if kyotaku > 0:
        counts["end_kyotaku"] += 1
    return counts


def _scan(games_dir: Path, spec: ValidationShardSpec) -> list[dict]:
    files = []
    for path in sorted(games_dir.glob("*.json.gz")):
        match = LOG_NAME_RE.match(path.name)
        if match is None:
            raise ValidationShardError(f"unexpected engine log name {path.name!r}")
        seed = int(match.group("seed"))
        key = int(match.group("key"))
        rot = match.group("rot")
        if key != spec.seed_key:
            raise ValidationShardError(
                f"{path.name}: key {key} != requested seed_key {spec.seed_key}")
        if not spec.seed_start <= seed < spec.seed_start + spec.walls:
            raise ValidationShardError(
                f"{path.name}: seed {seed} outside window "
                f"[{spec.seed_start},{spec.seed_start + spec.walls})")
        files.append({"path": path, "seed": seed, "key": key, "rotation": rot,
                      "game_id": f"{seed}:{key}:{ROTATIONS.index(rot)}",
                      "sha256": _sha_file(path)})
    if len(files) != spec.games:
        raise ValidationShardError(
            f"found {len(files)} engine logs, expected {spec.games}")
    per_seed: dict[int, list[str]] = {}
    for item in files:
        per_seed.setdefault(item["seed"], []).append(item["rotation"])
    for seed, rotations in sorted(per_seed.items()):
        if sorted(rotations) != list(ROTATIONS):
            raise ValidationShardError(
                f"seed {seed}: rotations {sorted(rotations)} != {list(ROTATIONS)}")
    files.sort(key=lambda item: (item["seed"], item["rotation"]))
    return files


def scan_engine_logs(games_dir: str | Path,
                     spec: ValidationShardSpec) -> list[dict]:
    """Validate log names/window/rotations and return the ordered file index."""
    return _scan(Path(games_dir), spec)


def _corpus_manifest(games: Sequence[dict], spec: ValidationShardSpec) -> dict:
    return {
        "format": "rf-validation-corpus/1",
        "ruleset_id": spec.ruleset_id,
        "seed_start": spec.seed_start,
        "games": spec.games,
        "seed_key": spec.seed_key,
        "lineup": list(spec.lineup),
        "generator_sha256": spec.generator_sha256,
        "game_logs": [[item["game_id"], item["sha256"]] for item in games],
    }


def build_validation_shard(*, games_dir: str | Path, out_shard: str | Path,
                           spec: ValidationShardSpec,
                           policy: PolicyConfig, strict: bool = True,
                           progress: Callable[[str], None] | None = None) -> dict:
    """Build the diagnostic shard; returns the full report (no write outside out)."""
    if not isinstance(policy, PolicyConfig):
        raise ValidationShardError("policy must be a PolicyConfig")
    if policy.profile != "operator_private_research":
        raise ValidationShardError(
            "a validation shard must use operator_private_research (the private "
            "profile records the real rights facts without claiming a licence)")
    if policy.release_allowed or policy.network_upload_allowed:
        raise ValidationShardError(
            "validation corpus must stay local (release/upload must be false)")
    games_dir = Path(games_dir)
    out_shard = Path(out_shard)
    if out_shard.exists():
        raise ValidationShardError(f"refusing to overwrite existing shard {out_shard}")
    games = _scan(games_dir, spec)
    manifest = _corpus_manifest(games, spec)
    manifest_sha = _sha_text(_canonical(manifest))
    index_sha = _sha_text(_canonical([[item["game_id"], item["sha256"]]
                                      for item in games]))
    window_sha = _sha_text(_canonical({
        "seed_start": spec.seed_start, "games": spec.games,
        "seed_key": spec.seed_key, "ruleset_id": spec.ruleset_id,
        "lineup": list(spec.lineup), "generator_sha256": spec.generator_sha256,
    }))
    metadata = ShardMetadata(
        purpose="diagnostic",
        source_id="selfplay_own",
        rights_status="UNKNOWN",
        external_license_status="UNKNOWN",
        train_allowed=False,
        train_eligible_under_policy=False,
        research_policy="operator_private_research",
        research_policy_hash=policy.policy_hash,
        policy_version=policy.policy_version,
        canonical_manifest_sha256=manifest_sha,
        canonical_index_sha256=index_sha,
        split_manifest_sha256=window_sha,
        split_version="rf-selfplay-validation/1",
        schema_version=FROZEN_SCHEMA["schema_version"],
        schema_hash=SCHEMA_HASH,
        action_map_hash=ACTION_HASH,
        normalizer_hash=NORMALIZER_HASH,
        ruleset_id=RulesetId.MORTAL_ARENA_PINNED.value,
        ruleset_deviations=ARENA_DEVIATIONS,
        source_kind=SourceKind.ARENA_SELFPLAY.value,
    )
    config = DecisionBuildConfig(
        ruleset_id=RulesetId.MORTAL_ARENA_PINNED,
        source_kind=SourceKind.ARENA_SELFPLAY,
        ruleset_deviation=DECISION_DEVIATIONS,
    )

    writer = ShardWriter(out_shard, metadata)
    game_rows = []
    scenario_totals: dict[str, int] = {}
    visibility = {"streams": 0, "events": 0, "unknown_tiles": 0,
                  "unknown_on_hidden_events": 0, "violation_count": 0}
    decisions = 0
    try:
        for item in games:
            raw = read_engine_log(item["path"])
            replay = replay_scores(raw, strict_protocol=False)
            if not replay.ok or replay.scores is None:
                raise ValidationShardError(
                    f"{item['game_id']}: score replay failed: {replay.reason}")
            scores = tuple(int(value) for value in replay.scores)
            if sum(scores) != 100_000:
                raise ValidationShardError(
                    f"{item['game_id']}: score sum {sum(scores)} != 100000")
            samples = build_decisions(raw, config)
            if not samples:
                raise ValidationShardError(f"{item['game_id']}: no decisions")
            record = make_game_record(
                raw, samples, game_id=item["game_id"],
                component_id=f"selfplay-log:{item['sha256'][:16]}",
                component_sha256=item["sha256"], split="validation",
                source_id="selfplay_own", train_allowed=False,
                rights_status="UNKNOWN")
            check = assert_visibility_contract(
                record.pov_streams, game_id=item["game_id"], strict=strict)
            visibility["streams"] += check["streams"]
            visibility["events"] += check["events"]
            visibility["unknown_tiles"] += check["unknown_tiles"]
            visibility["unknown_on_hidden_events"] += check["unknown_on_hidden_events"]
            visibility["violation_count"] += check["violation_count"]
            writer.add_game(record)
            decisions += len(record.decisions)
            scenarios = scenario_counts(raw)
            for key, value in scenarios.items():
                scenario_totals[key] = scenario_totals.get(key, 0) + value
            ranks = _ranks(scores)
            game_rows.append({
                "game_id": item["game_id"], "seed": item["seed"],
                "rotation": item["rotation"], "log_sha256": item["sha256"],
                "decisions": len(record.decisions),
                "scores": list(scores), "score_sum": sum(scores),
                "ranks": list(ranks), "scenarios": scenarios,
                "pov_stream_events": [len(stream) for stream in record.pov_streams],
            })
            if progress is not None:
                progress(f"{item['game_id']} decisions={len(record.decisions)}")
        shard_manifest = writer.finalize()
    finally:
        if not writer._closed:
            writer.close()
    integrity = verify_shard_integrity(out_shard, metadata=metadata, strict=strict)
    hashes = sample_hashes(out_shard)
    if integrity["inputs"] != decisions:
        raise ValidationShardError(
            f"stored inputs {integrity['inputs']} != aggregated decisions {decisions}")
    if visibility["violation_count"]:
        raise ValidationShardError(
            f"visibility audit recorded {visibility['violation_count']} violations")
    return {
        "format": "rf-validation-shard/1",
        "shard": str(out_shard),
        "shard_sha256": shard_manifest["shard_sha256"],
        "content_sha256": shard_manifest["content_sha256"],
        "manifest_sha256": _sha_file(Path(str(out_shard) + ".manifest.json")),
        "game_count": shard_manifest["game_count"],
        "decision_count": shard_manifest["decision_count"],
        "sample_hashes": hashes,
        "metadata": metadata.as_dict(),
        "hash_derivation": dict(SELFPLAY_HASH_DERIVATION),
        "corpus_manifest": manifest,
        "corpus_manifest_sha256": manifest_sha,
        "canonical_index_sha256": index_sha,
        "split_manifest_sha256": window_sha,
        "games": game_rows,
        "scenario_totals": dict(sorted(scenario_totals.items())),
        "visibility": visibility,
        "integrity": integrity,
        "policy": {"profile": policy.profile, "hash": policy.policy_hash,
                   "version": policy.policy_version},
        "note": (
            "diagnostic-only corpus generated by weak local heuristic bots; "
            "not a strong teacher, not Tenhou truth, and not training-eligible"),
    }


def _ranks(scores: Sequence[int]) -> tuple[int, int, int, int]:
    order = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
    ranks = [0, 0, 0, 0]
    for rank, seat in enumerate(order, start=1):
        ranks[seat] = rank
    return tuple(ranks)  # type: ignore[return-value]
