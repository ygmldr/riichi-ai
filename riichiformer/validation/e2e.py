#!/usr/bin/env python3
"""PR-47 end-to-end roundtrip on real native games.

``native self-play -> four POV streams -> decision snapshots -> diagnostic shard v3
-> multi-decision loader -> 2-step six-layer CPU train -> checkpoint -> reload ->
inference``

Everything below runs on CPU.  The model is the frozen MVP-S (5,217,582
parameters) and the optimizer is AdamW from the frozen training module; only two
optimizer steps are taken, so this is a plumbing roundtrip, not a training run and
not a strength result.

The module also carries the DB-level negative controls used by the tests: a
tampered row hash, a tampered public state and a tampered label must all fail
closed (see :func:`tamper_target_label`, :func:`tamper_public_state`,
:func:`tamper_cutoff`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import torch

from ..data.loader import ShardLoadError
from ..data.shard import ShardError
from ..models.policy import Policy, count_parameters
from ..policy.config import PolicyConfig
from ..schemas.decision_v3 import RulesetId, SourceKind
from ..train.bc import training_step_multi
from ..train.checkpoint import load_checkpoint, save_checkpoint
from ..train.model_config import resolve_mvp_config
from ..train.optimizer import TrainingConfig, make_optimizer, make_scheduler, set_seed
from .integrity import IntegrityError, verify_shard_integrity
from .reader import ValidationError, ValidationReader
from .shard_build import (
    ARENA_DEVIATIONS,
    ValidationShardSpec,
    build_validation_shard,
)

__all__ = [
    "E2EError",
    "expected_parameter_count",
    "run_roundtrip",
    "tamper_cutoff",
    "tamper_public_state",
    "tamper_target_label",
]

EXPECTED_PARAMETERS = 5_217_582


class E2EError(RuntimeError):
    """The PR-47 roundtrip failed a structural invariant."""


def expected_parameter_count() -> int:
    return EXPECTED_PARAMETERS


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# DB negative controls (copy first; never mutate the real shard)
# ---------------------------------------------------------------------------
def _copy_shard(source: Path, target: Path) -> Path:
    shutil.copy2(source, target)
    shutil.copy2(Path(str(source) + ".manifest.json"),
                 Path(str(target) + ".manifest.json"))
    _refresh_manifest(target)
    return target


def _refresh_manifest(path: Path) -> None:
    """Recompute the shard hash after a DB edit (otherwise the reader refuses it)."""
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shard_sha256"] = _sha_file(path)
    manifest["files"] = {Path(path).name: manifest["shard_sha256"]}
    manifest_path.write_text(_canonical(manifest) + "\n", encoding="utf-8")


def _first_decision(conn: sqlite3.Connection, where: str = "") -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM inputs " + where + " ORDER BY decision_id LIMIT 1").fetchone()
    if row is None:
        raise E2EError("no decision row to tamper")
    return row


def tamper_target_label(source: str | Path, target: str | Path) -> Path:
    """Inject a different label without rewriting ``target_sha256``."""
    out = _copy_shard(Path(source), Path(target))
    conn = sqlite3.connect(out)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT i.* FROM inputs i JOIN targets t USING(decision_id) "
            "WHERE t.supervision_valid=1 ORDER BY i.decision_id LIMIT 1").fetchone()
        if row is None:
            raise E2EError("no supervised decision row to tamper")
        current = conn.execute(
            "SELECT expert_action FROM targets WHERE decision_id=?",
            (row["decision_id"],)).fetchone()[0]
        injected = (int(current) + 1) % 46 if current is not None else 0
        conn.execute(
            "UPDATE targets SET expert_action=? WHERE decision_id=?",
            (injected, row["decision_id"]))
        conn.commit()
    finally:
        conn.close()
    _refresh_manifest(out)
    return out


def tamper_public_state(source: str | Path, target: str | Path) -> Path:
    """Change a public snapshot field and recompute the input hash (consistent DB)."""
    from ..data.shard import pack_json, unpack_json

    out = _copy_shard(Path(source), Path(target))
    conn = sqlite3.connect(out)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM inputs WHERE before_event_idx > 4 "
            "ORDER BY decision_id LIMIT 1").fetchone()
        if row is None:
            raise E2EError("no decision row to tamper")
        public = unpack_json(row["global_public"])
        public["kyotaku"] = int(public["kyotaku"]) + 1
        payload = {
            "before_event_idx": int(row["before_event_idx"]),
            "kyoku_idx": int(row["kyoku_idx"]),
            "phase": str(row["phase"]),
            "legal_mask": unpack_json(row["legal_mask"]),
            "tile_state": unpack_json(row["tile_state"]),
            "players_public": unpack_json(row["players_public"]),
            "global_public": public,
        }
        digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        conn.execute(
            "UPDATE inputs SET global_public=?, input_sha256=? WHERE decision_id=?",
            (pack_json(public), digest, row["decision_id"]))
        conn.commit()
    finally:
        conn.close()
    _refresh_manifest(out)
    return out


def tamper_cutoff(source: str | Path, target: str | Path) -> Path:
    """Move a cutoff by one event and recompute the input hash (loader must refuse)."""
    from ..data.shard import pack_json, unpack_json

    out = _copy_shard(Path(source), Path(target))
    conn = sqlite3.connect(out)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM inputs WHERE before_event_idx > 8 "
            "ORDER BY decision_id LIMIT 1").fetchone()
        if row is None:
            raise E2EError("no decision row to tamper")
        payload = {
            "before_event_idx": int(row["before_event_idx"]) - 1,
            "kyoku_idx": int(row["kyoku_idx"]),
            "phase": str(row["phase"]),
            "legal_mask": unpack_json(row["legal_mask"]),
            "tile_state": unpack_json(row["tile_state"]),
            "players_public": unpack_json(row["players_public"]),
            "global_public": unpack_json(row["global_public"]),
        }
        digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        conn.execute(
            "UPDATE inputs SET before_event_idx=?, input_sha256=? WHERE decision_id=?",
            (payload["before_event_idx"], digest, row["decision_id"]))
        conn.commit()
    finally:
        conn.close()
    _refresh_manifest(out)
    return out


# ---------------------------------------------------------------------------
# roundtrip
# ---------------------------------------------------------------------------
def run_roundtrip(*, games_dir: str | Path, shard_path: str | Path,
                  workdir: str | Path, policy: PolicyConfig,
                  spec: ValidationShardSpec, steps: int = 2,
                  batch_size: int = 2, seed: int = 20261121,
                  device: str = "cpu") -> dict:
    """Build the shard and run the full CPU roundtrip; returns a JSON report."""
    if steps <= 0 or steps > 10:
        raise E2EError("steps must be in 1..10 for this smoke")
    if device != "cpu":
        raise E2EError("PR-47 roundtrip is CPU-only in this lane")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    shard_path = Path(shard_path)
    build = build_validation_shard(
        games_dir=games_dir, out_shard=shard_path, spec=spec, policy=policy)
    if build["game_count"] != spec.games:
        raise E2EError(
            f"shard has {build['game_count']} games, corpus {spec.games}")
    for game in build["games"]:
        if game["score_sum"] != 100_000:
            raise E2EError(f"{game['game_id']}: score sum != 100000")
    metadata = build["metadata"]
    if metadata["ruleset_id"] != RulesetId.MORTAL_ARENA_PINNED.value:
        raise E2EError("roundtrip shard is not mortal_arena_pinned")
    if metadata["source_kind"] != SourceKind.ARENA_SELFPLAY.value:
        raise E2EError("roundtrip shard is not arena_selfplay")
    if metadata["purpose"] != "diagnostic":
        raise E2EError("roundtrip shard is not diagnostic")
    if not set(ARENA_DEVIATIONS) <= set(metadata["ruleset_deviations"]):
        raise E2EError("roundtrip shard lost the arena deviation provenance")

    set_seed(seed)
    resolved = resolve_mvp_config()
    model_config = resolved.config
    parameters = count_parameters(Policy(model_config))
    if parameters != EXPECTED_PARAMETERS:
        raise E2EError(
            f"model parameter count {parameters} != {EXPECTED_PARAMETERS}")
    model = Policy(model_config).to(device)
    config = TrainingConfig(seed=seed, lr=3e-4, bf16=False, device="cpu",
                            warmup_steps=0)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)

    losses = []
    decisions = 0
    tokens = 0
    kept_batch = None
    reader = ValidationReader(
        shard_path, expected_policy_hash=policy.policy_hash,
        batch_size=batch_size, max_positions=model_config.max_positions)
    try:
        describe = reader.describe()
        if describe["train_eligible_under_policy"]:
            raise E2EError("validation shard became train-eligible")
        model.train()
        iterator = reader.iter_batches()
        while len(losses) < steps:
            try:
                item = next(iterator)
            except StopIteration:
                raise E2EError(
                    f"loader ran out of batches after {len(losses)} steps")
            if kept_batch is None:
                kept_batch = item
            result = training_step_multi(
                model, item.batch.batch.to(device), item.batch.targets,
                optimizer=optimizer, scheduler=scheduler,
                max_grad_norm=config.grad_clip, bf16=False)
            if result.valid_decisions == 0:
                continue
            losses.append(float(result.loss.detach()))
            decisions += result.valid_decisions
            tokens += item.batch.tokens
        if kept_batch is None:
            raise E2EError("no batch was retained for inference")
        model.eval()
        with torch.no_grad():
            before = model.forward_multi(kept_batch.batch.batch.to(device))
            before_logits = before.logits.detach().cpu().clone()
        hashes = {
            "schema_hash": metadata["schema_hash"],
            "action_map_hash": metadata["action_map_hash"],
            "normalizer_hash": metadata["normalizer_hash"],
            "split_manifest_sha256": metadata["split_manifest_sha256"],
            "canonical_manifest_sha256": metadata["canonical_manifest_sha256"],
            "canonical_index_sha256": metadata["canonical_index_sha256"],
            "source_id": metadata["source_id"],
            "research_policy": metadata["research_policy"],
            "research_policy_hash": metadata["research_policy_hash"],
            "policy_version": metadata["policy_version"],
            "external_license_status": metadata["external_license_status"],
            "train_eligible_under_policy": metadata["train_eligible_under_policy"],
            "rights_status": metadata["rights_status"],
        }
        provenance = {key: hashes[key] for key in (
            "source_id", "split_manifest_sha256", "canonical_manifest_sha256",
            "canonical_index_sha256", "research_policy", "research_policy_hash")}
        checkpoint_path = workdir / "pr47_roundtrip.pt"
        info = save_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler,
            counters={"step": steps, "tokens": tokens, "decisions": decisions},
            hashes=hashes, config=config, seed=seed, model_config=model_config,
            provenance=provenance,
            extra={"training_path": "multi_decisions",
                   "token_accounting": "shared_events_plus_markers",
                   "pr47_roundtrip": "validation",
                   "shard_sha256": build["shard_sha256"]})
        reloaded = Policy(model_config).to(device)
        load_checkpoint(checkpoint_path, model=reloaded,
                        expected_model_config=model_config,
                        expected_model_config_hash=resolved.config_hash,
                        expected_hashes={key: hashes[key] for key in
                                         ("schema_hash", "action_map_hash",
                                          "normalizer_hash")})
        reloaded.eval()
        with torch.no_grad():
            after = reloaded.forward_multi(kept_batch.batch.batch.to(device))
        reload_diff = float((before_logits - after.logits.detach().cpu())
                            .abs().max().item())
        if reload_diff != 0.0:
            raise E2EError(
                f"reloaded checkpoint logits differ by {reload_diff:.3e}")
        if build["integrity"]["ok"] is not True:
            raise E2EError("post-build integrity audit failed")
    finally:
        reader.close()

    # The training loader must not open this shard at all.
    training_refused = None
    from ..data.loader import ShardReader
    from ..data.shard import ShardRightsError

    try:
        ShardReader(shard_path, purpose="train",
                    expected_policy_hash=policy.policy_hash)
    except ShardRightsError as exc:
        training_refused = type(exc).__name__
    except ShardError as exc:
        training_refused = type(exc).__name__
    else:
        raise E2EError("training loader opened a diagnostic shard")
    return {
        "format": "rf-pr47-roundtrip/1",
        "passed": True,
        "games": build["game_count"],
        "decisions": build["decision_count"],
        "shard": str(shard_path),
        "shard_sha256": build["shard_sha256"],
        "content_sha256": build["content_sha256"],
        "seed_key": spec.seed_key,
        "seed_key_hex": f"0x{spec.seed_key:016x}",
        "window": f"[{spec.seed_start},{spec.seed_start + spec.walls})",
        "parameter_count": parameters,
        "model_config_hash": resolved.config_hash,
        "steps": steps,
        "step_losses": losses,
        "trained_decisions": decisions,
        "trained_tokens": tokens,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": info.sha256,
        "reload_logit_max_abs": reload_diff,
        "reader": describe,
        "build": {key: build[key] for key in
                  ("game_count", "decision_count", "shard_sha256",
                   "content_sha256", "sample_hashes", "scenario_totals",
                   "visibility", "integrity")},
        "training_loader_refusal": training_refused,
        "gpu_used": False,
    }


__all__ += ["IntegrityError", "ShardLoadError", "ValidationError"]
