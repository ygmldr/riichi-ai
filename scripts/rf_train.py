#!/usr/bin/env python3
"""PR-14 synthetic-first BC trainer CLI.

Default is dry-run.  Training requires a shard whose metadata is purpose=train
and whose policy is authenticated before a model is constructed:

* ``licensed_only`` shards need ``train_allowed=True`` and
  ``rights_status=ALLOWED``;
* ``operator_private_research`` shards need an explicit ``--policy-config`` or
  ``--expected-policy-hash`` plus a matching allowlist, and keep the real
  external rights status as an independent fact (never a licence grant).

Frozen run/checkpoint/data roots are write-refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from riichiformer.data.encode import MAX_EVENTS_DEFAULT  # noqa: E402
from riichiformer.data.loader import ShardLoadError, ShardReader  # noqa: E402
from riichiformer.data.shard import ShardRightsError  # noqa: E402
from riichiformer.models.policy import Policy, count_parameters  # noqa: E402
from riichiformer.policy.config import (  # noqa: E402
    PROFILES, PolicyConfig, PolicyConfigError, load_policy_config,
)
from riichiformer.policy.research_policy import ResearchPolicyError  # noqa: E402
from riichiformer.train.bc import (  # noqa: E402
    TrainingError, training_step, training_step_multi,
)
from riichiformer.train.checkpoint import (  # noqa: E402
    CheckpointMismatch, config_hash, init_weights_only, load_checkpoint,
    save_checkpoint,
)
from riichiformer.train.model_config import (  # noqa: E402
    ModelConfigError, ResolvedModelConfig, resolve_mvp_config,
)
from riichiformer.train.optimizer import (  # noqa: E402
    OptimizerError, TrainingConfig, make_optimizer, make_scheduler, set_seed,
)

FROZEN_PREFIXES = ("datasets", "runs/1v3", "runs/variance", "checkpoints")
PRIVATE_PROFILE = "operator_private_research"


class TrainerRefusal(RuntimeError):
    pass


def _check_output_path(path: Path) -> None:
    resolved = path.resolve()
    for prefix in FROZEN_PREFIXES:
        frozen = (REPO_ROOT / prefix).resolve()
        if resolved == frozen or frozen in resolved.parents:
            raise TrainerRefusal(f"refusing to write frozen path {resolved}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)


def _resolve_model_config(args) -> ResolvedModelConfig:
    """MVP-S by default; tiny only when explicitly requested."""
    overrides = any(value is not None for value in
                    (args.layers, args.ffn, args.state_hidden, args.max_positions))
    if args.tiny_smoke:
        if args.model_config is not None or overrides:
            raise TrainerRefusal(
                "--tiny-smoke cannot be combined with --model-config or explicit "
                "architecture overrides")
        return resolve_mvp_config(tiny_smoke=True)
    if args.model_config is not None:
        if overrides:
            raise TrainerRefusal(
                "--model-config cannot be combined with --layers/--ffn/"
                "--state-hidden/--max-positions; put the architecture in the YAML")
        return resolve_mvp_config(args.model_config)
    if overrides:
        # Start from the frozen MVP-S and replace only the supplied fields.  A
        # partial override must not silently shrink the untouched dimensions to
        # the old tiny defaults.
        base = resolve_mvp_config()
        updates = {}
        if args.layers is not None:
            updates["layers"] = args.layers
        if args.ffn is not None:
            updates["ffn"] = args.ffn
        if args.state_hidden is not None:
            updates["state_hidden"] = args.state_hidden
        if args.max_positions is not None:
            updates["max_positions"] = args.max_positions
        config = dataclasses.replace(base.config, **updates)
        model = Policy(config)
        return ResolvedModelConfig(
            config=config, parameter_count=count_parameters(model),
            config_hash=config_hash(config), source_path=None,
            tiny_smoke=False, reference_verified=False)
    return resolve_mvp_config()


def _resolve_policy(args) -> tuple[PolicyConfig | None, str | None]:
    """Resolve the expected policy hash from --policy-config/--expected-policy-hash."""
    policy = None
    config_path = getattr(args, "policy_config", None)
    if config_path:
        policy = load_policy_config(config_path)
    requested = getattr(args, "policy", None)
    if requested is not None:
        if requested not in PROFILES:
            raise TrainerRefusal(f"unknown policy profile {requested!r}")
        if policy is not None and policy.profile != requested:
            raise TrainerRefusal(
                f"--policy {requested!r} does not match --policy-config profile "
                f"{policy.profile!r}")
        if policy is None and requested == PRIVATE_PROFILE:
            raise TrainerRefusal(
                "--policy operator_private_research requires --policy-config")
    expected = None
    if policy is not None and policy.profile == PRIVATE_PROFILE:
        expected = policy.policy_hash
    cli_expected = (getattr(args, "expected_policy_hash", None) or "").strip()
    if cli_expected:
        value = cli_expected.lower()
        if not _is_sha256(value):
            raise TrainerRefusal(
                "--expected-policy-hash must be a 64-character hex sha256")
        if policy is not None:
            if policy.profile != PRIVATE_PROFILE:
                raise TrainerRefusal(
                    "--expected-policy-hash is only valid with an "
                    "operator_private_research policy config")
            if value != policy.policy_hash:
                raise TrainerRefusal(
                    "--expected-policy-hash does not match --policy-config "
                    "policy_hash")
        expected = value
    return policy, expected


def _load_reader(path: str, policy: PolicyConfig | None = None,
                 expected_policy_hash: str | None = None) -> ShardReader:
    try:
        reader = ShardReader(
            path, purpose="train",
            expected_policy_hash=expected_policy_hash,
            allowed_local_sources=(policy.allowed_local_sources
                                   if policy is not None else None))
    except (ShardRightsError, ShardLoadError) as exc:
        raise TrainerRefusal(f"rights gate refused shard: {exc}") from exc
    if policy is not None and reader.metadata.research_policy != policy.profile:
        reader.close()
        raise TrainerRefusal(
            f"policy config profile {policy.profile!r} cannot consume shard "
            f"profile {reader.metadata.research_policy!r}")
    return reader


def run(args) -> int:
    if args.out:
        _check_output_path(Path(args.out))
    policy, expected_policy_hash = _resolve_policy(args)
    reader = _load_reader(args.shard, policy, expected_policy_hash)
    meta = reader.metadata
    policy_summary = (
        f"research_policy={meta.research_policy} "
        f"policy_version={meta.policy_version!r} "
        f"policy_hash={meta.research_policy_hash!r} "
        f"external_license_status={meta.external_license_status} "
        f"train_eligible_under_policy={meta.train_eligible_under_policy}")
    if meta.research_policy == PRIVATE_PROFILE:
        print("PR14_POLICY_NOTE research_policy=operator_private_research "
              "operational_allow=True "
              "basis=operator_attestation_not_external_license "
              f"external_license_status={meta.external_license_status}; this is "
              "not an external license or redistribution grant")
    try:
        training_path = "multi_decisions" if args.multi_decision else "prefixes"
        if args.dry_run or not args.train:
            batches = 0
            decisions = 0
            tokens = 0
            if args.multi_decision:
                dry_max_events = args.multi_max_events or MAX_EVENTS_DEFAULT
                dry_max_positions = args.multi_max_positions or None
                for batch in reader.iter_multi_decision_batches(
                        batch_size=args.batch_size,
                        max_events=dry_max_events,
                        max_positions=dry_max_positions,
                        verify_state_snapshots=args.multi_deep_verify):
                    batches += 1
                    decisions += len(batch.targets)
                    tokens += batch.tokens
                    if args.steps and batches >= args.steps:
                        break
                print(f"PR14_MULTI_DRY_RUN batches={batches} "
                      f"decisions={decisions} tokens={tokens} shard={args.shard} "
                      f"purpose={meta.purpose} source={meta.source_id} "
                      f"max_events={dry_max_events} "
                      f"max_positions={dry_max_positions} {policy_summary}")
                return 0
            for batch in reader.iter_batches(batch_size=args.batch_size):
                batches += 1
                decisions += len(batch.targets)
                if args.steps and batches >= args.steps:
                    break
            print(f"PR14_DRY_RUN shard={args.shard} batches={batches} "
                  f"decisions={decisions} purpose={meta.purpose} "
                  f"source={meta.source_id} {policy_summary}")
            return 0
        if not args.out:
            raise TrainerRefusal("--train requires --out")
        config = TrainingConfig(seed=args.seed, lr=args.lr, bf16=args.bf16,
                                device=args.device, warmup_steps=args.warmup)
        if args.cpu:
            config = TrainingConfig(**{**config.__dict__, "device": "cpu", "bf16": False})
        set_seed(config.seed)
        resolved = _resolve_model_config(args)
        model_config = resolved.config
        print(f"PR14_MODEL {resolved.describe()}")
        multi_max_events = args.multi_max_events or model_config.max_positions
        multi_max_positions = args.multi_max_positions or model_config.max_positions
        if args.multi_decision:
            if multi_max_events > model_config.max_positions:
                raise TrainerRefusal(
                    f"--multi-max-events {multi_max_events} exceeds model "
                    f"max_positions {model_config.max_positions}")
            if multi_max_positions > model_config.max_positions:
                raise TrainerRefusal(
                    f"--multi-max-positions {multi_max_positions} exceeds model "
                    f"max_positions {model_config.max_positions}")
            print(f"PR14_MULTI_MODEL {resolved.describe()} "
                  f"training_path={training_path} "
                  f"max_events={multi_max_events} "
                  f"max_positions={multi_max_positions}")
        model = Policy(model_config).to(config.device)
        if args.init_weights_only:
            if args.resume:
                raise TrainerRefusal(
                    "--init-weights-only and --resume are separate paths")
            init_weights_only(
                args.init_weights_only, model=model,
                expected_hashes={key: getattr(meta, key) for key in
                                 ("schema_hash", "action_map_hash", "normalizer_hash")},
                expected_model_config=model_config,
                expected_model_config_hash=resolved.config_hash)
            print(f"PR14_INIT_WEIGHTS_ONLY checkpoint={args.init_weights_only}")
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        step = tokens = decisions = 0
        hashes = {"schema_hash": meta.schema_hash,
                  "action_map_hash": meta.action_map_hash,
                  "normalizer_hash": meta.normalizer_hash,
                  "split_manifest_sha256": meta.split_manifest_sha256,
                  "canonical_manifest_sha256": meta.canonical_manifest_sha256,
                  "canonical_index_sha256": meta.canonical_index_sha256,
                  "source_id": meta.source_id,
                  "research_policy": meta.research_policy,
                  "research_policy_hash": meta.research_policy_hash,
                  "policy_version": meta.policy_version,
                  "external_license_status": meta.external_license_status,
                  "train_eligible_under_policy": meta.train_eligible_under_policy,
                  "rights_status": meta.rights_status}
        if args.resume:
            expected = {key: hashes[key] for key in
                        ("schema_hash", "action_map_hash", "normalizer_hash")}
            provenance_keys = ("split_manifest_sha256", "source_id",
                               "research_policy", "research_policy_hash", "rights_status",
                               "policy_version", "external_license_status",
                               "train_eligible_under_policy")
            expected_provenance = {key: hashes[key] for key in provenance_keys}
            info, payload = load_checkpoint(
                args.resume, model=model, optimizer=optimizer, scheduler=scheduler,
                expected_hashes=expected, expected_provenance=expected_provenance,
                expected_model_config=model_config,
                expected_model_config_hash=resolved.config_hash,
                strict_resume=True)
            stored_path = (payload.get("extra") or {}).get("training_path")
            # Legacy checkpoints predate the field; they can only be prefix
            # checkpoints.  Defaulting to "prefixes" makes a prefix -> multi
            # resume fail closed instead of silently rewriting the bookkeeping.
            if stored_path is None:
                stored_path = "prefixes"
            if not isinstance(stored_path, str) or stored_path != training_path:
                raise CheckpointMismatch(
                    f"checkpoint training_path {stored_path!r} != requested "
                    f"{training_path!r}; resume the same training path")
            step, tokens, decisions = info.step, info.tokens, info.decisions
        if args.multi_decision:
            batches = reader.iter_multi_decision_batches(
                batch_size=args.batch_size, max_events=multi_max_events,
                max_positions=multi_max_positions,
                verify_state_snapshots=args.multi_deep_verify)
        else:
            batches = reader.iter_batches(batch_size=args.batch_size)
        model.train()
        while step < args.max_steps:
            try:
                batch = next(batches)
            except StopIteration:
                break
            model_batch = batch.batch.to(config.device)
            if args.multi_decision:
                result = training_step_multi(
                    model, model_batch, batch.targets, optimizer=optimizer,
                    scheduler=scheduler, max_grad_norm=config.grad_clip,
                    bf16=config.bf16)
                batch_tokens = batch.tokens
            else:
                result = training_step(
                    model, model_batch, batch.targets, optimizer=optimizer,
                    scheduler=scheduler, max_grad_norm=config.grad_clip,
                    bf16=config.bf16)
                batch_tokens = int(batch.batch.padding_mask.sum())
            if result.valid_decisions == 0:
                continue
            step += 1
            decisions += result.valid_decisions
            tokens += batch_tokens
            if args.multi_decision:
                print(f"PR14_MULTI_STEP step={step} "
                      f"loss={float(result.loss):.6f} "
                      f"valid={result.valid_decisions}/{result.total_decisions} "
                      f"tokens={batch_tokens} decisions={len(batch.targets)}")
            else:
                print(f"PR14_STEP step={step} loss={float(result.loss):.6f} "
                      f"valid={result.valid_decisions}/{result.total_decisions}")
        provenance = {key: hashes[key] for key in (
            "source_id", "split_manifest_sha256", "canonical_manifest_sha256",
            "canonical_index_sha256", "research_policy", "research_policy_hash")}
        info = save_checkpoint(
            args.out, model=model, optimizer=optimizer, scheduler=scheduler,
            counters={"step": step, "tokens": tokens, "decisions": decisions},
            hashes=hashes, config=config, seed=config.seed,
            model_config=model_config, provenance=provenance,
            extra={"training_path": training_path,
                   "token_accounting": (
                       "shared_events_plus_markers" if args.multi_decision
                       else "prefix_decision_tokens"),
                   "shard_sha256": reader.manifest.get("shard_sha256", "")})
        if args.multi_decision:
            print(f"PR14_MULTI_DONE step={step} tokens={tokens} "
                  f"decisions={decisions} training_path={training_path}")
        print(f"PR14_DONE step={step} tokens={tokens} decisions={decisions} "
              f"sha256={info.sha256} model_sha256={resolved.config_hash} "
              f"parameters={resolved.parameter_count} {policy_summary}")
        return 0
    finally:
        reader.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--train", action="store_true",
                        help="actually train (requires --out; the shard policy is "
                             "authenticated before a model is built)")
    parser.add_argument("--multi-decision", action="store_true", default=False,
                        help="PR-38 shared event stream + decision markers instead "
                             "of one prefix row per decision (default: prefixes)")
    parser.add_argument("--multi-max-events", type=int, default=0,
                        help="multi-decision event-prefix cap per POV chunk "
                             "(0 = 4096 for dry-run, model max_positions for train)")
    parser.add_argument("--multi-max-positions", type=int, default=0,
                        help="multi-decision joined event+marker cap per row "
                             "(0 = model max_positions for train; unset for dry-run)")
    parser.add_argument("--multi-deep-verify", action="store_true", default=False,
                        help="replay the full snapshot tracker per decision (audit "
                             "mode; slower) instead of the cheap public-state check")
    parser.add_argument("--policy", choices=PROFILES, default=None,
                        help="expected research profile; private also requires "
                             "--policy-config")
    parser.add_argument("--policy-config", default=None,
                        help="strict YAML/JSON PolicyConfig used to authenticate a "
                             "private shard and check its source allowlist")
    parser.add_argument("--expected-policy-hash", default=None,
                        help="expected 64-char canonical policy hash; an alternative "
                             "to --policy-config for operator_private_research shards")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=0,
                        help="dry-run batch bound; 0 = all")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--model-config", default=None,
                        help="MVP-S YAML config; omitted loads the frozen six-layer "
                             "configs/riichiformer/mvp_s.yaml")
    parser.add_argument("--tiny-smoke", action="store_true",
                        help="explicit one-layer tiny model for smoke tests only")
    parser.add_argument("--init-weights-only", default=None,
                        help="load model weights only (no optimizer/scheduler/RNG)")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--state-hidden", type=int, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (TrainerRefusal, ShardLoadError, ShardRightsError,
            PolicyConfigError, ResearchPolicyError, ModelConfigError) as exc:
        print(f"PR14_REFUSED: {exc}", file=sys.stderr)
        return 2
    except (TrainingError, OptimizerError, CheckpointMismatch) as exc:
        print(f"PR14_FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"PR14_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
