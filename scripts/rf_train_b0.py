#!/usr/bin/env python3
"""PR-48 B0 wrapper: prereg-gated, bounded, two-seed training + validation.

This script is deliberately **not** a general trainer.  It executes one frozen
:class:`~riichiformer.train.b0.B0Protocol` (loaded from ``--prereg``) and refuses
anything the prereg does not authorise:

* the prereg file is mandatory; there is no implicit/default protocol;
* ``--train`` plus an explicit ``--max-steps`` is required for optimization, so
  an accidental invocation cannot start a full 6,191-step epoch;
* the full prereg budget additionally requires ``--allow-full-budget`` (and a
  user GPU authorization, which this lane never has);
* the device is frozen in the prereg, and ``cuda`` additionally requires
  ``--allow-cuda``;
* the train and validation shard SHA-256/manifest SHA-256 must match the prereg;
* the model config hash and parameter count must match the prereg;
* every seed must be one of the two frozen seeds.

It emits, under ``<out-dir>/<experiment_id>/``:

``PREREG.md`` / ``protocol.json``
    the frozen preregistration that was executed,
``run_seed<seed>/validation.jsonl``
    raw/EMA validation metrics at the fixed schedule,
``run_seed<seed>/candidates/step_XXXXXX.pt`` (+ sidecar)
    retained candidate checkpoints at the preregistered steps,
``run_seed<seed>/metrics.json``
    per-seed accounting (steps/tokens/decisions/losses/wall/stop events),
``selection.json``
    the one frozen selection rule applied to the emitted candidates,
``manifest.json``
    the reference manifest with fingerprints, candidate SHAs, stop events and
    explicit ``NOT_RUN`` items.

``--dry-run`` verifies the prereg/data/model and prints the plan without
training.  ``--validate-only --checkpoint ...`` evaluates an existing B0
checkpoint on the validation shard (no optimizer, no new checkpoint).

Exit codes: 0 ok, 1 runtime failure, 2 frozen-protocol refusal, 3 stop condition.
No strength claim is made from loss/CE anywhere in this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from riichiformer.data.loader import ShardLoadError, ShardReader  # noqa: E402
from riichiformer.data.shard import (  # noqa: E402
    ShardError, ShardMetadata, ShardRightsError, read_manifest,
)
from riichiformer.models.policy import MVPConfig, Policy  # noqa: E402
from riichiformer.policy.config import (  # noqa: E402
    PolicyConfig, PolicyConfigError, load_policy_config,
)
from riichiformer.policy.research_policy import ResearchPolicyError  # noqa: E402
from riichiformer.train.b0 import (  # noqa: E402
    B0_TRAINING_PATH, B0Candidate, B0Protocol, B0ProtocolError, B0Refusal,
    B0Selection, B0SelectionRule, B0StopEvent, B0StopMonitor, B0StopPolicy,
    B0StopTriggered, B0TrainSettings, ModelEMA, build_protocol_from_files,
    candidate_filename, checkpoint_provenance, evaluate_candidate, not_run_items,
    resume_checkpoint, run_accumulated_step, save_candidate,
    verify_data_fingerprints, verify_model_fingerprint, write_manifest,
)
from riichiformer.train.bc import TrainingError  # noqa: E402
from riichiformer.train.checkpoint import (  # noqa: E402
    CheckpointError, CheckpointMismatch, load_checkpoint,
)
from riichiformer.train.optimizer import (  # noqa: E402
    OptimizerError, make_optimizer, make_scheduler, set_seed,
)

PRIVATE_PROFILE = "operator_private_research"
FROZEN_WRITE_PREFIXES = ("datasets", "runs", "checkpoints", "vendor", ".git")
#: Default full-epoch budget from the PR-44 shard (6,191 multi-decision batches).
DEFAULT_FULL_BUDGET_STEPS = 6191
#: Default frozen seeds for a newly emitted prereg.
DEFAULT_SEEDS = (20261121, 20261122)
#: Exit code for a frozen stop condition (distinct from refusal/failure).
EXIT_STOPPED = 3


class WrapperRefusal(B0Refusal):
    """CLI-level refusal; still a frozen-protocol refusal (exit 2)."""


# ---------------------------------------------------------------------------
# Frozen paths / policy / readers
# ---------------------------------------------------------------------------
def _check_write_path(path: Path) -> Path:
    resolved = path.resolve()
    for prefix in FROZEN_WRITE_PREFIXES:
        frozen = (REPO_ROOT / prefix).resolve()
        if resolved == frozen or frozen in resolved.parents:
            raise WrapperRefusal(f"refusing to write frozen path {resolved}")
    return resolved


def _check_out_dir(path: Path) -> Path:
    resolved = _check_write_path(path)
    if resolved == REPO_ROOT.resolve():
        raise WrapperRefusal("refusing to use the repository root as a run dir")
    return resolved


def _resolve_policy(protocol: B0Protocol) -> PolicyConfig | None:
    if not protocol.policy_config:
        return None
    policy = load_policy_config(protocol.policy_config)
    if policy.profile != PRIVATE_PROFILE:
        raise WrapperRefusal(
            f"policy config profile {policy.profile!r} is not {PRIVATE_PROFILE!r}")
    if policy.policy_hash != protocol.policy_hash:
        raise WrapperRefusal(
            f"policy config hash {policy.policy_hash} != prereg "
            f"{protocol.policy_hash}")
    return policy


def _shard_metadata(path: str | Path) -> ShardMetadata:
    """Read-only metadata probe (audit purpose; no training authentication)."""
    try:
        reader = ShardReader(path, purpose="audit")
    except (ShardRightsError, ShardLoadError, ShardError) as exc:
        raise WrapperRefusal(f"cannot read shard metadata for {path}: {exc}") from exc
    try:
        return reader.metadata
    finally:
        reader.close()


def _open_reader(protocol: B0Protocol, *, purpose: str,
                 policy: PolicyConfig | None) -> ShardReader:
    path = protocol.train_shard if purpose == "train" else protocol.validation_shard
    try:
        reader = ShardReader(
            path, purpose=purpose,
            expected_policy_hash=(protocol.policy_hash or None),
            allowed_local_sources=(policy.allowed_local_sources
                                   if policy is not None else None))
    except (ShardRightsError, ShardLoadError, ShardError) as exc:
        raise WrapperRefusal(f"{purpose} shard refused: {exc}") from exc
    metadata = reader.metadata
    if metadata.research_policy == PRIVATE_PROFILE:
        if not protocol.policy_hash or policy is None:
            reader.close()
            raise WrapperRefusal(
                "prereg has no operator policy authentication for a private shard")
        if metadata.research_policy_hash != protocol.policy_hash:
            reader.close()
            raise WrapperRefusal("shard policy hash != prereg policy hash")
    elif protocol.policy_hash or protocol.policy_config:
        reader.close()
        raise WrapperRefusal(
            "prereg declares a private policy but the shard is "
            f"research_policy={metadata.research_policy!r}")
    expected = (protocol.train_shard_sha256 if purpose == "train"
                else protocol.validation_shard_sha256)
    if reader.manifest.get("shard_sha256") != expected:
        reader.close()
        raise WrapperRefusal(
            f"{purpose} shard manifest sha256 != prereg (data changed)")
    return reader


def _batch_factory(reader: ShardReader, protocol: B0Protocol):
    positions = int(protocol.model_config["max_positions"])
    return lambda: reader.iter_multi_decision_batches(
        batch_size=protocol.train.batch_size,
        max_events=positions, max_positions=positions)


def _resolve_device(protocol: B0Protocol, args, *, enforce_cuda: bool) -> str:
    device = protocol.train.device
    if args.device is not None and args.device != device:
        raise WrapperRefusal(
            f"--device {args.device!r} != the prereg-frozen device {device!r}; "
            "a different device needs a new prereg")
    if device == "cuda":
        if enforce_cuda and not args.allow_cuda:
            raise WrapperRefusal(
                "the prereg freezes device=cuda; pass --allow-cuda only after the "
                "user authorizes a GPU run")
        if enforce_cuda and not torch.cuda.is_available():
            raise WrapperRefusal("prereg device=cuda but CUDA is not available")
    return device


# ---------------------------------------------------------------------------
# Windows / training
# ---------------------------------------------------------------------------
def _next_window(iterator, protocol: B0Protocol) -> list | None:
    """Pull exactly ``grad_accumulation`` microbatches (no partial windows)."""
    window = []
    for _ in range(protocol.train.grad_accumulation):
        try:
            window.append(next(iterator))
        except StopIteration:
            if not window:
                return None
            raise B0Refusal(
                "training data exhausted inside an accumulation window "
                f"({len(window)}/{protocol.train.grad_accumulation} microbatches)")
    return window


def _seed_dir(out_dir: Path, seed: int) -> Path:
    path = out_dir / f"run_seed{seed}"
    (path / "candidates").mkdir(parents=True, exist_ok=True)
    return path


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_seed(
        protocol: B0Protocol, seed: int, *, out_dir: Path, max_steps: int,
        device: str, policy: PolicyConfig | None, resume: str | None = None,
        verbose: bool = False) -> tuple[dict, list[B0Candidate]]:
    """Run (or strictly resume) one frozen seed; returns (run record, candidates)."""
    if seed not in protocol.seeds:
        raise WrapperRefusal(f"seed {seed} is not one of {protocol.seeds}")
    if not 1 <= max_steps <= protocol.train.max_steps:
        raise WrapperRefusal(
            f"--max-steps {max_steps} outside 1..{protocol.train.max_steps}")
    training_config = protocol.train.training_config(seed)
    set_seed(seed)
    model = Policy(protocol.model_config_object()).to(device)
    optimizer = make_optimizer(model, training_config)
    scheduler = make_scheduler(optimizer, training_config)
    ema = (ModelEMA(model, protocol.train.ema_decay)
           if protocol.train.ema_decay is not None else None)
    if protocol.selection.weights == "ema" and ema is None:
        raise WrapperRefusal(
            "selection weights=ema requires an EMA (train.ema_decay is None)")

    seed_dir = _seed_dir(out_dir, seed)
    validation_path = seed_dir / "validation.jsonl"
    if validation_path.exists() and not resume:
        validation_path.unlink()
    reader = _open_reader(protocol, purpose="train", policy=policy)
    validation_reader = _open_reader(protocol, purpose="audit", policy=policy)
    metadata = reader.metadata
    step = tokens = decisions = 0
    resumed_from = ""
    if resume:
        info, _payload = resume_checkpoint(
            resume, protocol=protocol, metadata=metadata, seed=seed,
            model=model, optimizer=optimizer, scheduler=scheduler, ema=ema,
            strict=True)
        step, tokens, decisions = info.step, info.tokens, info.decisions
        resumed_from = str(resume)
        if step >= max_steps:
            reader.close()
            validation_reader.close()
            raise WrapperRefusal(
                f"checkpoint step {step} is already >= --max-steps {max_steps}")
        print(f"B0_RESUME seed={seed} step={step} tokens={tokens} "
              f"decisions={decisions} checkpoint={resume}")

    losses: list[float] = []
    candidates: list[B0Candidate] = []
    monitor = B0StopMonitor(protocol.stop_policy, device=device)
    started = time.monotonic()
    status = "completed"
    stop_event: B0StopEvent | None = None
    last_loss = float("nan")
    evaluation = None
    try:
        train_iterator = reader.iter_multi_decision_batches(
            batch_size=protocol.train.batch_size,
            max_events=int(protocol.model_config["max_positions"]),
            max_positions=int(protocol.model_config["max_positions"]))
        # A resumed run consumes exactly the same deterministic windows first.
        for skipped in range(step):
            if _next_window(train_iterator, protocol) is None:
                raise WrapperRefusal(
                    f"cannot skip {step} windows for resume; data exhausted at "
                    f"window {skipped}")
        while step < max_steps:
            window = _next_window(train_iterator, protocol)
            if window is None:
                raise WrapperRefusal(
                    f"training data exhausted at step {step} before --max-steps "
                    f"{max_steps}")
            result = run_accumulated_step(
                model, [batch.batch.to(device) for batch in window],
                [batch.targets for batch in window], optimizer=optimizer,
                scheduler=scheduler, settings=protocol.train, ema=ema,
                tokens=sum(batch.tokens for batch in window))
            step += 1
            tokens += int(result.tokens)
            decisions += int(result.valid_decisions)
            losses.append(float(result.loss))
            last_loss = float(result.loss)
            monitor.check_wall(step)
            monitor.check_vram(step)
            if protocol.is_validation_step(step):
                evaluation = evaluate_candidate(
                    model, _batch_factory(validation_reader, protocol),
                    step=step, device=device, ema=ema)
                record = {
                    "step": step,
                    "raw": evaluation.raw.as_json_dict(),
                    "ema": (None if evaluation.ema is None
                            else evaluation.ema.as_json_dict()),
                }
                _append_jsonl(validation_path, record)
                monitor.record_nonfinite(
                    step, evaluation.raw.nonfinite,
                    detail="non-finite model output on the validation shard")
                monitor.record_illegal(
                    step, evaluation.raw.supervised - evaluation.raw.legal,
                    detail="supervised validation target outside the legal mask")
                tracked = (evaluation.raw if protocol.selection.weights == "raw"
                           else evaluation.ema)
                monitor.observe_validation(tracked, step=step)
                if verbose:
                    ema_ce = ("NA" if evaluation.ema is None
                              else f"{evaluation.ema.ce:.6f}")
                    print(f"B0_VALIDATION seed={seed} step={step} "
                          f"raw_ce={evaluation.raw.ce:.6f} ema_ce={ema_ce} "
                          f"top1={evaluation.raw.top1:.4f} "
                          f"top3={evaluation.raw.top3:.4f} "
                          f"legal={evaluation.raw.legal}/"
                          f"{evaluation.raw.supervised}")
            if protocol.is_candidate_step(step):
                if evaluation is None:  # pragma: no cover - protocol invariant
                    raise B0ProtocolError("candidate step is not a validation step")
                path = seed_dir / "candidates" / candidate_filename(step)
                candidate = save_candidate(
                    path, protocol=protocol, metadata=metadata, model=model,
                    optimizer=optimizer, scheduler=scheduler, ema=ema, seed=seed,
                    step=step, tokens=tokens, decisions=decisions,
                    evaluation=evaluation)
                candidates.append(candidate)
                ema_ce = ("NA" if evaluation.ema is None
                          else f"{evaluation.ema.ce:.6f}")
                print(f"B0_CANDIDATE seed={seed} step={step} "
                      f"sha256={candidate.checkpoint_sha256} "
                      f"raw_ce={evaluation.raw.ce:.6f} ema_ce={ema_ce} "
                      f"tokens={tokens} decisions={decisions}")
            if step % 100 == 0:
                print(f"B0_STEP seed={seed} step={step} loss={last_loss:.6f} "
                      f"tokens={tokens}")
        if step != max_steps:
            raise WrapperRefusal(
                f"seed {seed} stopped at step {step} != --max-steps {max_steps}")
    except B0StopTriggered as exc:
        stop_event = exc.event
        status = "stopped"
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - no CUDA here
        stop_event = B0StopEvent("oom", str(exc), step, {})
        status = "stopped"
    except (WrapperRefusal, B0ProtocolError, TrainingError, OptimizerError,
            CheckpointMismatch, ShardLoadError, ShardRightsError):
        status = "incomplete"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        reader.close()
        validation_reader.close()
    wall = time.monotonic() - started
    run = {
        "seed": int(seed),
        "steps": int(step),
        "tokens": int(tokens),
        "decisions": int(decisions),
        "wall_seconds": round(wall, 3),
        "device": device,
        "training_path": B0_TRAINING_PATH,
        "status": status,
        "final_loss": last_loss,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "resumed_from": resumed_from,
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "validation_path": str(validation_path),
        "ema_updates": (ema.updates if ema is not None else None),
        "stop_event": (None if stop_event is None else stop_event.as_dict()),
        "stop_events": [event.as_dict() for event in monitor.events],
        "stop_state": monitor.state_dict(),
    }
    (seed_dir / "metrics.json").write_text(
        json.dumps({**run, "losses": losses}, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"B0_SEED_DONE seed={seed} steps={step} tokens={tokens} "
          f"decisions={decisions} wall={wall:.2f}s status={status} "
          f"candidates={len(candidates)}")
    return run, candidates


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _template_stop_policy(args) -> B0StopPolicy:
    return B0StopPolicy(
        max_wall_seconds=args.max_wall_seconds,
        max_nonfinite_events=0, max_illegal_events=0,
        max_peak_vram_mb=args.max_peak_vram_mb,
        validation_regression_patience=args.regression_patience,
        validation_regression_delta=args.regression_delta)


def _emit_template(args) -> int:
    if not args.out:
        raise WrapperRefusal("--emit-prereg-template requires --out PATH")
    path = Path(args.out)
    _check_write_path(path)
    if not args.shard or not args.validation_shard:
        raise WrapperRefusal(
            "--emit-prereg-template requires --shard and --validation-shard")
    if not args.experiment_id:
        raise WrapperRefusal("--emit-prereg-template requires --experiment-id")
    if args.validation_every <= 0 or args.candidate_every <= 0:
        raise WrapperRefusal("--validation-every/--candidate-every must be > 0")
    if args.candidate_every % args.validation_every:
        raise WrapperRefusal(
            "--candidate-every must be a multiple of --validation-every")
    if args.max_steps is None:
        raise WrapperRefusal("--emit-prereg-template requires --max-steps")
    if args.max_steps <= 0:
        raise WrapperRefusal("--max-steps must be > 0")
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    if len(seeds) != 2 or seeds[0] == seeds[1]:
        raise WrapperRefusal(
            f"a B0 prereg needs exactly two distinct seeds, got {list(seeds)}")
    validation_steps = list(range(args.validation_every, args.max_steps + 1,
                                  args.validation_every))
    if validation_steps[-1] != args.max_steps:
        validation_steps.append(args.max_steps)
    candidate_steps = list(range(args.candidate_every, args.max_steps + 1,
                                 args.candidate_every))
    if candidate_steps[-1] != args.max_steps:
        candidate_steps.append(args.max_steps)

    train_metadata = _shard_metadata(args.shard)
    validation_metadata = _shard_metadata(args.validation_shard)
    policy_hash = ""
    if args.policy_config:
        policy = load_policy_config(args.policy_config)
        if policy.profile != PRIVATE_PROFILE:
            raise WrapperRefusal(
                f"policy config profile {policy.profile!r} is not "
                f"{PRIVATE_PROFILE!r}")
        policy_hash = policy.policy_hash
    for label, metadata in (("train", train_metadata),
                            ("validation", validation_metadata)):
        if metadata.research_policy == PRIVATE_PROFILE:
            if not policy_hash:
                raise WrapperRefusal(
                    f"{label} shard is operator_private_research; the prereg "
                    "needs --policy-config authentication")
            if metadata.research_policy_hash != policy_hash:
                raise WrapperRefusal(
                    f"{label} shard policy hash != --policy-config policy hash")
        elif policy_hash:
            raise WrapperRefusal(
                f"{label} shard is {metadata.research_policy!r}; a private policy "
                "config cannot authenticate it")
    if args.model_config:
        config = MVPConfig.from_yaml(args.model_config)
    else:
        from riichiformer.train.model_config import resolve_mvp_config
        config = resolve_mvp_config().config
    settings = B0TrainSettings(
        lr=args.lr, weight_decay=args.weight_decay, beta1=0.9, beta2=0.999,
        warmup_steps=args.warmup, grad_clip=args.grad_clip,
        grad_accumulation=args.grad_accumulation, batch_size=args.batch_size,
        max_steps=args.max_steps, bf16=args.bf16,
        ema_decay=None if args.no_ema else args.ema_decay,
        training_path=B0_TRAINING_PATH, device=(args.device or "cpu"))
    protocol = build_protocol_from_files(
        experiment_id=args.experiment_id, train_shard=args.shard,
        validation_shard=args.validation_shard, model_config=config,
        train=settings, seeds=seeds, validation_steps=validation_steps,
        candidate_steps=candidate_steps,
        selection=B0SelectionRule(weights=args.selection_weights),
        stop_policy=_template_stop_policy(args),
        policy_config=str(args.policy_config or ""), policy_hash=policy_hash,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    protocol.write_json(path)
    protocol.write_prereg_markdown(path.with_suffix(".md"), base_sha=args.base_sha)
    print(f"B0_PREREG_TEMPLATE path={path} "
          f"fingerprint={protocol.fingerprint()} "
          f"config_fingerprint={protocol.config_fingerprint()} "
          f"parameters={protocol.parameter_count} seeds={list(protocol.seeds)} "
          f"validation_steps={len(validation_steps)} "
          f"candidate_steps={len(candidate_steps)}")
    print("B0_PREREG_TEMPLATE_REVIEW commit/freeze this file before any run; a "
          "changed prereg is a new experiment id")
    return 0


def _load_protocol(args) -> B0Protocol:
    if not args.prereg:
        raise WrapperRefusal(
            "--prereg is required; B0 never runs an implicit/default protocol")
    return B0Protocol.load_json(args.prereg)


def _dry_run(protocol: B0Protocol, args) -> int:
    data = verify_data_fingerprints(protocol)
    parameters = verify_model_fingerprint(protocol)
    device = _resolve_device(protocol, args, enforce_cuda=False)
    policy = _resolve_policy(protocol)
    readers = []
    try:
        for purpose in ("train", "audit"):
            reader = _open_reader(protocol, purpose=purpose, policy=policy)
            readers.append(reader)
            iterator = reader.iter_multi_decision_batches(
                batch_size=protocol.train.batch_size,
                max_events=int(protocol.model_config["max_positions"]),
                max_positions=int(protocol.model_config["max_positions"]))
            if next(iterator, None) is None:
                raise WrapperRefusal(f"{purpose} shard has no trainable batches")
    finally:
        for reader in readers:
            reader.close()
    print(f"B0_DRY_RUN experiment={protocol.experiment_id} "
          f"fingerprint={protocol.fingerprint()} "
          f"config_fingerprint={protocol.config_fingerprint()} "
          f"seeds={list(protocol.seeds)} max_steps={protocol.train.max_steps} "
          f"batch_size={protocol.train.batch_size} "
          f"grad_accumulation={protocol.train.grad_accumulation} "
          f"bf16={protocol.train.bf16} ema_decay={protocol.train.ema_decay} "
          f"training_path={B0_TRAINING_PATH} parameters={parameters} "
          f"device={device} "
          f"validation_steps={len(protocol.validation_steps)} "
          f"candidate_steps={len(protocol.candidate_steps)} "
          f"train_shard_sha256={data['train']['shard_sha256']} "
          f"validation_shard_sha256={data['validation']['shard_sha256']}")
    print("B0_DRY_RUN_OK no model built, no optimizer step, no checkpoint written")
    return 0


def _validate_only(protocol: B0Protocol, args) -> int:
    if not args.checkpoint:
        raise WrapperRefusal("--validate-only requires --checkpoint")
    device = _resolve_device(protocol, args, enforce_cuda=True)
    policy = _resolve_policy(protocol)
    verify_data_fingerprints(protocol)
    verify_model_fingerprint(protocol)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise WrapperRefusal(f"checkpoint not found: {checkpoint}")
    sidecar_path = Path(str(checkpoint) + ".json")
    if not sidecar_path.is_file():
        raise WrapperRefusal(f"checkpoint sidecar missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    seed = int(sidecar.get("seed", -1))
    if seed not in protocol.seeds:
        raise WrapperRefusal(
            f"checkpoint seed {seed} is not one of the prereg seeds")
    metadata = _shard_metadata(protocol.train_shard)
    model = Policy(protocol.model_config_object()).to(device)
    info, payload = load_checkpoint(
        checkpoint, model=model,
        expected_hashes={
            "schema_hash": metadata.schema_hash,
            "action_map_hash": metadata.action_map_hash,
            "normalizer_hash": metadata.normalizer_hash},
        expected_provenance=checkpoint_provenance(protocol, metadata),
        expected_config=protocol.training_config(seed),
        expected_model_config=protocol.model_config,
        expected_model_config_hash=protocol.model_config_hash)
    stored_path = (payload.get("extra") or {}).get("training_path")
    if stored_path is None:
        stored_path = "prefixes"
    if stored_path != B0_TRAINING_PATH:
        raise CheckpointMismatch(
            f"checkpoint training_path {stored_path!r} != {B0_TRAINING_PATH!r}")
    ema = None
    if payload.get("ema") is not None:
        decay = (protocol.train.ema_decay
                 if protocol.train.ema_decay is not None
                 else float(payload["ema"]["decay"]))
        ema = ModelEMA(model, decay)
        ema.load_state_dict(payload["ema"])
    validation_reader = _open_reader(protocol, purpose="audit", policy=policy)
    try:
        evaluation = evaluate_candidate(
            model, _batch_factory(validation_reader, protocol),
            step=int(info.step), device=device, ema=ema)
    finally:
        validation_reader.close()
    out = Path(args.out_dir) if args.out_dir else checkpoint.parent
    out = _check_out_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment_id": protocol.experiment_id,
        "prereg_fingerprint": protocol.fingerprint(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": info.sha256,
        "checkpoint_step": int(info.step),
        "seed": int(info.seed),
        "device": device,
        "raw": evaluation.raw.as_json_dict(),
        "ema": None if evaluation.ema is None else evaluation.ema.as_json_dict(),
        "not_run": list(not_run_items(protocol, [])),
        "strength_claim": "none",
    }
    path = out / "validation_only.json"
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    print(f"B0_VALIDATE step={info.step} seed={info.seed} "
          f"raw_ce={evaluation.raw.ce:.6f} "
          f"ema_ce={'NA' if evaluation.ema is None else f'{evaluation.ema.ce:.6f}'} "
          f"top1={evaluation.raw.top1:.4f} top3={evaluation.raw.top3:.4f} "
          f"legal={evaluation.raw.legal}/{evaluation.raw.supervised} "
          f"written={path}")
    return 0


def _select(protocol: B0Protocol, candidates: list[B0Candidate],
            runs: list[dict]) -> B0Selection:
    complete = (len(runs) == len(protocol.seeds)
                and all(run["status"] == "completed"
                        and run["steps"] == protocol.train.max_steps
                        for run in runs))
    return protocol.selection.select(candidates, complete=complete)


def _run(protocol: B0Protocol, args) -> int:
    if not args.train:
        raise WrapperRefusal(
            "training requires --train; B0 never starts an epoch implicitly")
    if not args.out_dir:
        raise WrapperRefusal("--train requires --out-dir")
    if args.max_steps is None:
        raise WrapperRefusal(
            "training requires an explicit bounded --max-steps; the full budget "
            "is never a default")
    if args.max_steps <= 0:
        raise WrapperRefusal("--max-steps must be > 0")
    if args.max_steps > protocol.train.max_steps:
        raise WrapperRefusal(
            f"--max-steps {args.max_steps} exceeds the prereg budget "
            f"{protocol.train.max_steps}")
    if args.max_steps == protocol.train.max_steps and not args.allow_full_budget:
        raise WrapperRefusal(
            f"--max-steps {args.max_steps} is the full prereg budget; pass "
            "--allow-full-budget only after the user authorizes it")
    device = _resolve_device(protocol, args, enforce_cuda=True)
    seeds = args.seed or list(protocol.seeds)
    if len(set(seeds)) != len(seeds):
        raise WrapperRefusal(f"duplicate --seed values {seeds}")
    for seed in seeds:
        if seed not in protocol.seeds:
            raise WrapperRefusal(
                f"seed {seed} is not one of the frozen prereg seeds "
                f"{protocol.seeds}")
    if args.resume and len(seeds) != 1:
        raise WrapperRefusal("--resume applies to exactly one --seed")
    data = verify_data_fingerprints(protocol)
    parameters = verify_model_fingerprint(protocol)
    policy = _resolve_policy(protocol)
    out_dir = _check_out_dir(Path(args.out_dir)) / protocol.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol.write_json(out_dir / "protocol.json")
    protocol.write_prereg_markdown(out_dir / "PREREG.md", base_sha=args.base_sha)
    print(f"B0_RUN experiment={protocol.experiment_id} "
          f"fingerprint={protocol.fingerprint()} "
          f"config_fingerprint={protocol.config_fingerprint()} "
          f"seeds={seeds} max_steps={args.max_steps} "
          f"prereg_budget={protocol.train.max_steps} parameters={parameters} "
          f"device={device} out_dir={out_dir}")
    runs: list[dict] = []
    candidates: list[B0Candidate] = []
    stop_events: list[B0StopEvent] = []
    exit_code = 0
    for seed in seeds:
        run, seed_candidates = run_seed(
            protocol, seed, out_dir=out_dir, max_steps=args.max_steps,
            device=device, policy=policy, resume=args.resume, verbose=args.verbose)
        runs.append(run)
        candidates.extend(seed_candidates)
        for event in run["stop_events"]:
            stop_events.append(B0StopEvent(
                code=event["code"], detail=event["detail"],
                step=int(event["step"]), metrics=event.get("metrics")))
        if run["stop_event"] is not None:
            exit_code = EXIT_STOPPED
    selection = None
    if candidates:
        selection = _select(protocol, candidates, runs)
        (out_dir / "selection.json").write_text(
            json.dumps(selection.as_dict(), sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"B0_SELECTED candidate={selection.candidate_id} "
              f"seed={selection.seed} step={selection.step} "
              f"weights={selection.weights} ce={selection.value:.6f} "
              f"complete={selection.complete}")
    requested_complete = all(
        run["status"] == "completed" and run["steps"] == args.max_steps
        for run in runs)
    if exit_code == EXIT_STOPPED:
        status = "stopped"
    elif requested_complete and args.max_steps == protocol.train.max_steps:
        status = "completed"
    elif requested_complete:
        status = "partial"
    else:
        status = "incomplete"
    manifest = write_manifest(
        out_dir / "manifest.json", protocol=protocol, runs=runs,
        selection=selection, status=status, stop_events=stop_events,
        candidates=candidates, prereg_path=args.prereg,
        extra={"requested_max_steps": args.max_steps,
               "executed_seeds": [run["seed"] for run in runs],
               "data": {key: {"shard_sha256": value["shard_sha256"],
                              "manifest_sha256": value["manifest_sha256"]}
                        for key, value in data.items()
                        if key in ("train", "validation")}})
    print(f"B0_DONE status={status} runs={len(runs)} "
          f"candidates={len(candidates)} "
          f"not_run={','.join(manifest['not_run'])} "
          f"manifest={out_dir / 'manifest.json'}")
    return exit_code


def run(args) -> int:
    # CPU smoke/validation is single-threaded: the tiny one-layer test models
    # otherwise spend most of their wall time in thread-pool contention (measured
    # ~60x slower with 7 threads than with 1 on this host), and a fixed thread
    # count keeps the bounded CPU runs reproducible.  CUDA runs are unaffected.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if args.emit_prereg_template:
            return _emit_template(args)
        protocol = _load_protocol(args)
        if args.dry_run:
            return _dry_run(protocol, args)
        if args.validate_only:
            return _validate_only(protocol, args)
        return _run(protocol, args)
    finally:
        torch.set_num_threads(previous_threads)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PR-48 B0 wrapper: prereg-gated bounded training, validation "
                    "and selection (no implicit epoch).")
    parser.add_argument("--prereg", default=None,
                        help="frozen rf-b0-prereg/1 JSON (mandatory except "
                             "--emit-prereg-template)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--train", action="store_true", default=False,
                        help="perform optimization (requires --out-dir and a "
                             "bounded --max-steps)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="verify prereg/data/model and print the plan; no "
                             "model construction, no checkpoint")
    parser.add_argument("--validate-only", action="store_true", default=False,
                        help="evaluate --checkpoint on the validation shard "
                             "(forward only, no optimizer step)")
    parser.add_argument("--checkpoint", default=None,
                        help="candidate checkpoint for --validate-only")
    parser.add_argument("--emit-prereg-template", "--prereg-template",
                        dest="emit_prereg_template", action="store_true",
                        default=False,
                        help="emit a complete prereg JSON+MD from real shards")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--shard", default=None,
                        help="train shard (template emission)")
    parser.add_argument("--validation-shard", default=None,
                        help="independent validation shard (template emission)")
    parser.add_argument("--policy-config", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--out", default=None,
                        help="prereg JSON path for --emit-prereg-template")
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="run only these frozen seed(s); repeatable")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="bounded optimizer steps; never defaults to the "
                             "full prereg budget")
    parser.add_argument("--allow-full-budget", action="store_true", default=False,
                        help="permit --max-steps == the prereg budget (needs an "
                             "explicit user GPU authorization)")
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"),
                        help="must match the prereg-frozen device")
    parser.add_argument("--allow-cuda", action="store_true", default=False)
    parser.add_argument("--resume", default=None,
                        help="strict B0 resume from one candidate checkpoint")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--base-sha", default="")
    # template/generation defaults (the frozen B0 prototype settings)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true", default=False)
    parser.add_argument("--selection-weights", default="raw",
                        choices=("raw", "ema"))
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--candidate-every", type=int, default=1000)
    parser.add_argument("--max-wall-seconds", type=float, default=14400.0)
    parser.add_argument("--max-peak-vram-mb", type=float, default=15000.0)
    parser.add_argument("--regression-patience", type=int, default=2)
    parser.add_argument("--regression-delta", type=float, default=0.02)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (WrapperRefusal, B0Refusal, B0ProtocolError, ShardLoadError,
            ShardRightsError, ShardError, PolicyConfigError,
            ResearchPolicyError) as exc:
        print(f"B0_REFUSED: {exc}", file=sys.stderr)
        return 2
    except B0StopTriggered as exc:
        print(f"B0_STOPPED: {exc}", file=sys.stderr)
        return EXIT_STOPPED
    except (CheckpointMismatch, TrainingError, OptimizerError,
            CheckpointError) as exc:
        print(f"B0_FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"B0_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
