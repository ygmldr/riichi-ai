#!/usr/bin/env python3
"""PR-45 read-only validation reader: forward inputs + CE, never an optimizer.

The training loader is already closed to a ``purpose=diagnostic`` shard:
``ShardReader(path, purpose="train")`` raises :class:`ShardRightsError` because the
metadata purpose is not ``train``.  This class makes the *positive* validation path
explicit instead of weakening that gate:

* it opens ``ShardReader(..., purpose="diagnostic")`` (never ``train``);
* it refuses a shard that is ``purpose=train``, ``train_eligible_under_policy=True``
  or carries a ``split="train"`` game row (defence in depth against a mislabelled
  corpus);
* it re-hashes every stored input/target row (:mod:`.integrity`) because the
  training loader does not;
* :meth:`ValidationReader.iter_forward_inputs` yields **only** forward tensors
  (no targets), so a caller that only forwards cannot see a label;
* :meth:`ValidationReader.evaluate` runs under ``torch.no_grad()`` / ``eval()``
  and asserts that no parameter gradient was produced;
* there is no optimizer argument anywhere in the class.

CE/top1/top3 here measure imitation of the validation corpus.  When the corpus
comes from weak self-play bots that only means "reproduces weak bots".
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator

import torch

from ..data.encode import MAX_EVENTS_DEFAULT
from ..data.loader import MultiShardBatch, ShardLoadError, ShardReader
from ..data.shard import ShardError, ShardMetadata, ShardRightsError, read_manifest, unpack_json
from ..models.policy import Policy
from .integrity import verify_shard_integrity
from .metrics import compute_metrics

__all__ = [
    "DecisionContext",
    "ValidationBatch",
    "ValidationError",
    "ValidationReader",
]


class ValidationError(RuntimeError):
    """The read-only validation contract was violated."""


@dataclass(frozen=True)
class DecisionContext:
    """Label-free public context of one stored decision (from the shard rows)."""

    decision_id: str
    phase: str
    kyoku: int
    bakaze_kind: int
    honba: int
    kyotaku: int
    tiles_left: int
    own_score: int
    own_riichi: str
    opponent_riichi: int
    is_oya: bool
    own_melds: int
    dora_count: int

    @property
    def is_south(self) -> bool:
        return self.bakaze_kind == 28

    @property
    def behind(self) -> bool:
        return self.own_score < 25_000

    def as_dict(self) -> dict:
        return {
            "decision_id": self.decision_id, "phase": self.phase,
            "kyoku": self.kyoku, "bakaze_kind": self.bakaze_kind,
            "honba": self.honba, "kyotaku": self.kyotaku,
            "tiles_left": self.tiles_left, "own_score": self.own_score,
            "own_riichi": self.own_riichi,
            "opponent_riichi": self.opponent_riichi,
            "is_oya": self.is_oya, "own_melds": self.own_melds,
            "dora_count": self.dora_count,
        }


@dataclass(frozen=True)
class ValidationBatch:
    """Forward batch + read-only targets + per-decision public context."""

    batch: MultiShardBatch
    contexts: tuple[DecisionContext, ...]

    def __post_init__(self):
        if len(self.contexts) != len(self.batch.targets):
            raise ValidationError(
                f"contexts {len(self.contexts)} != decisions {len(self.batch.targets)}")

    @property
    def decisions(self) -> int:
        return len(self.batch.targets)

    @property
    def read_only_targets(self) -> tuple[MappingProxyType, ...]:
        return tuple(MappingProxyType(dict(target)) for target in self.batch.targets)


class ValidationReader:
    """Open a ``diagnostic``/``audit`` shard read-only and never optimize it."""

    def __init__(self, path: str | Path, *, expected_policy_hash: str | None = None,
                 batch_size: int = 8, max_events: int = MAX_EVENTS_DEFAULT,
                 max_positions: int | None = None,
                 verify_integrity: bool = True,
                 verify_state_snapshots: bool = False,
                 allowed_source_kinds: tuple[str, ...] | None = None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) \
                or batch_size <= 0:
            raise ValidationError("batch_size must be a positive int")
        if max_positions is not None and max_positions <= 0:
            raise ValidationError("max_positions must be positive or None")
        self.path = Path(path)
        self._batch_size = batch_size
        self._max_events = max_events
        self._max_positions = max_positions
        self._verify_state_snapshots = bool(verify_state_snapshots)
        try:
            self._reader = ShardReader(
                self.path, purpose="diagnostic",
                expected_policy_hash=expected_policy_hash)
        except (ShardRightsError, ShardLoadError, ShardError) as exc:
            raise ValidationError(f"cannot open validation shard: {exc}") from exc
        self.metadata: ShardMetadata = self._reader.metadata
        if self.metadata.purpose not in ("diagnostic", "audit"):
            self._reader.close()
            raise ValidationError(
                f"validation reader refuses purpose={self.metadata.purpose!r}")
        if self.metadata.train_eligible_under_policy:
            self._reader.close()
            raise ValidationError(
                "validation reader refuses a train-eligible shard")
        if allowed_source_kinds is not None \
                and self.metadata.source_kind not in allowed_source_kinds:
            self._reader.close()
            raise ValidationError(
                f"source_kind {self.metadata.source_kind!r} not in "
                f"{sorted(allowed_source_kinds)}")
        self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        self._game_ids: tuple[str, ...] = ()
        self._split_audit()
        self.integrity = (verify_shard_integrity(self.path, metadata=self.metadata,
                                                strict=True)
                          if verify_integrity else None)

    # ------------------------------------------------------------------ gates
    def _split_audit(self) -> None:
        game_ids = []
        for game in self._reader.games():
            if game["split"] == "train":
                raise ValidationError(
                    f"validation shard contains train-split game {game['game_id']}")
            game_ids.append(game["game_id"])
        if not game_ids:
            raise ValidationError("validation shard contains no games")
        self._game_ids = tuple(game_ids)

    @property
    def game_ids(self) -> tuple[str, ...]:
        return self._game_ids

    @property
    def manifest(self) -> dict:
        return read_manifest(self.path)

    def describe(self) -> dict:
        return {
            "shard": str(self.path),
            "purpose": self.metadata.purpose,
            "source_id": self.metadata.source_id,
            "source_kind": self.metadata.source_kind,
            "ruleset_id": self.metadata.ruleset_id,
            "ruleset_deviations": list(self.metadata.ruleset_deviations),
            "research_policy": self.metadata.research_policy,
            "research_policy_hash": self.metadata.research_policy_hash,
            "train_allowed": self.metadata.train_allowed,
            "train_eligible_under_policy": self.metadata.train_eligible_under_policy,
            "games": len(self._game_ids),
            "integrity_ok": None if self.integrity is None else self.integrity["ok"],
        }

    # ------------------------------------------------------------- iteration
    def _contexts(self, batch: MultiShardBatch) -> tuple[DecisionContext, ...]:
        decision_ids = [f"{game_id}:{pov}:{index}" for game_id, pov, index in
                        zip(batch.game_ids, batch.povs, batch.decision_indices)]
        rows: dict[str, dict] = {}
        chunk = 400
        for start in range(0, len(decision_ids), chunk):
            part = decision_ids[start:start + chunk]
            placeholders = ",".join("?" for _ in part)
            query = (f"SELECT decision_id, phase, tile_state, players_public, "
                     f"global_public FROM inputs WHERE decision_id IN ({placeholders})")
            for row in self._conn.execute(query, part):
                rows[row["decision_id"]] = {
                    "phase": row["phase"],
                    "tile_state": unpack_json(row["tile_state"]),
                    "players_public": unpack_json(row["players_public"]),
                    "global_public": unpack_json(row["global_public"]),
                }
        contexts = []
        for decision_id in decision_ids:
            row = rows.get(decision_id)
            if row is None:
                raise ValidationError(f"stored decision {decision_id} disappeared")
            players = row["players_public"]
            public = row["global_public"]
            own = players[0]
            tile_state = row["tile_state"]
            contexts.append(DecisionContext(
                decision_id=decision_id,
                phase=str(row["phase"]),
                kyoku=int(public["kyoku"]),
                bakaze_kind=int(public["bakaze"]["kind"]),
                honba=int(public["honba"]),
                kyotaku=int(public["kyotaku"]),
                tiles_left=int(public["tiles_left"]),
                own_score=int(own["score"]),
                own_riichi=str(own["riichi"]),
                opponent_riichi=sum(1 for player in players[1:]
                                    if str(player["riichi"]) != "none"),
                is_oya=bool(own["is_oya"]),
                own_melds=len(own["melds"]),
                dora_count=len(tile_state["dora_indicators"]),
            ))
        return tuple(contexts)

    def iter_batches(self, *, limit: int | None = None) -> Iterator[ValidationBatch]:
        """Yield read-only validation batches (decisions, not games)."""
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValidationError("limit must be a positive int or None")
        seen = 0
        iterator = self._reader.iter_multi_decision_batches(
            batch_size=self._batch_size, max_events=self._max_events,
            max_positions=self._max_positions,
            verify_state_snapshots=self._verify_state_snapshots)
        for batch in iterator:
            yield ValidationBatch(batch=batch, contexts=self._contexts(batch))
            seen += len(batch.targets)
            if limit is not None and seen >= limit:
                return

    def iter_forward_inputs(self, *, limit: int | None = None) -> Iterator:
        """Yield only ``MultiDecisionBatch`` forward tensors (no target objects)."""
        for item in self.iter_batches(limit=limit):
            yield item.batch.batch

    # ------------------------------------------------------------ evaluation
    def evaluate(self, model: Policy, *, limit: int | None = None,
                 prefix_parity: bool = False, parity_max_decisions: int = 64) -> dict:
        """Forward-only CE/top1/top3; asserts that no optimizer step happened."""
        if not isinstance(model, Policy):
            raise ValidationError("model must be a Policy")
        was_training = model.training
        grads_before = [parameter.grad is not None for parameter in model.parameters()]
        model.eval()
        logprobs = []
        legal_masks = []
        targets = []
        contexts = []
        cutoffs = []
        lengths = []
        batches = 0
        try:
            with torch.no_grad():
                for item in self.iter_batches(limit=limit):
                    batch = item.batch
                    output = model.forward_multi(batch.batch)
                    legal_logprobs = output.logprobs[output.legal_mask]
                    if not bool(torch.isfinite(legal_logprobs).all()):
                        raise ValidationError(
                            f"model produced non-finite legal logprobs on batch "
                            f"{batches}")
                    logprobs.append(output.logprobs.detach().cpu())
                    legal_masks.append(batch.batch.legal_mask.detach().cpu())
                    targets.extend(dict(target) for target in batch.targets)
                    contexts.extend(item.contexts)
                    cutoffs.extend(int(value) for value in
                                   batch.batch.decision_cutoff_idx.tolist())
                    lengths.extend(int(value) for value in batch.lengths)
                    batches += 1
        finally:
            model.train(was_training)
        if not logprobs:
            raise ValidationError("validation shard produced no batches")
        if any(parameter.grad is not None and not before
               for parameter, before in zip(model.parameters(), grads_before)):
            raise ValidationError(
                "forward-only evaluation produced a parameter gradient")
        result = compute_metrics(
            logprobs=torch.cat(logprobs, dim=0),
            legal_mask=torch.cat(legal_masks, dim=0),
            targets=targets, contexts=contexts, cutoffs=cutoffs, lengths=lengths,
            max_events=self._max_events)
        result["batches"] = batches
        result["shard"] = str(self.path)
        result["reader"] = {"purpose": self.metadata.purpose,
                            "source_kind": self.metadata.source_kind,
                            "ruleset_id": self.metadata.ruleset_id,
                            "games": len(self._game_ids),
                            "read_only": True,
                            "optimizer_steps": 0}
        result["warning"] = (
            "offline imitation diagnostics only; CE/top1 are not strength, not a "
            "dan/rank and not evidence about Tenhou play")
        if prefix_parity:
            from .prefix_parity import prefix_multi_parity

            result["prefix_multi_parity"] = prefix_multi_parity(
                self, model, max_decisions=parity_max_decisions)
        return result

    # ------------------------------------------------------------- refusals
    def optimizer_update_forbidden(self, *args, **kwargs):
        """Explicit negative control: this reader cannot step an optimizer."""
        raise ValidationError(
            "ValidationReader is read-only: it has no optimizer and never updates "
            "model weights; use the training loader on a purpose=train shard")

    def close(self) -> None:
        if not self._closed:
            self._conn.close()
            self._reader.close()
            self._closed = True

    def __enter__(self) -> "ValidationReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
