"""VNext BC data adapter (PR-08): shard decisions -> VNextModelBatch.

This module does **not** mutate the existing shard format.  It consumes the
already-authenticated ``ShardReader.iter_decisions()`` rows, rebuilds each
current-kyoku BOS stream, and groups decisions from the same
``(game, POV, kyoku)`` into one shared event row.  Targets remain a separate
tuple; no target field enters :class:`VNextModelBatch`.

The adapter is intentionally explicit about cost: it currently decodes each
decision's visible prefix through the existing shard reader.  A pre-encoded
memmap is a later optimization only if profiling shows this is the bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .contracts import VNextModelBatch, VNextTargets
from .observable import VNextBatchBuilder, VNextEventStream, encode_stream


class VNextDataError(ValueError):
    pass


def normalise_rank(rank, *, rank_base: int) -> int:
    """Map a shard rank label to the Critic class axis (0..3)."""
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise VNextDataError(f"rank_final must be int, got {type(rank).__name__}")
    if rank_base == 0:
        if 0 <= rank <= 3:
            return rank
    elif rank_base == 1:
        if 1 <= rank <= 4:
            return rank - 1
    raise VNextDataError(f"rank_final={rank} is invalid for rank_base={rank_base}")


@dataclass(frozen=True)
class VNextBCBatch:
    batch: VNextModelBatch
    targets: tuple[dict, ...]
    game_ids: tuple[str, ...]
    povs: tuple[int, ...]
    decision_indices: tuple[int, ...]
    kyoku_ids: tuple[int, ...]
    #: Target order is exactly the VNextModelBatch decision order.  This is an
    #: explicit invariant, not an assumption about the reader's stream order.
    decision_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.targets)
        for name in ("game_ids", "povs", "decision_indices", "kyoku_ids",
                     "decision_ids"):
            if len(getattr(self, name)) != n:
                raise VNextDataError(f"{name} length != targets length")
        if self.batch.n_decisions != n:
            raise VNextDataError("batch decisions != target rows")
        if len(set(self.decision_ids)) != n:
            raise VNextDataError("decision_ids must be unique in a batch")


class VNextBatchAdapter:
    """Group shard decisions into current-kyoku VNext batches."""

    def __init__(self, reader, *, batch_size: int = 8,
                 rank_base: int = 0,
                 max_prefix_tokens: int | None = None) -> None:
        if not hasattr(reader, "iter_decisions"):
            raise VNextDataError("reader must expose iter_decisions()")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) \
                or batch_size <= 0:
            raise VNextDataError("batch_size must be a positive int")
        if rank_base not in (0, 1):
            raise VNextDataError("rank_base must be 0 (labels 0..3) or 1 (1..4)")
        if max_prefix_tokens is not None:
            if isinstance(max_prefix_tokens, bool) \
                    or not isinstance(max_prefix_tokens, int) \
                    or max_prefix_tokens <= 0:
                raise VNextDataError("max_prefix_tokens must be a positive int or None")
        self.reader = reader
        self.batch_size = int(batch_size)
        self.rank_base = int(rank_base)
        self.max_prefix_tokens = max_prefix_tokens
        self.long_prefix_skipped = 0

    def _finish(self, streams: dict, records: dict) -> VNextBCBatch:
        if not records:
            raise VNextDataError("cannot finish an empty batch")
        builder = VNextBatchBuilder(list(streams.values()))
        batch = builder.to_batch()
        ordered_ids: list[str] = []
        targets: list[dict] = []
        game_ids: list[str] = []
        povs: list[int] = []
        indices: list[int] = []
        kyoku_ids: list[int] = []
        # Reorder target metadata into the model batch's decision order.
        for stream in streams.values():
            for index, _state in stream.decisions:
                key = index.decision_id
                record = records.get(key)
                if record is None:
                    raise VNextDataError(
                        f"missing target record for batched decision {key}")
                target, game_id, pov, decision_index, kyoku = record
                ordered_ids.append(key)
                target = dict(target)
                rank = target.get("rank_final")
                if rank is not None:
                    target["rank_final"] = normalise_rank(
                        rank, rank_base=self.rank_base)
                targets.append(target)
                game_ids.append(game_id)
                povs.append(pov)
                indices.append(decision_index)
                kyoku_ids.append(kyoku)
        if len(ordered_ids) != batch.n_decisions:
            raise VNextDataError("target/model decision count mismatch")
        return VNextBCBatch(
            batch=batch, targets=tuple(targets), game_ids=tuple(game_ids),
            povs=tuple(povs), decision_indices=tuple(indices),
            kyoku_ids=tuple(kyoku_ids), decision_ids=tuple(ordered_ids))

    def iter_batches(self) -> Iterator[VNextBCBatch]:
        streams: dict[tuple[str, int, int], VNextEventStream] = {}
        records: dict[str, tuple] = {}
        for decision, target, _meta, game_id, pov, decision_index in \
                self.reader.iter_decisions():
            current = [event for event in decision.events if int(event.kyoku_idx) >= 0]
            if not current:
                raise VNextDataError(
                    f"decision {game_id}/{pov}/{decision_index} has no current kyoku")
            kyoku = int(current[-1].kyoku_idx)
            if self.max_prefix_tokens is not None and \
                    len(current) + 1 > self.max_prefix_tokens:
                # The positional contract is a hard bound: skip this decision
                # before creating or extending an event stream.
                self.long_prefix_skipped += 1
                continue
            key = (str(game_id), int(pov), kyoku)
            stream = streams.get(key)
            if stream is None:
                stream = VNextEventStream(
                    match_id=str(game_id), kyoku_id=kyoku, pov=int(pov))
                streams[key] = stream
            decision_id = f"{game_id}:{pov}:{decision_index}"
            stream.add_decision(
                decision, decision_id=decision_id,
                decision_order=int(decision_index))
            if decision_id in records:
                raise VNextDataError(f"duplicate decision id {decision_id}")
            records[decision_id] = (dict(target), str(game_id), int(pov),
                                    int(decision_index), kyoku)
            if len(records) >= self.batch_size:
                yield self._finish(streams, records)
                streams, records = {}, {}
        if records:
            yield self._finish(streams, records)


def targets_from_dicts(targets: tuple[dict, ...], *, device=None) -> VNextTargets:
    """Build the trainer-only tensor object from shard target dicts."""
    import torch

    n = len(targets)
    if n == 0:
        raise VNextDataError("need at least one target")
    action = torch.zeros(n, dtype=torch.long)
    rank = torch.zeros(n, dtype=torch.long)
    supervised = torch.zeros(n, dtype=torch.bool)
    weight = torch.zeros(n, dtype=torch.float32)
    forced = torch.zeros(n, dtype=torch.bool)
    for index, target in enumerate(targets):
        value = target.get("expert_action")
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 46:
            action[index] = value
        final = target.get("rank_final")
        if isinstance(final, int) and not isinstance(final, bool) and 0 <= final < 4:
            rank[index] = final
        supervised[index] = bool(target.get("supervision_valid", False))
        raw_weight = target.get("loss_weight", 1.0)
        weight[index] = float(raw_weight) if isinstance(
            raw_weight, (int, float)) and not isinstance(raw_weight, bool) else 0.0
        forced[index] = bool(target.get("forced", False))
    if device is not None:
        action, rank, supervised, weight, forced = (
            t.to(device) for t in (action, rank, supervised, weight, forced))
    return VNextTargets(action=action, final_rank=rank, supervise_mask=supervised,
                        weight=weight, forced=forced)


__all__ = ["VNextBCBatch", "VNextBatchAdapter", "VNextDataError",
           "normalise_rank", "targets_from_dicts"]
