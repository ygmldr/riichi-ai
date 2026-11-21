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


@dataclass(frozen=True)
class VNextBCBatch:
    batch: VNextModelBatch
    targets: tuple[dict, ...]
    game_ids: tuple[str, ...]
    povs: tuple[int, ...]
    decision_indices: tuple[int, ...]
    kyoku_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.targets)
        for name in ("game_ids", "povs", "decision_indices", "kyoku_ids"):
            if len(getattr(self, name)) != n:
                raise VNextDataError(f"{name} length != targets length")
        if self.batch.n_decisions != n:
            raise VNextDataError("batch decisions != target rows")


class VNextBatchAdapter:
    """Group shard decisions into current-kyoku VNext batches."""

    def __init__(self, reader, *, batch_size: int = 8) -> None:
        if not hasattr(reader, "iter_decisions"):
            raise VNextDataError("reader must expose iter_decisions()")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) \
                or batch_size <= 0:
            raise VNextDataError("batch_size must be a positive int")
        self.reader = reader
        self.batch_size = int(batch_size)

    def _finish(self, streams: dict, targets: list, game_ids: list,
                povs: list, indices: list, kyoku_ids: list) -> VNextBCBatch:
        if not targets:
            raise VNextDataError("cannot finish an empty batch")
        builder = VNextBatchBuilder(list(streams.values()))
        return VNextBCBatch(
            batch=builder.to_batch(), targets=tuple(targets),
            game_ids=tuple(game_ids), povs=tuple(povs),
            decision_indices=tuple(indices), kyoku_ids=tuple(kyoku_ids))

    def iter_batches(self) -> Iterator[VNextBCBatch]:
        streams: dict[tuple[str, int, int], VNextEventStream] = {}
        targets: list[dict] = []
        game_ids: list[str] = []
        povs: list[int] = []
        indices: list[int] = []
        kyoku_ids: list[int] = []
        for decision, target, _meta, game_id, pov, decision_index in \
                self.reader.iter_decisions():
            current = [event for event in decision.events if int(event.kyoku_idx) >= 0]
            if not current:
                raise VNextDataError(
                    f"decision {game_id}/{pov}/{decision_index} has no current kyoku")
            kyoku = int(current[-1].kyoku_idx)
            key = (str(game_id), int(pov), kyoku)
            stream = streams.get(key)
            if stream is None:
                stream = VNextEventStream(
                    match_id=str(game_id), kyoku_id=kyoku, pov=int(pov))
                streams[key] = stream
            stream.add_decision(
                decision, decision_id=f"{game_id}:{pov}:{decision_index}",
                decision_order=int(decision_index))
            targets.append(dict(target))
            game_ids.append(str(game_id))
            povs.append(int(pov))
            indices.append(int(decision_index))
            kyoku_ids.append(kyoku)
            if len(targets) >= self.batch_size:
                yield self._finish(streams, targets, game_ids, povs, indices,
                                   kyoku_ids)
                streams, targets, game_ids, povs, indices, kyoku_ids = (
                    {}, [], [], [], [], [])
        if targets:
            yield self._finish(streams, targets, game_ids, povs, indices, kyoku_ids)


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
           "targets_from_dicts"]
