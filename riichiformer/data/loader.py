#!/usr/bin/env python3
"""PR-12 streaming shard reader: ModelBatch rows + separate targets.

The loader imports no target/label class.  Targets stay plain dictionaries from the
``targets`` table; only the forward side is converted to PR-13 tensors.
"""

from __future__ import annotations

import json
import sqlite3
import string
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

from ..models.batch import ModelBatch, ModelBatchError, MultiDecisionBatch
from ..models.state_encoder import state_features_from_snapshot
from ..schemas.action_v3 import ActionPhase
from ..schemas.decision_v3 import (
    DecisionInput,
    GlobalPublicState,
    MeldKind,
    PlayerPublicState,
    PublicMeld,
    RiichiState,
    RulesetDeviation,
    RulesetId,
    SourceKind,
    VisibleTileState,
)
from ..schemas.event_v3 import (
    AnkanEvent,
    ChiEvent,
    DahaiEvent,
    DaiminkanEvent,
    DoraEvent,
    EndGameEvent,
    EndKyokuEvent,
    HoraEvent,
    KakanEvent,
    PonEvent,
    ReachAcceptedEvent,
    ReachEvent,
    RyukyokuEvent,
    StartGameEvent,
    StartKyokuEvent,
    TileRef,
    TsumoEvent,
    event_from_dict,
)
from .decision import DecisionBuildConfig, _SnapshotTracker
from .encode import (
    MAX_EVENTS_DEFAULT,
    EncodedBatch,
    EncoderError,
    StreamMeta,
    _state_snapshot,
    encode_batch,
    encode_decision,
    encode_multi_decision_batch,
)
from .multi_decision import (
    DecisionMarker,
    MultiDecisionError,
    MultiDecisionStream,
    check_prefix,
)
from .shard import (
    ACTION_HASH,
    FROZEN_SCHEMA,
    NORMALIZER_HASH,
    SCHEMA_HASH,
    ShardError,
    ShardMetadata,
    ShardRightsError,
    file_sha256,
    read_manifest,
    unpack_json,
)

PURPOSES = ("train", "audit", "diagnostic")


class ShardLoadError(ShardError):
    """The shard cannot be read under the requested purpose/hash/source."""


_HEX = frozenset(string.hexdigits)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX for char in value)


def _tile(value: dict) -> TileRef:
    return TileRef(int(value["kind"]), value.get("aka"))


def _meld(value: dict) -> PublicMeld:
    return PublicMeld(
        meld_kind=MeldKind(value["meld_kind"]),
        tiles=tuple(_tile(tile) for tile in value["tiles"]),
        from_rel=int(value["from_rel"]),
    )


def _tile_state(value: dict) -> VisibleTileState:
    return VisibleTileState(
        own_counts=tuple(int(v) for v in value["own_counts"]),
        own_aka=tuple(int(v) for v in value["own_aka"]),
        public_counts=tuple(int(v) for v in value["public_counts"]),
        public_aka=tuple(int(v) for v in value["public_aka"]),
        public_counts_by_zone=tuple(tuple(int(v) for v in zone)
                                    for zone in value["public_counts_by_zone"]),
        dora_indicators=tuple(_tile(tile) for tile in value["dora_indicators"]),
        ruleset_id=RulesetId(value["ruleset_id"]),
    )


def _player(value: dict) -> PlayerPublicState:
    return PlayerPublicState(
        seat_rel=int(value["seat_rel"]),
        score=int(value["score"]),
        wind=_tile(value["wind"]),
        is_oya=bool(value["is_oya"]),
        riichi=RiichiState(value["riichi"]),
        river_count=int(value["river_count"]),
        melds=tuple(_meld(meld) for meld in value["melds"]),
    )


def _global(value: dict) -> GlobalPublicState:
    return GlobalPublicState(
        bakaze=_tile(value["bakaze"]),
        kyoku=int(value["kyoku"]),
        honba=int(value["honba"]),
        kyotaku=int(value["kyotaku"]),
        tiles_left=int(value["tiles_left"]),
        self_wind=_tile(value["self_wind"]),
        oya_rel=int(value["oya_rel"]),
        ruleset_id=RulesetId(value["ruleset_id"]),
        ruleset_deviation=tuple(RulesetDeviation(code)
                                for code in value["ruleset_deviation"]),
        source_kind=SourceKind(value["source_kind"]),
    )


def _decision_input(stream: tuple[dict, ...], payload: dict, *,
                    stream_events: tuple | None = None) -> DecisionInput:
    """Build one decision from the stored POV stream prefix.

    ``stream_events`` is an optional already-decoded full POV event sequence.  The
    multi-decision loader passes it so a long stream is decoded once instead of
    once per decision (the legacy per-decision path keeps its exact behaviour).
    """
    before = int(payload["before_event_idx"])
    if before < 1 or before > len(stream):
        raise ShardLoadError(f"bad decision cutoff {before} / stream {len(stream)}")
    if stream_events is not None:
        if len(stream_events) != len(stream):
            raise ShardLoadError("decoded POV stream length mismatch")
        events = stream_events[:before]
    else:
        events = tuple(event_from_dict(event) for event in stream[:before])
    if events[-1].seq_idx != before - 1:
        raise ShardLoadError("stored POV stream is not contiguous at the cutoff")
    return DecisionInput(
        events=events,
        tile_state=_tile_state(payload["tile_state"]),
        players_public=tuple(_player(player) for player in payload["players_public"]),
        global_public=_global(payload["global_public"]),
        legal_mask=tuple(bool(v) for v in payload["legal_mask"]),
        phase=ActionPhase(payload["phase"]),
        before_event_idx=before,
        schema_version=payload.get("schema_version", FROZEN_SCHEMA["schema_version"]),
    )


def _decision_payload(row) -> dict:
    """Unpack one ``inputs`` row exactly as :meth:`ShardReader.iter_decisions` does."""
    return {
        "before_event_idx": row["before_event_idx"],
        "kyoku_idx": row["kyoku_idx"],
        "phase": row["phase"],
        "legal_mask": unpack_json(row["legal_mask"]),
        "tile_state": unpack_json(row["tile_state"]),
        "players_public": unpack_json(row["players_public"]),
        "global_public": unpack_json(row["global_public"]),
    }


@dataclass(frozen=True)
class ShardBatch:
    """One deterministic batch: forward ModelBatch + plain target dicts."""

    batch: ModelBatch
    targets: tuple[dict, ...]
    metas: tuple[StreamMeta, ...]
    lengths: tuple[int, ...]
    game_ids: tuple[str, ...]
    povs: tuple[int, ...]
    decision_indices: tuple[int, ...]

    def __post_init__(self):
        n = len(self.targets)
        for name in ("metas", "lengths", "game_ids", "povs", "decision_indices"):
            if len(getattr(self, name)) != n:
                raise ShardLoadError(f"{name} length != targets length")
        if self.batch.n_decisions != n:
            raise ShardLoadError("ModelBatch decisions != target rows")


@dataclass(frozen=True)
class MultiShardBatch:
    """One deterministic PR-38 batch: shared streams + per-decision targets.

    ``batch`` is a :class:`MultiDecisionBatch`: one row per ``(game, POV)`` stream
    and one transformer marker per decision.  Every named tuple below is
    **per decision** (``N = batch.n_decisions``), matching :class:`ShardBatch`;
    ``decision_stream_index[i]`` is the batch row (stream) that decision ``i``
    belongs to.  Row-level views are exposed as properties
    (:attr:`stream_game_ids`, :attr:`stream_povs`, :attr:`stream_lengths`,
    :attr:`stream_metas`).

    ``lengths[i]`` is the number of **event** tokens of decision ``i``'s stream
    (not its cutoff), i.e. ``batch.lengths[decision_stream_index[i]]``.
    :attr:`tokens` is the exact joined transformer cost: event tokens once per
    stream plus one marker per decision.
    """

    batch: MultiDecisionBatch
    targets: tuple[dict, ...]
    metas: tuple[StreamMeta, ...]
    lengths: tuple[int, ...]
    game_ids: tuple[str, ...]
    povs: tuple[int, ...]
    decision_indices: tuple[int, ...]
    #: Batch row of each decision (row-major, matching the encoder's order).
    decision_stream_index: tuple[int, ...]

    def __post_init__(self):
        n = len(self.targets)
        if self.batch.n_decisions != n:
            raise ShardLoadError(
                f"MultiDecisionBatch decisions {self.batch.n_decisions} != target "
                f"rows {n}")
        for name in ("metas", "lengths", "game_ids", "povs", "decision_indices",
                     "decision_stream_index"):
            if len(getattr(self, name)) != n:
                raise ShardLoadError(f"{name} length != targets length (N={n})")
        rows = self.batch.batch_size
        if n <= 0 or rows <= 0:
            raise ShardLoadError("multi-decision batch must contain decisions/rows")
        expected_rows = tuple(int(value) for value in self.batch.decision_batch_index)
        if tuple(int(value) for value in self.decision_stream_index) != expected_rows:
            raise ShardLoadError(
                "decision_stream_index is not the MultiDecisionBatch decision order")
        stream_lengths = tuple(int(value) for value in self.batch.lengths)
        for index in range(n):
            row = int(self.decision_stream_index[index])
            if not 0 <= row < rows:
                raise ShardLoadError(f"decision[{index}] stream row {row} out of range")
            if int(self.lengths[index]) != stream_lengths[row]:
                raise ShardLoadError(
                    f"decision[{index}] length {self.lengths[index]} != stream "
                    f"{row} length {stream_lengths[row]}")
            meta = self.metas[index]
            if not isinstance(meta, StreamMeta):
                raise ShardLoadError(f"metas[{index}] must be StreamMeta")
            if meta.pov != int(self.povs[index]):
                raise ShardLoadError(
                    f"decision[{index}] meta.pov {meta.pov} != pov {self.povs[index]}")
        # Every row's decisions are contiguous in marker order, have one fixed
        # (game_id, pov) identity, and keep strictly increasing stream-local
        # decision indices (a POV is never reordered or silently deduplicated).
        for row in range(rows):
            positions = [index for index in range(n)
                         if int(self.decision_stream_index[index]) == row]
            if not positions:
                raise ShardLoadError(f"stream row {row} has no decision")
            identities = {(self.game_ids[index], int(self.povs[index]))
                          for index in positions}
            if len(identities) != 1:
                raise ShardLoadError(
                    f"stream row {row} mixes game/POV identities {sorted(identities)}")
            indices = [int(self.decision_indices[index]) for index in positions]
            if any(left >= right for left, right in zip(indices, indices[1:])):
                raise ShardLoadError(
                    f"stream row {row} decision indices are not strictly increasing: "
                    f"{indices}")

    # ------------------------------------------------------------- row views
    def _row_first(self) -> tuple[int, ...]:
        first: dict[int, int] = {}
        for index, row in enumerate(self.decision_stream_index):
            first.setdefault(int(row), index)
        return tuple(first[row] for row in range(self.batch.batch_size))

    @property
    def stream_game_ids(self) -> tuple[str, ...]:
        return tuple(self.game_ids[index] for index in self._row_first())

    @property
    def stream_povs(self) -> tuple[int, ...]:
        return tuple(self.povs[index] for index in self._row_first())

    @property
    def stream_lengths(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.batch.lengths)

    @property
    def stream_metas(self) -> tuple[StreamMeta, ...]:
        return tuple(self.metas[index] for index in self._row_first())

    # ------------------------------------------------------------- accounting
    @property
    def event_tokens(self) -> int:
        return int(self.batch.padding_mask.sum())

    @property
    def marker_tokens(self) -> int:
        return int(self.batch.marker_padding_mask.sum())

    @property
    def tokens(self) -> int:
        """Joined transformer tokens: shared events + one marker per decision."""
        return self.event_tokens + self.marker_tokens


@dataclass(frozen=True)
class _MultiStreamUnit:
    """One row ready to join a multi-decision batch (one POV chunk)."""

    game_id: str
    pov: int
    meta: StreamMeta
    events: tuple
    stream: MultiDecisionStream
    targets: tuple[dict, ...]
    decision_indices: tuple[int, ...]


class _ObservablePrefixState:
    """O(1)-per-event public state used for the cheap prefix-consistency check.

    It tracks only aggregate facts that are directly observable from the projected
    stream and independently stored in the decision snapshot: wall count, table
    state, river counts, riichi flags, meld counts, scores and dora indicators.
    Unlike the full :class:`~riichiformer.data.decision._SnapshotTracker` replay it
    never rebuilds the tile ledger, so it costs a few microseconds per event rather
    than tens; ``verify_state_snapshots=True`` still runs the exact replay.
    """

    def __init__(self) -> None:
        self.started = False
        self.kyoku = 1
        self.bakaze = TileRef(27)
        self.honba = 0
        self.kyotaku = 0
        self.oya_rel = 0
        self.scores = [0, 0, 0, 0]
        self.rivers = [0, 0, 0, 0]
        self.riichi = [RiichiState.NONE] * 4
        self.melds = [0, 0, 0, 0]
        self.dora_indicators: list[TileRef] = []
        self.tsumo_count = 0

    def apply(self, event) -> None:
        if isinstance(event, StartKyokuEvent):
            self.started = True
            self.kyoku = event.kyoku
            self.bakaze = event.bakaze
            self.honba = event.honba
            self.kyotaku = event.kyotaku
            self.oya_rel = event.oya_rel
            self.scores = list(event.scores_rel)
            self.rivers = [0, 0, 0, 0]
            self.riichi = [RiichiState.NONE] * 4
            self.melds = [0, 0, 0, 0]
            self.dora_indicators = [event.dora_marker]
            self.tsumo_count = 0
            return
        if not self.started:
            return
        if isinstance(event, TsumoEvent):
            self.tsumo_count += 1
        elif isinstance(event, DahaiEvent):
            self.rivers[event.actor_rel] += 1
        elif isinstance(event, (ChiEvent, PonEvent, DaiminkanEvent)):
            self.rivers[event.target_rel] -= 1
            self.melds[event.actor_rel] += 1
        elif isinstance(event, KakanEvent):
            return  # a pon becomes a kakan; the meld count is unchanged
        elif isinstance(event, AnkanEvent):
            self.melds[event.actor_rel] += 1
        elif isinstance(event, ReachEvent):
            self.riichi[event.actor_rel] = RiichiState.DECLARED
        elif isinstance(event, ReachAcceptedEvent):
            self.riichi[event.actor_rel] = RiichiState.ACCEPTED
            self.scores[event.actor_rel] -= 1000
            self.kyotaku += 1
        elif isinstance(event, DoraEvent):
            self.dora_indicators.append(event.dora_marker)
        elif isinstance(event, HoraEvent):
            for seat, delta in enumerate(event.deltas):
                self.scores[seat] += delta
            self.kyotaku = 0
        elif isinstance(event, RyukyokuEvent):
            for seat, delta in enumerate(event.deltas):
                self.scores[seat] += delta
        elif isinstance(event, (EndKyokuEvent, EndGameEvent, StartGameEvent)):
            return

    def check(self, snapshot, *, game_id: str, pov: int,
              index: int) -> None:
        def fail(message: str) -> None:
            raise ShardLoadError(
                f"game {game_id} POV {pov} decision[{index}] snapshot does not "
                f"match the full stream prefix: {message}")

        public = snapshot.global_public
        if min(self.rivers) < 0:
            fail(f"negative river count in stream {self.rivers}")
        expected_left = max(0, min(70, 70 - self.tsumo_count))
        if public.tiles_left != expected_left:
            fail(f"tiles_left {public.tiles_left} != {expected_left}")
        if public.kyoku != self.kyoku:
            fail(f"kyoku {public.kyoku} != {self.kyoku}")
        if public.bakaze != self.bakaze:
            fail(f"bakaze {public.bakaze} != {self.bakaze}")
        if public.honba != self.honba:
            fail(f"honba {public.honba} != {self.honba}")
        if public.kyotaku != self.kyotaku:
            fail(f"kyotaku {public.kyotaku} != {self.kyotaku}")
        if public.oya_rel != self.oya_rel:
            fail(f"oya_rel {public.oya_rel} != {self.oya_rel}")
        if snapshot.tile_state.dora_indicators != tuple(self.dora_indicators):
            fail("dora_indicators differ")
        if len(snapshot.players_public) != 4:
            fail("players_public must have four entries")
        for seat, player in enumerate(snapshot.players_public):
            if player.score != self.scores[seat]:
                fail(f"player[{seat}].score {player.score} != {self.scores[seat]}")
            if player.riichi != self.riichi[seat]:
                fail(f"player[{seat}].riichi {player.riichi} != {self.riichi[seat]}")
            if player.river_count != self.rivers[seat]:
                fail(f"player[{seat}].river_count {player.river_count} != "
                     f"{self.rivers[seat]}")
            if len(player.melds) != self.melds[seat]:
                fail(f"player[{seat}].melds {len(player.melds)} != "
                     f"{self.melds[seat]}")
            expected_wind = TileRef(
                27 + ((self.bakaze.kind - 27 + seat - self.oya_rel) % 4))
            if player.wind != expected_wind:
                fail(f"player[{seat}].wind {player.wind} != {expected_wind}")
            if player.is_oya != (seat == self.oya_rel):
                fail(f"player[{seat}].is_oya is inconsistent with oya_rel")


class ShardReader:
    """Read a finalized shard; validate metadata before yielding anything."""

    def __init__(self, path: str | Path, *, purpose: str = "train",
                 expected_hashes: dict | None = None,
                 expected_source_id: str | None = None,
                 expected_component_sha256: dict[str, str] | None = None,
                 expected_policy_hash: str | None = None,
                 allowed_local_sources: Iterable[str] | None = None,
                 verify_file_hash: bool = True):
        if purpose not in PURPOSES:
            raise ShardLoadError(f"purpose must be one of {PURPOSES}")
        self.path = Path(path)
        self.purpose = purpose
        if expected_policy_hash is not None:
            if not _is_sha256(expected_policy_hash):
                raise ShardLoadError(
                    "expected_policy_hash must be a 64-char hex sha256")
            expected_policy_hash = expected_policy_hash.lower()
        self._expected_policy_hash = expected_policy_hash
        if allowed_local_sources is not None:
            if isinstance(allowed_local_sources, str):
                raise ShardLoadError(
                    "allowed_local_sources must be a collection of source ids")
            try:
                sources = frozenset(allowed_local_sources)
            except TypeError as exc:
                raise ShardLoadError(
                    "allowed_local_sources must be an iterable of source ids") from exc
            if any(not isinstance(item, str) or not item for item in sources):
                raise ShardLoadError(
                    "allowed_local_sources entries must be non-empty strings")
            self._allowed_local_sources = sources
        else:
            self._allowed_local_sources = None
        if not self.path.exists():
            raise ShardLoadError(f"shard not found: {self.path}")
        self.manifest = read_manifest(self.path)
        self.metadata = ShardMetadata.from_dict(self.manifest["metadata"])
        if verify_file_hash:
            # Chunked, never path.read_bytes(): a 100k-game shard can be ~90 GB.
            digest = file_sha256(self.path)
            if digest != self.manifest.get("shard_sha256"):
                raise ShardLoadError("shard file hash does not match manifest")
        if expected_hashes is not None:
            required = {"schema_hash", "action_map_hash", "normalizer_hash"}
            optional = {"canonical_manifest_sha256", "canonical_index_sha256",
                        "split_manifest_sha256", "schema_version"}
            if not required <= set(expected_hashes):
                raise ShardLoadError(
                    "expected_hashes must contain schema_hash/action_map_hash/"
                    "normalizer_hash")
            if set(expected_hashes) - required - optional:
                raise ShardLoadError(
                    f"unsupported expected_hashes keys: "
                    f"{sorted(set(expected_hashes) - required - optional)}")
            for key, expected in expected_hashes.items():
                if expected != getattr(self.metadata, key):
                    raise ShardLoadError(f"{key} mismatch")
        if expected_source_id is not None and self.metadata.source_id != expected_source_id:
            raise ShardLoadError(
                f"source_id {self.metadata.source_id!r} != expected {expected_source_id!r}")
        if expected_component_sha256 is not None:
            if not isinstance(expected_component_sha256, dict):
                raise ShardLoadError("expected_component_sha256 must be a dict")
            for game_id, digest in expected_component_sha256.items():
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ShardLoadError(
                        f"expected component sha for {game_id!r} must be 64-char sha256")
        self._expected_component_sha256 = dict(expected_component_sha256 or {})
        if self.metadata.schema_hash != SCHEMA_HASH \
                or self.metadata.action_map_hash != ACTION_HASH \
                or self.metadata.normalizer_hash != NORMALIZER_HASH:
            raise ShardLoadError("shard schema/action/normalizer hash mismatch")
        if self._expected_policy_hash is not None:
            if self.metadata.research_policy != "operator_private_research":
                raise ShardRightsError(
                    "expected_policy_hash was provided but the shard is "
                    f"research_policy={self.metadata.research_policy!r}")
            if self.metadata.research_policy_hash != self._expected_policy_hash:
                raise ShardRightsError(
                    "research_policy_hash mismatch: shard records "
                    f"{self.metadata.research_policy_hash!r}, expected "
                    f"{self._expected_policy_hash!r}")
        if purpose == "train":
            if self.metadata.purpose != "train":
                raise ShardRightsError(
                    f"training loader refuses purpose={self.metadata.purpose!r} shard")
            if self.metadata.research_policy == "licensed_only":
                if not self.metadata.train_allowed \
                        or self.metadata.rights_status.upper() != "ALLOWED":
                    raise ShardRightsError(
                        "licensed training loader refuses blocked source shard")
            else:
                # P1 closure: a private shard is never authenticated by itself.
                if self._expected_policy_hash is None:
                    raise ShardRightsError(
                        "operator_private_research training shard requires an "
                        "explicit expected policy hash (--policy-config or "
                        "--expected-policy-hash); refusing an unauthenticated "
                        "private shard")
                if not self.metadata.train_eligible_under_policy:
                    raise ShardRightsError(
                        "private training shard is not marked "
                        "train_eligible_under_policy")
                if self._allowed_local_sources is not None:
                    uncovered = self._allowed_sources() - self._allowed_local_sources
                    if uncovered:
                        raise ShardRightsError(
                            "shard source(s) "
                            f"{sorted(uncovered)} are outside the expected policy "
                            "allowlist")
        self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._conn.close()
            self._closed = True

    def _stream_meta(self, pov_index: int) -> StreamMeta:
        return StreamMeta(
            pov=pov_index,
            projection_version=self.metadata.projection_version,
            schema_version=self.metadata.schema_version,
            schema_hash=self.metadata.schema_hash,
            ruleset_id=self.metadata.ruleset_id,
            source_kind=self.metadata.source_kind,
        )

    def _allowed_sources(self) -> set[str]:
        source = self.metadata.source_id
        if source.startswith("mixed:"):
            return {item for item in source[len("mixed:"):].split("+") if item}
        return {source}

    def _assert_train_game(self, game: dict) -> None:
        if game["split"] != "train":
            raise ShardRightsError(
                f"training loader refuses game {game['game_id']} from split "
                f"{game['split']!r}")
        if self.metadata.research_policy == "licensed_only":
            if not game["train_allowed"]:
                raise ShardRightsError(
                    f"licensed training loader refuses game {game['game_id']} with "
                    "per-game train_allowed=False")
            if str(game["rights_status"]).upper() != "ALLOWED":
                raise ShardRightsError(
                    f"licensed training loader refuses game {game['game_id']} with rights_status "
                    f"{game['rights_status']!r}")
        if game["source_id"] not in self._allowed_sources():
            raise ShardRightsError(
                f"training loader refuses game {game['game_id']} source "
                f"{game['source_id']!r} not covered by metadata source "
                f"{self.metadata.source_id!r}")

    def _validate_expected_component(self, game: dict) -> None:
        expected = self._expected_component_sha256.get(game["game_id"])
        if expected is not None and expected != game["component_sha256"]:
            raise ShardLoadError(
                f"component sha mismatch for {game['game_id']}: "
                f"{game['component_sha256']} != expected {expected}")

    def games(self) -> Iterator[dict]:
        # Stream game rows: a 100k-game shard is ~90 GB, so fetchall() would defeat
        # the chunked-hash fix and load every compressed POV stream into memory.
        rows = self._conn.execute("SELECT * FROM games ORDER BY game_id")
        for row in rows:
            game = {
                "game_id": row["game_id"], "component_id": row["component_id"],
                "component_sha256": row["component_sha256"], "split": row["split"],
                "source_id": row["source_id"],
                "train_allowed": bool(row["train_allowed"]),
                "rights_status": str(row["rights_status"]).upper(),
                "pov_streams": unpack_json(row["pov_streams"]),
                "kyoku_offsets": unpack_json(row["kyoku_offsets"]),
                "decision_count": row["decision_count"],
            }
            self._validate_expected_component(game)
            if self.purpose == "train":
                self._assert_train_game(game)
            yield game

    def _load_target(self, decision_id: str) -> dict:
        target = self._conn.execute(
            "SELECT * FROM targets WHERE decision_id=?",
            (decision_id,)).fetchone()
        if target is None:
            raise ShardLoadError(f"missing target {decision_id}")
        target_dict = {
            "expert_action": target["expert_action"],
            "supervision_valid": bool(target["supervision_valid"]),
            "censor_reason": target["censor_reason"],
            "rank_final": target["rank_final"],
        }
        if bool(target["supervision_valid"]) and target_dict["censor_reason"] is not None:
            raise ShardLoadError("supervised target has censor reason")
        if not bool(target["supervision_valid"]) and target_dict["censor_reason"] is None:
            raise ShardLoadError("censored target has no reason")
        return target_dict

    def iter_decisions(self):
        """Yield ``(input, target_dict, meta, game_id, pov, decision_index)``."""
        for game in self.games():
            raw_streams = tuple(tuple(stream) for stream in game["pov_streams"])
            # Decode each POV stream once per game.  The previous path decoded
            # the full prefix again for every decision (O(décisions × events)),
            # which dominates VNext/BC data loading on long hanchan streams.
            streams = tuple(
                tuple(event_from_dict(event) for event in stream)
                for stream in raw_streams)
            inputs = self._conn.execute(
                "SELECT * FROM inputs WHERE game_id=? ORDER BY pov,decision_index",
                (game["game_id"],)).fetchall()
            for row in inputs:
                target_dict = self._load_target(row["decision_id"])
                decision = _decision_input(
                    raw_streams[row["pov"]], _decision_payload(row),
                    stream_events=streams[row["pov"]])
                yield (decision, target_dict, self._stream_meta(row["pov"]),
                       game["game_id"], int(row["pov"]), int(row["decision_index"]))

    def iter_batches(self, batch_size: int = 8,
                     bucket_by_length: bool = True) -> Iterator[ShardBatch]:
        """Deterministic batches; sorts each buffer by sequence length (dynamic padding)."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ShardLoadError("batch_size must be a positive int")
        buffer = []
        for item in self.iter_decisions():
            buffer.append(item)
            if len(buffer) >= batch_size:
                yield self._make_batch(buffer, bucket_by_length)
                buffer = []
        if buffer:
            yield self._make_batch(buffer, bucket_by_length)

    # ------------------------------------------------- PR-38 multi-decision
    def _multi_config(self, snapshot) -> DecisionBuildConfig:
        """Recreate the snapshot config recorded in one stored decision."""
        state = snapshot.tile_state
        public = snapshot.global_public
        if state.ruleset_id.value != self.metadata.ruleset_id:
            raise ShardLoadError(
                f"decision ruleset {state.ruleset_id.value!r} != shard metadata "
                f"{self.metadata.ruleset_id!r}")
        if public.ruleset_id != state.ruleset_id:
            raise ShardLoadError("decision tile_state/global_public ruleset mismatch")
        if public.source_kind.value != self.metadata.source_kind:
            raise ShardLoadError(
                f"decision source_kind {public.source_kind.value!r} != shard "
                f"metadata {self.metadata.source_kind!r}")
        try:
            return DecisionBuildConfig(
                ruleset_id=state.ruleset_id,
                source_kind=public.source_kind,
                ruleset_deviation=public.ruleset_deviation,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a shard error
            raise ShardLoadError(f"decision ruleset config invalid: {exc}") from exc

    def _check_multi_prefixes(self, game_id: str, pov: int, events, cutoffs,
                              snapshots, payloads, *,
                              verify_state_snapshots: bool) -> None:
        """Fail closed unless each decision is the exact full-stream prefix.

        Always enforced:

        * ``before_event_idx`` equals the visible prefix length and the decision
          snapshot was built from the exact ``events[:cutoff]`` slice (plus the
          stored ``kyoku_idx``);
        * the stored public snapshot (wall count, table state, river/riichi/meld
          counts, scores, dora indicators) equals an incremental replay of the
          stream prefix.  This catches a tampered ``before_event_idx`` without the
          cost of rebuilding the tile ledger.

        ``verify_state_snapshots=True`` additionally runs the element-wise
        ``check_prefix(..., verify="exact")`` and replays the full
        :class:`~riichiformer.data.decision._SnapshotTracker` (tile ledger
        included), comparing the complete stored snapshot.  It is opt-in because
        that replay costs tens of microseconds per event on long streams.
        """
        structural = _ObservablePrefixState()
        applied = 0
        for index, (cutoff, snapshot, payload) in enumerate(
                zip(cutoffs, snapshots, payloads)):
            if cutoff != len(snapshot.events):
                raise ShardLoadError(
                    f"decision[{index}] before_event_idx {cutoff} != visible prefix "
                    f"length {len(snapshot.events)}")
            if int(payload["kyoku_idx"]) != events[cutoff - 1].kyoku_idx:
                raise ShardLoadError(
                    f"decision[{index}] kyoku_idx {payload['kyoku_idx']} != cutoff "
                    f"event kyoku {events[cutoff - 1].kyoku_idx}")
            while applied < cutoff:
                structural.apply(events[applied])
                applied += 1
            structural.check(snapshot, game_id=game_id, pov=pov, index=index)
        if not verify_state_snapshots:
            return
        for index, cutoff in enumerate(cutoffs):
            try:
                check_prefix(events[:cutoff], events, cutoff, verify="exact")
            except MultiDecisionError as exc:
                raise ShardLoadError(
                    f"decision[{index}] exact prefix mismatch: {exc}") from exc
        config = self._multi_config(snapshots[0])
        tracker = _SnapshotTracker(config)
        applied = 0
        for index, (cutoff, snapshot) in enumerate(zip(cutoffs, snapshots)):
            try:
                while applied < cutoff:
                    tracker.apply(events[applied])
                    applied += 1
                opportunity = SimpleNamespace(
                    before_event_idx=cutoff,
                    phase=snapshot.phase,
                    mask=snapshot.legal_mask,
                )
                rebuilt = tracker.build_input(events[:cutoff], opportunity, None)
            except Exception as exc:  # noqa: BLE001 - re-raised as a shard error
                raise ShardLoadError(
                    f"decision[{index}] snapshot replay failed: "
                    f"{type(exc).__name__}: {exc}") from exc
            if rebuilt.tile_state != snapshot.tile_state \
                    or rebuilt.players_public != snapshot.players_public \
                    or rebuilt.global_public != snapshot.global_public:
                raise ShardLoadError(
                    f"decision[{index}] full state snapshot does not match the "
                    f"stream prefix ending at before_event_idx={cutoff}")

    def _multi_decisions_for_pov(self, game: dict, pov: int, stream, rows, *,
                                 chunk_cap: int, max_events: int,
                                 max_positions: int | None,
                                 verify_state_snapshots: bool) -> list:
        """Build markers/chunks for one POV without re-validating every prefix.

        The PR-10 ``DecisionInput`` prefix validation is O(prefix length) per
        decision; on a 4,800-event POV with hundreds of decisions that dominates
        loader time.  The loader therefore builds the target-free
        :class:`DecisionMarker` directly from the stored snapshot (the same state
        vector :func:`decision_marker_from_input` would build), while its own
        exact-prefix/observable-state/audit checks pin the cutoff to the stream.
        """
        game_id = game["game_id"]
        meta = self._stream_meta(pov)
        stream_events = tuple(event_from_dict(event) for event in stream)
        markers = []
        snapshots = []
        payloads = []
        targets = []
        cutoffs = []
        indices = []
        for row in rows:
            if int(row["pov"]) != pov:
                raise ShardLoadError(
                    f"input row pov {row['pov']} != grouped pov {pov}")
            payload = _decision_payload(row)
            cutoff = int(payload["before_event_idx"])
            if cutoff < 1 or cutoff > len(stream):
                raise ShardLoadError(
                    f"game {game_id} POV {pov} decision cutoff {cutoff} outside "
                    f"stream 0..{len(stream)}")
            if stream_events[cutoff - 1].seq_idx != cutoff - 1:
                raise ShardLoadError(
                    "stored POV stream is not contiguous at the cutoff")
            tile_state = _tile_state(payload["tile_state"])
            players = tuple(_player(player) for player in payload["players_public"])
            public = _global(payload["global_public"])
            phase = ActionPhase(payload["phase"])
            legal_mask = tuple(bool(v) for v in payload["legal_mask"])
            snapshot = SimpleNamespace(
                events=stream_events[:cutoff], tile_state=tile_state,
                players_public=players, global_public=public, phase=phase,
                legal_mask=legal_mask)
            try:
                features = state_features_from_snapshot(_state_snapshot(snapshot))
                marker = DecisionMarker(
                    cutoff_idx=cutoff, phase=phase, legal_mask=legal_mask,
                    state_features=features)
            except (EncoderError, MultiDecisionError, ValueError) as exc:
                raise ShardLoadError(
                    f"game {game_id} POV {pov} decision invalid: {exc}") from exc
            markers.append(marker)
            snapshots.append(snapshot)
            payloads.append(payload)
            targets.append(self._load_target(row["decision_id"]))
            cutoffs.append(cutoff)
            indices.append(int(row["decision_index"]))
        if indices != list(range(len(indices))):
            raise ShardLoadError(
                f"game {game_id} POV {pov} decision_index is not contiguous "
                f"0..N-1: {indices[:8]}...")
        previous_cutoff = 0
        for index, cutoff in enumerate(cutoffs):
            if cutoff < previous_cutoff:
                raise ShardLoadError(
                    f"game {game_id} POV {pov} decision[{index}] cutoff "
                    f"{cutoff} < previous {previous_cutoff}")
            previous_cutoff = cutoff
        self._check_multi_prefixes(
            game_id, pov, stream_events, cutoffs, snapshots, payloads,
            verify_state_snapshots=verify_state_snapshots)
        # Chunk on decision boundaries while respecting both limits: the event
        # prefix (max_events) and, when the caller knows the model, the joined
        # event+marker length (max_positions).  A single decision whose own prefix
        # cannot fit stays a hard error; no event or decision is ever dropped.
        units = []
        start = 0
        while start < len(markers):
            first_cutoff = cutoffs[start]
            if first_cutoff > max_events:
                raise ShardLoadError(
                    f"game {game_id} POV {pov} decision[{indices[start]}] prefix "
                    f"{first_cutoff} > max_events {max_events}; refusing to "
                    "truncate a single decision")
            if max_positions is not None and first_cutoff + 1 > max_positions:
                raise ShardLoadError(
                    f"game {game_id} POV {pov} decision[{indices[start]}] joined "
                    f"length {first_cutoff + 1} > max_positions {max_positions}")
            end = start + 1
            last_cutoff = first_cutoff
            while end < len(markers) and (end - start) < chunk_cap:
                cutoff = cutoffs[end]
                count = end - start + 1
                if cutoff > max_events:
                    break
                if max_positions is not None and cutoff + count > max_positions:
                    break
                last_cutoff = cutoff
                end += 1
            chunk_markers = tuple(markers[start:end])
            chunk_targets = tuple(targets[start:end])
            chunk_indices = tuple(indices[start:end])
            prefix = stream_events[:last_cutoff]
            try:
                multi_stream = MultiDecisionStream(
                    events=prefix, decisions=chunk_markers, meta=meta)
            except MultiDecisionError as exc:
                raise ShardLoadError(
                    f"game {game_id} POV {pov} multi-decision stream invalid: "
                    f"{exc}") from exc
            units.append(_MultiStreamUnit(
                game_id=game_id, pov=pov, meta=meta, events=prefix,
                stream=multi_stream, targets=chunk_targets,
                decision_indices=chunk_indices))
            start = end
        return units

    def iter_multi_decision_batches(
            self, batch_size: int = 8,
            max_decisions: int | None = None,
            *, max_events: int = MAX_EVENTS_DEFAULT,
            max_positions: int | None = None,
            verify_state_snapshots: bool = False) -> Iterator[MultiShardBatch]:
        """Deterministic PR-38 batches grouped by ``(game_id, pov)``.

        * ``batch_size`` is the maximum number of **decisions** (marker slots) in
          one yielded batch.
        * A POV chunk is split on decision boundaries whenever adding the next
          decision would exceed ``max_events`` (event prefix) or ``max_positions``
          (joined event + marker length).  Every chunk keeps the **same full-stream
          prefix** through its last cutoff, cutoff order is preserved and nothing is
          truncated.  ``max_decisions`` optionally lowers the per-POV marker cap
          (clamped to ``batch_size``) but never overrides the event limits.
        * A single decision whose own prefix exceeds ``max_events``/``max_positions``
          is a hard :class:`ShardLoadError`; it is never dropped or truncated.
        * All decisions of one POV chunk share one row and one batch; POVs of the
          same game may be padded together, but a batch never mixes games and a
          POV never contributes two rows to the same batch.
        * ``verify_state_snapshots=True`` adds the full tile-ledger snapshot replay
          (audit mode); the default cheap check still validates the exact prefix
          and the observable public state at every cutoff.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) \
                or batch_size <= 0:
            raise ShardLoadError("batch_size must be a positive int")
        if max_decisions is not None and (
                isinstance(max_decisions, bool)
                or not isinstance(max_decisions, int) or max_decisions <= 0):
            raise ShardLoadError("max_decisions must be a positive int or None")
        if isinstance(max_events, bool) or not isinstance(max_events, int) \
                or max_events <= 0:
            raise ShardLoadError("max_events must be a positive int")
        if max_positions is not None and (
                isinstance(max_positions, bool)
                or not isinstance(max_positions, int) or max_positions <= 0):
            raise ShardLoadError("max_positions must be a positive int or None")
        if not isinstance(verify_state_snapshots, bool):
            raise ShardLoadError("verify_state_snapshots must be bool")
        chunk_cap = batch_size if max_decisions is None \
            else min(batch_size, max_decisions)
        for game in self.games():
            game_id = game["game_id"]
            rows = self._conn.execute(
                "SELECT * FROM inputs WHERE game_id=? ORDER BY pov,decision_index",
                (game_id,)).fetchall()
            if len(rows) != int(game["decision_count"]):
                raise ShardLoadError(
                    f"game {game_id} stores {len(rows)} inputs but declares "
                    f"decision_count={game['decision_count']}")
            if not rows:
                continue
            target_rows = self._conn.execute(
                "SELECT COUNT(*) FROM targets t JOIN inputs i "
                "ON i.decision_id=t.decision_id WHERE i.game_id=?",
                (game_id,)).fetchone()[0]
            if int(target_rows) != len(rows):
                raise ShardLoadError(
                    f"game {game_id} has {target_rows} targets for {len(rows)} "
                    "decisions")
            streams = tuple(tuple(stream) for stream in game["pov_streams"])
            by_pov: dict[int, list] = {}
            for row in rows:
                by_pov.setdefault(int(row["pov"]), []).append(row)
            pending: list = []
            pending_decisions = 0
            pending_povs: set[int] = set()

            def batch_fits(units) -> bool:
                if max_positions is None:
                    return True
                longest = max(len(unit.events) for unit in units)
                markers = max(len(unit.targets) for unit in units)
                return longest + markers <= max_positions

            for pov in sorted(by_pov):
                if not 0 <= pov < len(streams):
                    raise ShardLoadError(
                        f"game {game_id} input POV {pov} outside stored streams")
                units = self._multi_decisions_for_pov(
                    game, pov, streams[pov], by_pov[pov], chunk_cap=chunk_cap,
                    max_events=max_events, max_positions=max_positions,
                    verify_state_snapshots=verify_state_snapshots)
                for unit in units:
                    if pending and (
                            pov in pending_povs
                            or pending_decisions + len(unit.targets) > batch_size
                            or not batch_fits(pending + [unit])):
                        yield self._make_multi_batch(
                            tuple(pending), max_events=max_events)
                        pending = []
                        pending_decisions = 0
                        pending_povs = set()
                    if not batch_fits([unit]):
                        raise ShardLoadError(
                            f"game {game_id} POV {pov} chunk of "
                            f"{len(unit.targets)} decisions needs "
                            f"{len(unit.events)} events + markers > max_positions "
                            f"{max_positions}")
                    pending.append(unit)
                    pending_decisions += len(unit.targets)
                    pending_povs.add(pov)
            if pending:
                yield self._make_multi_batch(tuple(pending), max_events=max_events)

    def _make_multi_batch(self, units, *, max_events: int) -> MultiShardBatch:
        try:
            encoded = encode_multi_decision_batch(
                [unit.stream for unit in units], max_events=max_events,
                verify_prefix="exact")
        except (EncoderError, MultiDecisionError, ModelBatchError) as exc:
            raise ShardLoadError(f"multi-decision encode failed: {exc}") from exc
        targets = tuple(target for unit in units for target in unit.targets)
        if encoded.n_decisions != len(targets):
            raise ShardLoadError(
                f"encoded decisions {encoded.n_decisions} != targets {len(targets)}")
        return MultiShardBatch(
            batch=encoded,
            targets=targets,
            metas=tuple(unit.meta for unit in units for _ in unit.targets),
            lengths=tuple(len(unit.events) for unit in units
                          for _ in unit.targets),
            game_ids=tuple(unit.game_id for unit in units for _ in unit.targets),
            povs=tuple(unit.pov for unit in units for _ in unit.targets),
            decision_indices=tuple(index for unit in units
                                   for index in unit.decision_indices),
            decision_stream_index=tuple(row for row, unit in enumerate(units)
                                        for _ in unit.targets),
        )

    def _make_batch(self, items, bucket_by_length: bool) -> ShardBatch:
        if bucket_by_length:
            items = sorted(items, key=lambda item: (
                len(item[0].events), item[3], item[4], item[5]))
        decisions = [item[0] for item in items]
        targets = tuple(item[1] for item in items)
        metas = tuple(item[2] for item in items)
        encoded: EncodedBatch = encode_batch(decisions, metas)
        return ShardBatch(
            batch=encoded.to_model_batch(), targets=targets, metas=metas,
            lengths=encoded.lengths,
            game_ids=tuple(item[3] for item in items),
            povs=tuple(item[4] for item in items),
            decision_indices=tuple(item[5] for item in items),
        )


__all__ = ["MultiShardBatch", "ShardBatch", "ShardLoadError", "ShardReader"]
