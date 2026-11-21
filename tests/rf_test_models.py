#!/usr/bin/env python3
"""PR-13 MVP-S model tests (synthetic only, CPU smoke required).

Covers the acceptance list from the V3.1 roadmap PR-13 / Recipe E:

* batch and ``[B, L, d]`` shapes (event stream, state injection, decision gather);
* illegal actions get exact zero probability (``-inf`` masked logits);
* causal prefix invariance (eval, dropout=0: future events cannot change earlier
  decision logits);
* different POV sequences do not share state (batch independence, no leakage);
* all gradients finite for the full MVP-S configuration on CPU;
* strict ``state_dict`` roundtrip between independent models;
* parameter count pinned to the measured MVP-S size and inside the declared budget;
* synthetic overfit: fixed tiny batch, loss decreases, argmax matches labels;
* CPU forward/backward; optional one-step BF16 GPU smoke (skipped without CUDA).

Also guards the frozen interface: the YAML config's schema/action hashes must match
``configs/riichiformer/schema_v3.json``, legal masks are validated against the frozen
46-action phase sets, and no label/target/oracle field exists on ``ModelBatch`` or
``Policy.forward``.

Run::

    PYTHONPATH= /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q tests/rf_test_models.py
    PYTHONPATH= /home/ygsldr/riichi-ai/.venv/bin/python tests/rf_test_models.py bench --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import inspect
import json
import math
import sys
import time
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.models import (  # noqa: E402
    EVENT_TENSOR_FIELDS,
    EVENT_TENSOR_VOCABS,
    MVPConfig,
    MVPConfigError,
    PARAMETER_BUDGET_MAX,
    PARAMETER_BUDGET_MIN,
    PHASE_LEGAL_INDEX_SETS,
    PHASE_ORDER,
    EventEmbedding,
    EventEmbeddingError,
    LocalTransformer,
    ModelBatch,
    ModelBatchError,
    Policy,
    PolicyOutput,
    StateFeatureError,
    StateMLP,
    StateSnapshot,
    TransformerShapeError,
    build_model,
    count_parameters,
    parameter_breakdown,
    state_features_from_snapshot,
)
from riichiformer.models.batch import (  # noqa: E402
    ANKAN_EVENT_TYPE,
    DAIMINKAN_EVENT_TYPE,
    KAKAN_EVENT_TYPE,
    PHASE_INDEX,
    TSUMO_EVENT_TYPE,
)
from riichiformer.models.event_embedding import (  # noqa: E402
    ACTOR_REL_VOCAB,
    AKA_VOCAB,
    EVENT_TYPE_VOCAB,
    KYOKU_POS_VOCAB,
    PHASE_VOCAB,
    TILE_KIND_VOCAB,
    TSUMOGIRI_VOCAB,
)
from riichiformer.models.state_encoder import STATE_FEATURE_DIM  # noqa: E402
from riichiformer.schemas import action_v3, schema_v3  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "riichiformer" / "mvp_s.yaml"

#: Measured MVP-S parameter count at the frozen PR-13 configuration.
EXPECTED_MVP_S_PARAMETERS = 5_217_582

#: Fully self-contained CPU overfit budget (must stay short in CI).
OVERFIT_STEPS = 200


# ---------------------------------------------------------------------------
# synthetic batch construction helpers
# ---------------------------------------------------------------------------


def _randint(generator: torch.Generator, low: int, high: int, shape) -> torch.Tensor:
    return torch.randint(low, high, shape, generator=generator)


def _even_decision_times(length: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if length < 2 * count + 1:
        raise ValueError(f"length {length} too short for {count} decisions")
    times = torch.linspace(1, length - 1, count + 2)[1:-1].round().long().tolist()
    if len(set(times)) != count:
        raise AssertionError(f"decision times are not unique: {times}")
    return times


def _visibility_valid_event_fields(
    generator: torch.Generator, batch_size: int, seq_len: int
) -> dict[str, torch.Tensor]:
    """Random event ids that satisfy the POV visibility contract."""
    shape = (batch_size, seq_len)
    event_type = _randint(generator, 0, EVENT_TYPE_VOCAB, shape)
    actor = _randint(generator, 0, ACTOR_REL_VOCAB, shape)
    tile = _randint(generator, 0, TILE_KIND_VOCAB, shape)
    aka = torch.zeros(shape, dtype=torch.long)
    with_aka = _randint(generator, 0, 4, shape) == 0
    colour = _randint(generator, 1, AKA_VOCAB, shape)  # embedding index 1..3
    aka = torch.where(with_aka, colour, aka)
    for colour_index, kind in enumerate(action_v3.AKA_KINDS):
        aka_index = colour_index + 1
        tile = torch.where(aka == aka_index, torch.full_like(tile, kind), tile)
    unknown = _randint(generator, 0, 8, shape) == 0
    tile = torch.where(unknown, torch.full_like(tile, action_v3.UNKNOWN_TILE_KIND),
                       tile)
    aka = torch.where(tile == action_v3.UNKNOWN_TILE_KIND, torch.zeros_like(aka), aka)
    opponent_private = (
        (event_type == TSUMO_EVENT_TYPE) | (event_type == ANKAN_EVENT_TYPE)
    ) & (actor != 0)
    tile = torch.where(opponent_private,
                       torch.full_like(tile, action_v3.UNKNOWN_TILE_KIND), tile)
    aka = torch.where(opponent_private, torch.zeros_like(aka), aka)
    return {
        "event_type_id": event_type,
        "event_actor_rel": actor,
        "event_tile_kind": tile,
        "event_aka": aka,
        "event_tsumogiri": _randint(generator, 0, TSUMOGIRI_VOCAB, shape),
    }


def make_batch(
    *,
    batch_size: int = 2,
    seq_len: int = 24,
    decisions_per_row: int = 2,
    seed: int = 0,
    device: str = "cpu",
    lengths: list[int] | None = None,
    legal_size: int = 3,
    phases: list[int] | None = None,
) -> ModelBatch:
    """Build a deterministic random-but-valid :class:`ModelBatch` for tests."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if lengths is None:
        lengths = [seq_len] * batch_size
    if len(lengths) != batch_size:
        raise ValueError("lengths must have one entry per batch row")
    if any(length <= 0 or length > seq_len for length in lengths):
        raise ValueError("lengths must be in 1..seq_len")

    mask = (
        torch.arange(seq_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    ).to(device)

    decision_rows: list[int] = []
    decision_times: list[int] = []
    decision_phases: list[int] = []
    phase_cursor = 0
    for row, length in enumerate(lengths):
        for time_index in _even_decision_times(length, decisions_per_row):
            decision_rows.append(row)
            decision_times.append(time_index)
            if phases is not None:
                decision_phases.append(phases[phase_cursor % len(phases)])
            else:
                decision_phases.append(
                    int(_randint(generator, 0, len(PHASE_ORDER), (1,)).item())
                )
            phase_cursor += 1

    n_decisions = len(decision_rows)
    legal = torch.zeros(n_decisions, action_v3.ACTION_SPACE, dtype=torch.bool)
    for index, phase_index in enumerate(decision_phases):
        allowed = sorted(PHASE_LEGAL_INDEX_SETS[phase_index])
        take = min(legal_size, len(allowed))
        chosen = torch.randperm(len(allowed), generator=generator)[:take].tolist()
        for offset in chosen:
            legal[index, allowed[offset]] = True

    event_fields = {
        name: value.to(device)
        for name, value in _visibility_valid_event_fields(
            generator, batch_size, seq_len
        ).items()
    }
    return ModelBatch(
        **event_fields,
        event_phase=_randint(generator, 0, PHASE_VOCAB, (batch_size, seq_len)).to(device),
        event_kyoku_pos=(torch.arange(seq_len, dtype=torch.long) % KYOKU_POS_VOCAB)
        .unsqueeze(0).repeat(batch_size, 1).to(device),
        padding_mask=mask,
        state_features=torch.randn(n_decisions, STATE_FEATURE_DIM,
                                   generator=generator).to(device),
        decision_batch_index=torch.tensor(decision_rows, dtype=torch.long, device=device),
        decision_time_index=torch.tensor(decision_times, dtype=torch.long, device=device),
        legal_mask=legal.to(device),
        decision_phase=torch.tensor(decision_phases, dtype=torch.long, device=device),
    )


def tiny_config(**overrides) -> MVPConfig:
    base = dict(layers=2, ffn=64, state_hidden=64, max_positions=1024)
    base.update(overrides)
    return MVPConfig(**base)


def clone_batch(batch: ModelBatch, **changes) -> ModelBatch:
    fields = {
        field.name: getattr(batch, field.name).clone()
        for field in dataclasses.fields(batch)
        if isinstance(getattr(batch, field.name), torch.Tensor)
    }
    fields["schema_version"] = batch.schema_version
    fields.update(changes)
    return ModelBatch(**fields)


def make_pov_batch(seq_len: int = 16, decisions_per_row: int = 2,
                   seed: int = 13) -> ModelBatch:
    """A hand-built PR-09-projected-like stream: public tiles stay known, opponent
    draws and opponent ankan stay UNKNOWN, own tsumo/ankan may be known."""
    rows: list[tuple[int, int, int, int, int]] = [
        (0, 4, 34, 0, 0),   # start_game (no tile)
        (1, 4, 33, 0, 0),   # start_kyoku public dora marker
        (2, 0, 4, 1, 0),    # own tsumo: aka 5m
        (3, 0, 4, 1, 1),    # own dahai tsumogiri of that aka 5m
        (2, 1, 34, 0, 0),   # opponent tsumo: hidden
        (3, 1, 2, 0, 0),    # opponent dahai: public 3m
        (2, 0, 13, 2, 0),   # own tsumo: aka 5p
        (8, 0, 13, 2, 0),   # own ankan: known to self
        (2, 3, 34, 0, 0),   # opponent tsumo: hidden
        (8, 3, 34, 0, 0),   # opponent ankan: concealed
        (6, 2, 7, 0, 0),    # daiminkan: called tile is public
        (7, 0, 9, 0, 0),    # kakan: added tile is public
        (9, 1, 34, 0, 0),   # reach
        (10, 1, 34, 0, 0),  # reach_accepted
        (11, 4, 22, 0, 0),  # dora indicator (public 5s)
        (12, 2, 34, 0, 0),  # hora
    ]
    if len(rows) > seq_len:
        raise ValueError(f"seq_len {seq_len} < {len(rows)} fixture events")
    filler = (0, 4, 34, 0, 0)
    rows = rows + [filler] * (seq_len - len(rows))
    batch = make_batch(batch_size=1, seq_len=seq_len,
                       decisions_per_row=decisions_per_row, seed=seed)
    fields = {
        "event_type_id": torch.tensor([[row[0] for row in rows]]),
        "event_actor_rel": torch.tensor([[row[1] for row in rows]]),
        "event_tile_kind": torch.tensor([[row[2] for row in rows]]),
        "event_aka": torch.tensor([[row[3] for row in rows]]),
        "event_tsumogiri": torch.tensor([[row[4] for row in rows]]),
    }
    return clone_batch(batch, **fields)


def _batch_with_first_token(event_type: int, actor: int, tile: int, aka: int,
                            *, seq_len: int = 6, seed: int = 31) -> ModelBatch:
    batch = make_batch(batch_size=1, seq_len=seq_len, decisions_per_row=1, seed=seed)
    fields = {}
    for name, value in (
        ("event_type_id", event_type),
        ("event_actor_rel", actor),
        ("event_tile_kind", tile),
        ("event_aka", aka),
    ):
        tensor = getattr(batch, name).clone()
        tensor[0, 0] = value
        fields[name] = tensor
    return clone_batch(batch, **fields)


def truncate_batch(batch: ModelBatch, length: int) -> ModelBatch:
    keep = batch.decision_time_index < length
    fields = {
        name: getattr(batch, name)[:, :length].clone() for name in EVENT_TENSOR_FIELDS
    }
    fields["padding_mask"] = batch.padding_mask[:, :length].clone()
    fields["state_features"] = batch.state_features[keep].clone()
    fields["decision_batch_index"] = batch.decision_batch_index[keep].clone()
    fields["decision_time_index"] = batch.decision_time_index[keep].clone()
    fields["legal_mask"] = batch.legal_mask[keep].clone()
    fields["decision_phase"] = batch.decision_phase[keep].clone()
    return ModelBatch(schema_version=batch.schema_version, **fields)


def labels_from_batch(batch: ModelBatch, *, seed: int = 7) -> torch.Tensor:
    """A deterministic, legal, non-constant function of the state features."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(action_v3.ACTION_SPACE, generator=generator).to(
        batch.state_features.device
    )
    scores = batch.state_features[:, :action_v3.ACTION_SPACE] * weights
    labels: list[int] = []
    for index in range(batch.n_decisions):
        legal = torch.nonzero(batch.legal_mask[index], as_tuple=False).flatten()
        best = legal[int(torch.argmax(scores[index, legal]))]
        labels.append(int(best))
    return torch.tensor(labels, dtype=torch.long, device=batch.state_features.device)


def masked_cross_entropy(output: PolicyOutput, labels: torch.Tensor) -> torch.Tensor:
    picked = output.logprobs.gather(1, labels.unsqueeze(1)).squeeze(1)
    return -picked.mean()


# ---------------------------------------------------------------------------
# module / config tests
# ---------------------------------------------------------------------------


class TestConfigAndContract(unittest.TestCase):
    def test_yaml_matches_dataclass_and_frozen_hashes(self):
        self.assertTrue(CONFIG_PATH.is_file(), f"missing config {CONFIG_PATH}")
        config = MVPConfig.from_yaml(CONFIG_PATH)
        self.assertEqual(config, MVPConfig())
        self.assertEqual(config.hidden, 256)
        self.assertEqual(config.layers, 6)
        self.assertEqual(config.heads, 4)
        self.assertEqual(config.ffn, 704)
        self.assertEqual(config.state_hidden, 512)
        self.assertEqual(config.dropout, 0.0)
        self.assertFalse(config.memory)

        import yaml

        document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        train = document["train"]
        self.assertEqual(train["precision"], "bf16")
        self.assertEqual(train["attention"], "sdpa")
        self.assertEqual(train["max_sequence_policy"],
                         "dynamic_bucket_no_silent_truncate")
        self.assertIs(train["dataset_training"], False)
        self.assertEqual(document["contract"]["action_space"], action_v3.ACTION_SPACE)
        # Schema version/hashes are resolved from code at runtime, never hardcoded:
        # a v3/1 -> v3/2 bump with an unchanged action map must not break PR-13.
        schema = schema_v3.load_schema()
        self.assertEqual(schema["schema_version"], action_v3.SCHEMA_VERSION)
        self.assertEqual(schema["action_map_hash"], action_v3.ACTION_MAP_HASH)
        self.assertEqual(MVPConfig().name, "mvp_s")

    def test_config_rejects_drift(self):
        with self.assertRaises(MVPConfigError):
            MVPConfig.from_mapping({"hidden": 256, "bogus_key": 1})
        with self.assertRaises(MVPConfigError):
            MVPConfig(memory=True)
        with self.assertRaises(MVPConfigError):
            MVPConfig(architecture="flat")
        with self.assertRaises(MVPConfigError):
            MVPConfig(state_encoder="linear")
        with self.assertRaises(MVPConfigError):
            MVPConfig(policy_head="candidate")
        with self.assertRaises(MVPConfigError):
            MVPConfig(hidden=250, heads=4)
        with self.assertRaises(MVPConfigError):
            MVPConfig(dropout=1.0)
        with self.assertRaises(MVPConfigError):
            MVPConfig(max_positions=0)
        self.assertEqual(MVPConfig.from_mapping({"hidden": 128, "heads": 2}).hidden, 128)

    def test_phase_order_matches_schema_enum(self):
        self.assertEqual(PHASE_ORDER, (action_v3.ActionPhase.SELF_TURN,
                                       action_v3.ActionPhase.CALL_RESPONSE,
                                       action_v3.ActionPhase.KAN_SELECTION))
        self.assertEqual(PHASE_INDEX[action_v3.ActionPhase.KAN_SELECTION], 2)
        for index, phase in enumerate(PHASE_ORDER):
            self.assertEqual(PHASE_LEGAL_INDEX_SETS[index],
                             frozenset(action_v3.PHASE_LEGAL_ACTIONS[phase]))
        self.assertEqual(action_v3.ACTION_SPACE, 46)

    def test_no_target_or_oracle_fields(self):
        forbidden = ("label", "target", "oracle", "rank_final", "expert_action",
                     "censor", "supervision")
        for field in dataclasses.fields(ModelBatch):
            for word in forbidden:
                self.assertNotIn(word, field.name,
                                 f"ModelBatch must not carry target field {field.name!r}")
        signature = inspect.signature(Policy.forward)
        self.assertEqual(list(signature.parameters), ["self", "batch"])
        model = Policy(tiny_config())
        for name, _ in model.named_parameters():
            for word in forbidden:
                self.assertNotIn(word, name)

    def test_parameter_count_matches_budget(self):
        model = build_model(MVPConfig())
        total = count_parameters(model)
        self.assertEqual(total, EXPECTED_MVP_S_PARAMETERS)
        self.assertGreaterEqual(total, PARAMETER_BUDGET_MIN)
        self.assertLessEqual(total, PARAMETER_BUDGET_MAX)
        breakdown = parameter_breakdown(model)
        self.assertEqual(breakdown["total"], total)
        self.assertEqual(sum(value for key, value in breakdown.items() if key != "total"),
                         total)
        self.assertEqual(breakdown["policy_head"], 256 * 46 + 46)
        self.assertEqual(model.policy_head.linear.out_features, 46)


class TestEventEmbedding(unittest.TestCase):
    def setUp(self):
        self.embedding = EventEmbedding(32, dropout=0.0)
        self.embedding.eval()

    def _inputs(self, batch_size=2, seq_len=6, seed=0, mask=None):
        generator = torch.Generator().manual_seed(seed)
        ids = {
            "event_type_id": _randint(generator, 0, EVENT_TYPE_VOCAB, (batch_size, seq_len)),
            "event_actor_rel": _randint(generator, 0, ACTOR_REL_VOCAB,
                                        (batch_size, seq_len)),
            "event_tile_kind": _randint(generator, 0, TILE_KIND_VOCAB,
                                        (batch_size, seq_len)),
            "event_aka": _randint(generator, 0, AKA_VOCAB, (batch_size, seq_len)),
            "event_tsumogiri": _randint(generator, 0, TSUMOGIRI_VOCAB,
                                        (batch_size, seq_len)),
            "event_phase": _randint(generator, 0, PHASE_VOCAB, (batch_size, seq_len)),
            "event_kyoku_pos": torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
        }
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        return ids, mask

    def test_shape_and_padding_zero(self):
        ids, mask = self._inputs()
        mask[0, -2:] = False
        out = self.embedding(**ids, padding_mask=mask)
        self.assertEqual(tuple(out.shape), (2, 6, 32))
        self.assertEqual(float(out[~mask].detach().abs().max()), 0.0)
        self.assertTrue(torch.isfinite(out).all())

    def test_padded_content_does_not_matter(self):
        ids, mask = self._inputs(mask=torch.ones(2, 6, dtype=torch.bool))
        mask[1, -3:] = False
        first = self.embedding(**ids, padding_mask=mask)
        mutated = {key: value.clone() for key, value in ids.items()}
        mutated["event_tile_kind"][1, -3:] = 34
        mutated["event_type_id"][1, -3:] = 16
        mutated["event_kyoku_pos"][1, -3:] = 511
        second = self.embedding(**mutated, padding_mask=mask)
        self.assertTrue(torch.allclose(first, second, atol=0.0, rtol=0.0))

    def test_out_of_vocab_rejected(self):
        ids, mask = self._inputs()
        bad = dict(ids)
        bad["event_tile_kind"] = ids["event_tile_kind"].clone()
        bad["event_tile_kind"][0, 0] = TILE_KIND_VOCAB
        with self.assertRaises(EventEmbeddingError):
            self.embedding(**bad, padding_mask=mask)
        bad = dict(ids)
        bad["event_kyoku_pos"] = ids["event_kyoku_pos"].clone()
        bad["event_kyoku_pos"][0, 0] = KYOKU_POS_VOCAB
        with self.assertRaises(EventEmbeddingError):
            self.embedding(**bad, padding_mask=mask)
        with self.assertRaises(EventEmbeddingError):
            self.embedding(**ids, padding_mask=mask.to(torch.long))

    def test_vocab_sizes_match_frozen_schema(self):
        vocab = self.embedding.vocab_sizes
        self.assertEqual(vocab["event_type"], 16 + 1)
        self.assertEqual(vocab["tile_kind"], action_v3.TILE_KIND_COUNT + 1)
        self.assertEqual(vocab["tile_kind"], 35)
        self.assertEqual(vocab["aka"], 4)
        self.assertEqual(vocab["phase"], 3 + 1)


class TestStateEncoder(unittest.TestCase):
    def _snapshot(self, **overrides) -> StateSnapshot:
        base = dict(
            own_counts=tuple(
                1 if kind in (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13) else 0
                for kind in range(34)
            ),
            public_counts=(0,) * 34,
            own_aka=(0, 0, 0),
            public_aka=(0, 0, 0),
            player_scores=(25_000, 25_000, 25_000, 25_000),
        )
        base.update(overrides)
        return StateSnapshot(**base)

    def test_feature_layout(self):
        snapshot = self._snapshot()
        features = state_features_from_snapshot(snapshot)
        self.assertEqual(tuple(features.shape), (STATE_FEATURE_DIM,))
        self.assertEqual(STATE_FEATURE_DIM, 172)
        self.assertTrue(torch.isfinite(features).all())
        self.assertLessEqual(float(features.abs().max()), 4.0)
        # own_counts segment: thirteen ones / 4
        self.assertAlmostEqual(float(features[:34].sum()), 13 / 4)

    def test_full_snapshot_with_dora_and_scores(self):
        own = [0] * 34
        for kind in range(13):
            own[kind] += 1
        self.assertEqual(own[action_v3.AKA_KINDS[0]], 1)
        snapshot = self._snapshot(
            own_counts=tuple(own),
            own_aka=(1, 0, 0),
            public_counts=tuple(1 if index == 10 else 0 for index in range(34)),
            player_scores=(32_000, 24_000, 25_000, 19_000),
            player_riichi=(2, 0, 1, 0),
            player_river_counts=(4, 2, 10, 0),
            player_meld_counts=(0, 1, 0, 2),
            player_is_oya=(True, False, False, False),
            player_winds=(27, 28, 29, 30),
            bakaze_kind=27,
            kyoku=2,
            honba=3,
            kyotaku=0,
            tiles_left=42,
            oya_rel=0,
            dora_indicator_counts=tuple(1 if index in (0, 5, 33) else 0
                                        for index in range(34)),
            phase=2,
        )
        features = state_features_from_snapshot(snapshot)
        self.assertEqual(tuple(features.shape), (STATE_FEATURE_DIM,))
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(float(features[STATE_FEATURE_DIM - 1]), 1.0)  # phase one-hot

    def test_snapshot_invariants_reject_leaks(self):
        with self.assertRaises(StateFeatureError):
            self._snapshot(own_counts=(5,) + (0,) * 33)
        with self.assertRaises(StateFeatureError):
            self._snapshot(own_aka=(1, 0, 0))  # no own 5m in the default hand
        with self.assertRaises(StateFeatureError):
            self._snapshot(player_scores=(25_000, 25_000, 25_000, 24_000))
        with self.assertRaises(StateFeatureError):
            self._snapshot(player_winds=(27, 29, 28, 30))
        with self.assertRaises(StateFeatureError):
            self._snapshot(phase=3)
        with self.assertRaises(StateFeatureError):
            self._snapshot(own_counts=(1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1) + (0,) * 21,
                           public_counts=(0, 0, 0, 0, 1) + (0,) * 29)

    def test_mlp_shapes_and_guards(self):
        mlp = StateMLP(hidden=32, d_model=16)
        features = torch.randn(5, STATE_FEATURE_DIM)
        out = mlp(features)
        self.assertEqual(tuple(out.shape), (5, 16))
        with self.assertRaises(StateFeatureError):
            mlp(torch.randn(5, STATE_FEATURE_DIM - 1))
        bad = features.clone()
        bad[0, 0] = float("nan")
        with self.assertRaises(StateFeatureError):
            mlp(bad)
        with self.assertRaises(StateFeatureError):
            mlp(features.to(torch.long))
        with self.assertRaises(ValueError):
            StateMLP(input_dim=STATE_FEATURE_DIM - 1)


class TestLocalTransformer(unittest.TestCase):
    def _transformer(self, **overrides) -> LocalTransformer:
        base = dict(d_model=32, n_layers=2, n_heads=4, ffn_hidden=64,
                    max_positions=32, dropout=0.0)
        base.update(overrides)
        return LocalTransformer(**base)

    def test_shape_preserved(self):
        transformer = self._transformer().eval()
        x = torch.randn(2, 12, 32)
        mask = torch.ones(2, 12, dtype=torch.bool)
        mask[1, -3:] = False
        out = transformer(x, mask)
        self.assertEqual(tuple(out.shape), (2, 12, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_no_silent_truncation(self):
        transformer = self._transformer(max_positions=16)
        x = torch.randn(1, 17, 32)
        mask = torch.ones(1, 17, dtype=torch.bool)
        with self.assertRaises(TransformerShapeError):
            transformer(x, mask)
        out = transformer(torch.randn(1, 16, 32), torch.ones(1, 16, dtype=torch.bool))
        self.assertEqual(out.size(1), 16)
        with self.assertRaises(TransformerShapeError):
            transformer(torch.randn(1, 16, 24), torch.ones(1, 16, dtype=torch.bool))
        with self.assertRaises(TransformerShapeError):
            transformer(torch.randn(1, 16, 32), torch.ones(1, 15, dtype=torch.bool))

    def test_causal_future_independence(self):
        torch.manual_seed(0)
        transformer = self._transformer().eval()
        x = torch.randn(1, 12, 32)
        mask = torch.ones(1, 12, dtype=torch.bool)
        base = transformer(x, mask)
        mutated = x.clone()
        mutated[:, 7:] = torch.randn(1, 5, 32)
        changed = transformer(mutated, mask)
        self.assertTrue(torch.allclose(base[:, :7], changed[:, :7], atol=1e-6))
        self.assertFalse(torch.allclose(base[:, 7:], changed[:, 7:], atol=1e-6))

    def test_key_padding_mask_blocks_holes(self):
        torch.manual_seed(0)
        transformer = self._transformer().eval()
        mask = torch.tensor([[True, True, False, True, True, False, True]])
        x = torch.randn(1, 7, 32)
        first = transformer(x, mask)
        mutated = x.clone()
        mutated[:, ~mask[0]] = torch.randn(1, 2, 32)
        second = transformer(mutated, mask)
        self.assertTrue(torch.allclose(first[mask], second[mask], atol=1e-6))
        self.assertFalse(torch.allclose(first[:, ~mask[0]], second[:, ~mask[0]],
                                        atol=1e-6))

    def test_right_padding_fast_path_matches_explicit_mask(self):
        torch.manual_seed(0)
        transformer = self._transformer().eval()
        x = torch.randn(1, 8, 32)
        mask = torch.tensor([[True] * 5 + [False] * 3])
        fast = transformer(x, mask)
        self.assertTrue(torch.isfinite(fast).all())
        # Valid outputs do not depend on the padded tail.
        mutated = x.clone()
        mutated[:, 5:] = torch.randn(1, 3, 32)
        self.assertTrue(torch.allclose(fast[:, :5], transformer(mutated, mask)[:, :5],
                                       atol=1e-6))

    def test_rope_and_rmsnorm_shapes(self):
        block = self._transformer().blocks[0]
        self.assertEqual(block.attn.rope.head_dim, 8)
        self.assertEqual(block.attn.rope.max_positions, 32)
        out = block.attn_norm(torch.randn(3, 7, 32))
        self.assertEqual(tuple(out.shape), (3, 7, 32))
        self.assertEqual(tuple(block.ffn.w1.weight.shape), (64, 32))
        self.assertEqual(tuple(block.ffn.w2.weight.shape), (32, 64))


class TestModelBatchValidation(unittest.TestCase):
    def test_valid_batch_properties(self):
        batch = make_batch(batch_size=2, seq_len=16, decisions_per_row=2)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.seq_len, 16)
        self.assertEqual(batch.n_decisions, 4)
        self.assertEqual(tuple(batch.decision_indices.shape), (4, 2))
        self.assertEqual(tuple(batch.flat_decision_offsets.shape), (4,))
        self.assertEqual(batch.schema_version, action_v3.SCHEMA_VERSION)
        self.assertEqual(len(batch.decision_phases), 4)
        for phase in batch.decision_phases:
            self.assertIsInstance(phase, action_v3.ActionPhase)

    def test_prefix_padding_required(self):
        batch = make_batch(batch_size=2, seq_len=10, decisions_per_row=1)
        mask = batch.padding_mask.clone()
        mask[0, 5] = False
        mask[0, 8] = False
        with self.assertRaises(ModelBatchError):
            clone_batch(batch, padding_mask=mask)

    def test_decision_must_point_at_real_token(self):
        batch = make_batch(batch_size=1, seq_len=8, decisions_per_row=1)
        with self.assertRaises(ModelBatchError):
            clone_batch(batch, decision_time_index=torch.tensor([9]))
        with self.assertRaises(ModelBatchError):
            clone_batch(batch, decision_batch_index=torch.tensor([1]))

    def test_duplicate_or_empty_legal_mask_rejected(self):
        batch = make_batch(batch_size=1, seq_len=8, decisions_per_row=1)
        duplicated = torch.cat([batch.decision_time_index,
                                batch.decision_time_index])
        duplicated_batch = torch.cat([batch.decision_batch_index,
                                      batch.decision_batch_index])
        with self.assertRaises(ModelBatchError):
            clone_batch(
                batch,
                decision_time_index=duplicated,
                decision_batch_index=duplicated_batch,
                state_features=torch.cat([batch.state_features,
                                          batch.state_features]),
                legal_mask=torch.cat([batch.legal_mask, batch.legal_mask]),
                decision_phase=torch.cat([batch.decision_phase,
                                          batch.decision_phase]),
            )
        with self.assertRaises(ModelBatchError):
            clone_batch(batch, legal_mask=torch.zeros_like(batch.legal_mask))

    def test_phase_mask_contract_enforced(self):
        batch = make_batch(batch_size=1, seq_len=8, decisions_per_row=1,
                           legal_size=len(PHASE_LEGAL_INDEX_SETS[1]),
                           phases=[PHASE_INDEX[action_v3.ActionPhase.CALL_RESPONSE]])
        self.assertTrue(batch.legal_mask[0, 45])
        with self.assertRaises(ModelBatchError):
            clone_batch(batch, decision_phase=torch.tensor([0]))  # call mask in SELF_TURN
        # A SELF_TURN mask accepts discard and riichi.
        batch_self = make_batch(batch_size=1, seq_len=8, decisions_per_row=1,
                                phases=[0])
        illegal_phase = torch.zeros_like(batch_self.legal_mask)
        illegal_phase[0, 45] = True  # pass is CALL_RESPONSE only
        with self.assertRaises(ModelBatchError):
            clone_batch(batch_self, legal_mask=illegal_phase)

    def test_visibility_rejects_opponent_known_tsumo(self):
        with self.assertRaises(ModelBatchError):
            _batch_with_first_token(TSUMO_EVENT_TYPE, 1, 24, 0)
        with self.assertRaises(ModelBatchError):
            # even a well-formed aka five is concealed when the actor is an opponent
            _batch_with_first_token(TSUMO_EVENT_TYPE, 1, 4, 1)

    def test_visibility_rejects_unknown_or_mismatched_aka(self):
        with self.assertRaises(ModelBatchError):
            _batch_with_first_token(3, 2, action_v3.UNKNOWN_TILE_KIND, 1)
        with self.assertRaises(ModelBatchError):
            _batch_with_first_token(3, 2, 5, 1)       # aka 5m on 6m
        with self.assertRaises(ModelBatchError):
            _batch_with_first_token(3, 2, 22, 2)      # aka 5p colour on 5s
        with self.assertRaises(ModelBatchError):
            _batch_with_first_token(3, 2, 4, 3)       # aka 5s colour on 5m

    def test_visibility_rejects_opponent_known_ankan(self):
        with self.assertRaises(ModelBatchError):
            _batch_with_first_token(ANKAN_EVENT_TYPE, 2, 0, 0)
        with self.assertRaises(ModelBatchError):
            _batch_with_first_token(ANKAN_EVENT_TYPE, 3, 4, 1)

    def test_visibility_allows_public_tiles_and_own_private_tiles(self):
        allowed = [
            (TSUMO_EVENT_TYPE, 1, action_v3.UNKNOWN_TILE_KIND, 0),  # hidden opponent draw
            (TSUMO_EVENT_TYPE, 0, 4, 1),                            # own aka draw
            (ANKAN_EVENT_TYPE, 0, 0, 0),                            # own ankan, known
            (ANKAN_EVENT_TYPE, 3, action_v3.UNKNOWN_TILE_KIND, 0),  # hidden opponent ankan
            (DAIMINKAN_EVENT_TYPE, 2, 7, 0),                        # public called tile
            (KAKAN_EVENT_TYPE, 2, 9, 0),                            # public added tile
            (3, 2, action_v3.UNKNOWN_TILE_KIND, 0),                 # UNKNOWN with aka none
            (3, 2, 22, 3),                                          # well-formed aka 5s
        ]
        for event_type, actor, tile, aka in allowed:
            with self.subTest(event_type=event_type, actor=actor, tile=tile, aka=aka):
                batch = _batch_with_first_token(event_type, actor, tile, aka)
                batch.validate_visibility()

    def test_visibility_ignores_padding_slots(self):
        batch = make_batch(batch_size=1, seq_len=8, decisions_per_row=1, lengths=[4])
        fields = {}
        for name, value in (
            ("event_type_id", TSUMO_EVENT_TYPE),
            ("event_actor_rel", 1),
            ("event_tile_kind", 0),
            ("event_aka", 0),
        ):
            tensor = getattr(batch, name).clone()
            tensor[0, 6] = value  # padded slot, not an observable event
            fields[name] = tensor
        mutated = clone_batch(batch, **fields)
        self.assertFalse(bool(mutated.padding_mask[0, 6]))
        mutated.validate_visibility()

    def test_event_type_ids_match_frozen_schema(self):
        from riichiformer.schemas import event_v3

        self.assertEqual(TSUMO_EVENT_TYPE, event_v3.EVENT_TYPE_IDS["tsumo"])
        self.assertEqual(ANKAN_EVENT_TYPE, event_v3.EVENT_TYPE_IDS["ankan"])
        self.assertEqual(DAIMINKAN_EVENT_TYPE, event_v3.EVENT_TYPE_IDS["daiminkan"])
        self.assertEqual(KAKAN_EVENT_TYPE, event_v3.EVENT_TYPE_IDS["kakan"])

    def test_event_ranges_enforced(self):
        batch = make_batch(batch_size=1, seq_len=6, decisions_per_row=1)
        for field, vocab in zip(EVENT_TENSOR_FIELDS, EVENT_TENSOR_VOCABS):
            bad = getattr(batch, field).clone()
            bad[0, 0] = vocab
            with self.subTest(field=field):
                with self.assertRaises(ModelBatchError):
                    clone_batch(batch, **{field: bad})
        bad_mask = batch.padding_mask.clone()
        with self.assertRaises(ModelBatchError):
            clone_batch(batch, padding_mask=bad_mask.to(torch.long))

    def test_to_device_roundtrip(self):
        batch = make_batch(batch_size=2, seq_len=8, decisions_per_row=1)
        moved = batch.to("cpu")
        self.assertIsNot(moved, batch)
        self.assertEqual(moved.schema_version, batch.schema_version)
        self.assertTrue(torch.equal(moved.padding_mask, batch.padding_mask))
        self.assertTrue(torch.equal(moved.legal_mask, batch.legal_mask))

    def test_state_features_must_be_finite(self):
        batch = make_batch(batch_size=1, seq_len=6, decisions_per_row=1)
        bad = batch.state_features.clone()
        bad[0, 0] = float("inf")
        with self.assertRaises(ModelBatchError):
            clone_batch(batch, state_features=bad)
        with self.assertRaises(ModelBatchError):
            clone_batch(batch, schema_version=action_v3.SCHEMA_VERSION + "-tampered")


# ---------------------------------------------------------------------------
# policy behaviour
# ---------------------------------------------------------------------------


class TestPolicyForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Policy(tiny_config())
        self.model.eval()

    def test_shapes_and_output_contract(self):
        batch = make_batch(batch_size=2, seq_len=24, decisions_per_row=3)
        with torch.no_grad():
            output = self.model(batch)
        self.assertEqual(tuple(output.logits.shape), (6, 46))
        self.assertEqual(tuple(output.masked_logits.shape), (6, 46))
        self.assertEqual(tuple(output.logprobs.shape), (6, 46))
        self.assertEqual(tuple(output.selected_decision_offsets.shape), (6,))
        self.assertTrue(torch.equal(output.selected_decision_offsets,
                                    batch.flat_decision_offsets))
        output.validate()
        probs = output.probs
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(6), atol=1e-5))

    def test_illegal_actions_have_zero_probability(self):
        batch = make_batch(batch_size=2, seq_len=24, decisions_per_row=3)
        with torch.no_grad():
            output = self.model(batch)
        illegal = ~batch.legal_mask
        self.assertTrue(torch.isinf(output.masked_logits[illegal]).all())
        self.assertTrue((output.masked_logits[illegal] < 0).all())
        self.assertTrue((output.probs[illegal] == 0).all())
        self.assertTrue((output.logprobs[illegal] == float("-inf")).all())
        legal = batch.legal_mask
        self.assertTrue(torch.isfinite(output.logprobs[legal]).all())
        row_sums = output.logprobs.exp().sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_single_legal_action_probability_one(self):
        batch = make_batch(batch_size=1, seq_len=8, decisions_per_row=1, legal_size=1)
        with torch.no_grad():
            output = self.model(batch)
        self.assertTrue(torch.allclose(output.probs.sum(dim=-1), torch.ones(1)))
        self.assertTrue(torch.allclose(output.probs[batch.legal_mask], torch.ones(1)))

    def test_causal_prefix_invariance(self):
        batch = make_batch(batch_size=1, seq_len=80, decisions_per_row=2, seed=11)
        prefix = truncate_batch(batch, 60)
        self.assertEqual(prefix.n_decisions, 2)
        self.assertTrue(torch.equal(prefix.event_tile_kind,
                                    batch.event_tile_kind[:, :60]))
        with torch.no_grad():
            full_output = self.model(batch)
            prefix_output = self.model(prefix)
        full_rows = torch.tensor([0, 1])
        # The earlier decision logits are numerically equal within atol=1e-5; this is
        # deliberately not a bitwise promise (SDPA kernel reduction order may vary).
        self.assertTrue(torch.allclose(full_output.logits[full_rows],
                                       prefix_output.logits, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(full_output.selected_decision_offsets[full_rows],
                                    prefix_output.selected_decision_offsets))

    def test_different_pov_sequences_do_not_share_state(self):
        first = make_batch(batch_size=1, seq_len=32, decisions_per_row=2, seed=1)
        second = make_batch(batch_size=1, seq_len=32, decisions_per_row=2, seed=2)
        # Combine both POVs into one batch by concatenating row tensors.
        combined = clone_batch(
            first,
            **{
                name: torch.cat([getattr(first, name), getattr(second, name)], dim=0)
                for name in EVENT_TENSOR_FIELDS
            },
            padding_mask=torch.cat([first.padding_mask, second.padding_mask], dim=0),
            state_features=torch.cat([first.state_features, second.state_features]),
            decision_batch_index=torch.cat([first.decision_batch_index,
                                            second.decision_batch_index + 1]),
            decision_time_index=torch.cat([first.decision_time_index,
                                           second.decision_time_index]),
            legal_mask=torch.cat([first.legal_mask, second.legal_mask]),
            decision_phase=torch.cat([first.decision_phase, second.decision_phase]),
        )
        with torch.no_grad():
            combined_output = self.model(combined)
            first_output = self.model(first)
            second_output = self.model(second)
        n_first = first.n_decisions
        self.assertTrue(torch.allclose(combined_output.logits[:n_first],
                                       first_output.logits, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(combined_output.logits[n_first:],
                                       second_output.logits, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(first_output.logits, second_output.logits,
                                        atol=1e-6))

        # Mutating only POV 2's events/state must not change POV 1's decisions.
        mutated = clone_batch(combined)
        mutated.event_tsumogiri[1] = 1 - mutated.event_tsumogiri[1]
        mutated.state_features[n_first:] = torch.randn_like(
            mutated.state_features[n_first:])
        with torch.no_grad():
            mutated_output = self.model(mutated)
        self.assertTrue(torch.allclose(mutated_output.logits[:n_first],
                                       combined_output.logits[:n_first],
                                       atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(mutated_output.logits[n_first:],
                                        combined_output.logits[n_first:],
                                        atol=1e-6))

    def test_state_injection_is_decision_local_and_causal(self):
        batch = make_batch(batch_size=1, seq_len=32, decisions_per_row=2, seed=5)
        self.assertEqual(batch.n_decisions, 2)
        with torch.no_grad():
            base = self.model(batch)
        mutated = clone_batch(batch)
        mutated.state_features[1] = mutated.state_features[1] + 3.0
        with torch.no_grad():
            later_state = self.model(mutated)
        self.assertTrue(torch.allclose(later_state.logits[0], base.logits[0],
                                       atol=1e-6))
        self.assertFalse(torch.allclose(later_state.logits[1], base.logits[1],
                                        atol=1e-6))

        mutated_early = clone_batch(batch)
        mutated_early.state_features[0] = mutated_early.state_features[0] + 3.0
        with torch.no_grad():
            early_state = self.model(mutated_early)
        self.assertFalse(torch.allclose(early_state.logits[0], base.logits[0],
                                        atol=1e-6))

    def test_deterministic_eval_and_dropout_zero(self):
        batch = make_batch(batch_size=2, seq_len=16, decisions_per_row=2)
        with torch.no_grad():
            first = self.model(batch)
            second = self.model(batch)
        self.assertTrue(torch.equal(first.logits, second.logits))
        self.model.train()
        with torch.no_grad():
            train_first = self.model(batch)
            train_second = self.model(batch)
        self.assertTrue(torch.equal(train_first.logits, train_second.logits))
        self.assertEqual(self.model.config.dropout, 0.0)

    def test_batch_length_beyond_default_bucket_works(self):
        batch = make_batch(batch_size=1, seq_len=513, decisions_per_row=2)
        with torch.no_grad():
            output = self.model(batch)
        self.assertEqual(output.logits.shape[0], 2)
        self.assertEqual(batch.seq_len, 513)
        self.assertLess(int(output.selected_decision_offsets.max()), 513)

    def test_policy_refuses_overlong_sequence(self):
        model = Policy(tiny_config(max_positions=64))
        batch = make_batch(batch_size=1, seq_len=65, decisions_per_row=1)
        with self.assertRaises(TransformerShapeError):
            model(batch)

    def test_select_rows_keeps_only_that_pov(self):
        batch = make_batch(batch_size=2, seq_len=16, decisions_per_row=2, seed=4)
        selected = batch.select_rows(torch.tensor([1]))
        self.assertEqual(selected.batch_size, 1)
        self.assertEqual(selected.n_decisions, 2)
        self.assertTrue((selected.decision_batch_index == 0).all())
        with torch.no_grad():
            full = self.model(batch)
            subset = self.model(selected)
        rows = batch.decision_batch_index == 1
        self.assertTrue(torch.allclose(subset.logits, full.logits[rows],
                                       atol=1e-6, rtol=1e-6))

    def test_pr09_like_pov_batch_forwards(self):
        batch = make_pov_batch()
        # Opponent tsumo/ankan tokens in the fixture are UNKNOWN; public calls are known.
        self.assertEqual(int(batch.event_type_id[0, 4]), TSUMO_EVENT_TYPE)
        self.assertEqual(int(batch.event_tile_kind[0, 4]),
                         action_v3.UNKNOWN_TILE_KIND)
        self.assertEqual(int(batch.event_tile_kind[0, 10]), 7)
        self.assertEqual(int(batch.event_tile_kind[0, 9]),
                         action_v3.UNKNOWN_TILE_KIND)
        batch.validate_visibility()
        with torch.no_grad():
            output = self.model(batch)
        output.validate()
        self.assertEqual(output.logits.shape[0], batch.n_decisions)

    def test_forward_rechecks_visibility_after_in_place_mutation(self):
        batch = make_batch(batch_size=1, seq_len=8, decisions_per_row=1)
        batch.event_type_id[0, 0] = TSUMO_EVENT_TYPE
        batch.event_actor_rel[0, 0] = 1
        batch.event_tile_kind[0, 0] = 0
        batch.event_aka[0, 0] = 0
        with self.assertRaises(ModelBatchError):
            self.model(batch)

    def test_policy_rejects_wrong_input_type(self):
        with self.assertRaises(TypeError):
            self.model("not a batch")
        with self.assertRaises(TypeError):
            self.model({"event_type_id": torch.zeros(1, 1)})


class TestTraining(unittest.TestCase):
    def test_cpu_forward_backward_all_gradients_finite(self):
        torch.manual_seed(0)
        model = Policy(MVPConfig())
        batch = make_batch(batch_size=2, seq_len=32, decisions_per_row=2)
        labels = labels_from_batch(batch)
        model.train()
        output = model(batch)
        output.validate()
        loss = masked_cross_entropy(output, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        n_with_grad = 0
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, f"{name} has no gradient")
            self.assertTrue(torch.isfinite(parameter.grad).all(),
                            f"{name} gradient is not finite")
            n_with_grad += 1
        self.assertGreater(n_with_grad, 50)
        total_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                    for p in model.parameters()))
        self.assertTrue(torch.isfinite(total_norm))
        self.assertGreater(float(total_norm), 0.0)

    def test_cpu_optimizer_step_changes_weights(self):
        torch.manual_seed(0)
        model = Policy(tiny_config())
        batch = make_batch(batch_size=1, seq_len=16, decisions_per_row=1)
        labels = labels_from_batch(batch)
        before = model.policy_head.linear.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        loss = masked_cross_entropy(model(batch), labels)
        loss.backward()
        optimizer.step()
        after = model.policy_head.linear.weight.detach()
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.isfinite(after).all())

    def test_state_dict_strict_roundtrip(self):
        torch.manual_seed(1)
        first = Policy(tiny_config())
        second = Policy(tiny_config())
        result = second.load_state_dict(first.state_dict(), strict=True)
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])
        batch = make_batch(batch_size=1, seq_len=16, decisions_per_row=2)
        first.eval()
        second.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(first(batch).logits, second(batch).logits))
        broken = first.state_dict()
        broken.pop("policy_head.linear.weight")
        with self.assertRaises(RuntimeError):
            second.load_state_dict(broken, strict=True)
        extra = first.state_dict()
        extra["bogus.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            second.load_state_dict(extra, strict=True)

    def test_synthetic_overfit(self):
        torch.manual_seed(0)
        model = Policy(MVPConfig())
        batch = make_batch(batch_size=6, seq_len=64, decisions_per_row=2, seed=21)
        labels = labels_from_batch(batch, seed=99)
        self.assertGreater(len(set(labels.tolist())), 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
        model.train()
        with torch.no_grad():
            initial_output = model(batch)
            initial_loss = float(masked_cross_entropy(initial_output, labels))
        losses: list[float] = []
        for _ in range(OVERFIT_STEPS):
            output = model(batch)
            loss = masked_cross_entropy(output, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertTrue(all(math.isfinite(value) for value in losses))
        final_loss = losses[-1]
        with torch.no_grad():
            final_output = model(batch)
            accuracy = float((final_output.logits.argmax(dim=-1) == labels)
                             .to(torch.float32).mean())
        self.assertLessEqual(final_loss, 0.25 * initial_loss,
                             f"loss did not decrease enough: "
                             f"{initial_loss:.4f} -> {final_loss:.4f}")
        self.assertEqual(accuracy, 1.0,
                         f"argmax accuracy {accuracy:.3f} after {OVERFIT_STEPS} steps; "
                         f"losses={losses[:3]}...{losses[-3:]}")


class TestGPUSmoke(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_one_step_bf16_gpu(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        model = Policy(tiny_config(max_positions=128)).to("cuda")
        model.train()
        batch = make_batch(batch_size=2, seq_len=64, decisions_per_row=2,
                           device="cuda")
        labels = labels_from_batch(batch)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
            loss = masked_cross_entropy(output, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, f"{name} has no gradient")
            self.assertTrue(torch.isfinite(parameter.grad).all(),
                            f"{name} gradient is not finite")
        optimizer.step()
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self.assertGreater(peak_mib, 0.0)
        print(f"PR13 GPU smoke peak allocated VRAM: {peak_mib:.1f} MiB")
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# benchmark / overfit curve entry point (report evidence, not part of pytest)
# ---------------------------------------------------------------------------


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_calls(device: torch.device, fn, iters: int) -> float:
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _synchronize(device)
    return (time.perf_counter() - start) / iters


def _bench_forward(model: Policy, batch: ModelBatch, device: torch.device,
                   *, iters: int = 5) -> dict:
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(batch)
        seconds = _time_calls(device, lambda: model(batch), iters)
    tokens = batch.batch_size * batch.seq_len
    return {
        "seconds_per_forward": seconds,
        "tokens_per_second": tokens / seconds,
        "decisions_per_second": batch.n_decisions / seconds,
        "tokens": tokens,
        "decisions": batch.n_decisions,
    }


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _bench_train_step(model: Policy, batch: ModelBatch, labels: torch.Tensor,
                      device: torch.device, *, iters: int = 3) -> dict:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step() -> None:
        with _autocast(device):
            loss = masked_cross_entropy(model(batch), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with _autocast(device):
        loss = masked_cross_entropy(model(batch), labels)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)
    seconds = _time_calls(device, step, iters)
    tokens = batch.batch_size * batch.seq_len
    return {
        "seconds_per_train_step": seconds,
        "tokens_per_second": tokens / seconds,
        "decisions_per_second": batch.n_decisions / seconds,
    }


def _overfit_curve(*, steps: int = OVERFIT_STEPS, device: str = "cpu",
                   report_every: int = 20) -> dict:
    torch.manual_seed(0)
    model = Policy(MVPConfig()).to(device)
    batch = make_batch(batch_size=6, seq_len=64, decisions_per_row=2, seed=21,
                       device=device)
    labels = labels_from_batch(batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    curve = []
    with torch.no_grad():
        initial = float(masked_cross_entropy(model(batch), labels))
    for step in range(steps):
        loss = masked_cross_entropy(model(batch), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == steps - 1:
            with torch.no_grad():
                output = model(batch)
                accuracy = float((output.logits.argmax(dim=-1) == labels)
                                 .to(torch.float32).mean())
            curve.append({"step": step, "loss": float(loss.detach()),
                          "argmax_accuracy": accuracy})
    return {"initial_loss": initial, "steps": steps, "curve": curve}


def bench_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PR-13 MVP-S synthetic benchmark")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--decisions-per-row", type=int, default=4)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--overfit-steps", type=int, default=OVERFIT_STEPS)
    parser.add_argument("--skip-overfit", action="store_true")
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = Policy(MVPConfig()).to(device)
    summary: dict = {
        "device": str(device),
        "torch_version": torch.__version__,
        "device_name": (torch.cuda.get_device_name(0)
                        if device.type == "cuda" else "cpu"),
        "parameters": count_parameters(model),
        "parameter_breakdown": parameter_breakdown(model),
        "config": dataclasses.asdict(model.config),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batch = make_batch(batch_size=args.batch_size, seq_len=args.seq_len,
                       decisions_per_row=args.decisions_per_row, seed=3,
                       device=str(device), legal_size=8)
    labels = labels_from_batch(batch)
    summary["benchmark_batch"] = {
        "batch_size": batch.batch_size, "seq_len": batch.seq_len,
        "n_decisions": batch.n_decisions,
    }
    summary["forward"] = _bench_forward(model, batch, device, iters=args.iters)
    summary["train_step"] = _bench_train_step(model, batch, labels, device,
                                              iters=max(2, args.iters // 2))
    if device.type == "cuda":
        summary["peak_vram_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        summary["peak_vram_reserved_mib"] = (torch.cuda.max_memory_reserved()
                                             / (1024 ** 2))
        torch.cuda.empty_cache()
    if not args.skip_overfit:
        summary["overfit"] = _overfit_curve(steps=args.overfit_steps,
                                            device=str(device))
    print("PR13_BENCH_JSON " + json.dumps(summary, sort_keys=True))
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "parameter_breakdown"}, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bench":
        return bench_main(argv[1:])
    if argv and argv[0] in ("-h", "--help"):
        print("usage: rf_test_models.py [bench [--device cpu|cuda ...]]")
        return 0
    unittest.main(argv=[sys.argv[0]] + argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
