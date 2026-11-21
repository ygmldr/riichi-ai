"""MVP-S policy network for PR-13 (synthetic-only forward/backward deliverable).

``Policy`` ties the four owned modules together:

``event ids -> EventEmbedding -> [B,L,d]``
``state features -> StateMLP -> [N,d]`` injected at each decision token
``-> LocalTransformer (6 Pre-RMSNorm causal blocks, RoPE, SwiGLU, SDPA)``
``-> gather decision tokens -> Linear(d, 46) -> masked logits -> log-softmax``

The objective/loss and the trainer belong to PR-14; this module only exposes the
forward contract.  ``forward(batch)`` accepts **one** argument, a
:class:`~riichiformer.models.batch.ModelBatch`; there is deliberately no way to pass
a label, target, rank or oracle tensor into the network.

Illegal actions get ``masked_logits = -inf`` and therefore exact zero probability.
``logprobs`` is ``log_softmax(masked_logits)`` over the frozen 46-action space.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from ..schemas.action_v3 import ACTION_SPACE
from .batch import ModelBatch, ModelBatchError
from .event_embedding import EventEmbedding
from .local_transformer import DEFAULT_MAX_POSITIONS, LocalTransformer
from .state_encoder import STATE_FEATURE_DIM, StateMLP

#: MVP-S weight-count guardrails (measured 5.22M; see PR13_MODEL_REPORT.md).
PARAMETER_BUDGET_MIN: int = 4_000_000
PARAMETER_BUDGET_MAX: int = 6_500_000


class MVPConfigError(ValueError):
    """Invalid or drifted MVP-S configuration."""


@dataclass(frozen=True)
class MVPConfig:
    """Architecture config mirrored by ``configs/riichiformer/mvp_s.yaml``."""

    name: str = "mvp_s"
    architecture: str = "local_causal"
    hidden: int = 256
    layers: int = 6
    heads: int = 4
    ffn: int = 704
    state_encoder: str = "mlp"
    state_hidden: int = 512
    policy_head: str = "linear46"
    memory: bool = False
    dropout: float = 0.0
    rope_base: float = 10_000.0
    rmsnorm_eps: float = 1.0e-6
    max_positions: int = DEFAULT_MAX_POSITIONS

    def __post_init__(self) -> None:
        if self.architecture != "local_causal":
            raise MVPConfigError(
                f"PR-13 supports architecture 'local_causal', got {self.architecture!r}"
            )
        if self.state_encoder != "mlp":
            raise MVPConfigError(f"state_encoder must be 'mlp', got {self.state_encoder!r}")
        if self.policy_head != "linear46":
            raise MVPConfigError(
                f"policy_head must be 'linear46', got {self.policy_head!r}"
            )
        if self.memory is not False:
            raise MVPConfigError(
                "MVP-S has no cross-kyoku memory; memory must be false"
            )
        for name in ("hidden", "layers", "heads", "ffn", "state_hidden", "max_positions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MVPConfigError(f"{name} must be a positive int, got {value!r}")
        if self.hidden % self.heads != 0:
            raise MVPConfigError(
                f"hidden {self.hidden} is not divisible by heads {self.heads}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise MVPConfigError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.rope_base <= 1.0:
            raise MVPConfigError(f"rope_base must be > 1, got {self.rope_base}")
        if self.rmsnorm_eps <= 0.0:
            raise MVPConfigError(f"rmsnorm_eps must be > 0, got {self.rmsnorm_eps}")
        if not isinstance(self.name, str) or not self.name:
            raise MVPConfigError(f"name must be a non-empty string, got {self.name!r}")

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> "MVPConfig":
        if not isinstance(mapping, Mapping):
            raise MVPConfigError(
                f"config section must be a mapping, got {type(mapping).__name__}"
            )
        allowed = set(cls.field_names())
        unknown = set(mapping) - allowed
        if unknown:
            raise MVPConfigError(f"unknown MVP-S config keys: {sorted(unknown)}")
        return cls(**dict(mapping))

    @classmethod
    def from_yaml(cls, path: Path | str, section: str = "model") -> "MVPConfig":
        try:
            import yaml  # local import: the library itself does not need PyYAML
        except ImportError as exc:  # pragma: no cover - PyYAML present in the env
            raise MVPConfigError("PyYAML is required to read the MVP-S config") from exc
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise MVPConfigError(f"{path} is not a YAML mapping")
        if section not in document:
            raise MVPConfigError(f"{path} has no {section!r} section")
        return cls.from_mapping(document[section])


@dataclass
class PolicyOutput:
    """Forward result; illegal actions keep ``-inf`` logits / zero probability."""

    logits: Tensor                 # [N, 46] raw head output
    masked_logits: Tensor          # [N, 46] with -inf on illegal actions
    logprobs: Tensor               # [N, 46] log_softmax(masked_logits)
    selected_decision_offsets: Tensor | None = None  # [N] flat B*L token offsets
    legal_mask: Tensor | None = None                 # [N, 46] bool

    @property
    def probs(self) -> Tensor:
        return self.logprobs.exp()

    def validate(self, *, rtol: float = 1e-5, atol: float = 1e-5) -> None:
        for name in ("logits", "masked_logits", "logprobs"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.dim() != 2 \
                    or value.size(1) != ACTION_SPACE:
                raise ModelBatchError(f"{name} must be [N, {ACTION_SPACE}]")
        n_decisions = self.logits.size(0)
        if n_decisions <= 0:
            raise ModelBatchError("PolicyOutput must contain at least one decision")
        if self.masked_logits.shape != self.logits.shape \
                or self.logprobs.shape != self.logits.shape:
            raise ModelBatchError("PolicyOutput tensors must share one shape")
        if self.legal_mask is None:
            raise ModelBatchError("PolicyOutput.validate needs legal_mask")
        if self.legal_mask.shape != self.logits.shape:
            raise ModelBatchError("legal_mask shape must match logits")
        if not self.legal_mask.any(dim=1).all():
            raise ModelBatchError("every decision needs at least one legal action")
        illegal = ~self.legal_mask
        if not bool(torch.isinf(self.masked_logits[illegal]).all()):
            raise ModelBatchError("illegal masked_logits must be -inf")
        if not bool((self.masked_logits[illegal] < 0).all()):
            raise ModelBatchError("illegal masked_logits must be -inf (negative)")
        if not bool(torch.isfinite(self.logprobs[self.legal_mask]).all()):
            raise ModelBatchError("legal logprobs contain NaN/Inf")
        if not bool(torch.isfinite(self.masked_logits[self.legal_mask]).all()):
            raise ModelBatchError("legal masked_logits contain NaN/Inf")
        if not bool((self.logprobs[illegal] < -1e30).all()):
            raise ModelBatchError("illegal logprobs must be -inf")
        row_logsumexp = torch.logsumexp(self.logprobs, dim=-1)
        if not torch.allclose(row_logsumexp, torch.zeros_like(row_logsumexp),
                              rtol=rtol, atol=atol):
            raise ModelBatchError("logprobs do not normalize over legal actions")
        if self.selected_decision_offsets is not None:
            if self.selected_decision_offsets.shape != (n_decisions,):
                raise ModelBatchError("selected_decision_offsets must be [N]")


class PolicyHead(nn.Module):
    """Linear 46-action head (``Linear(256, 46)`` for the default config)."""

    def __init__(self, d_model: int = 256, action_space: int = ACTION_SPACE) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.action_space = int(action_space)
        self.linear = nn.Linear(d_model, action_space, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.dim() != 2 or hidden.size(1) != self.d_model:
            raise ModelBatchError(
                f"policy head input must be [N, {self.d_model}], got "
                f"{tuple(hidden.shape)}"
            )
        return self.linear(hidden)


class Policy(nn.Module):
    """Complete MVP-S network; forward consumes only a :class:`ModelBatch`."""

    def __init__(self, config: MVPConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else MVPConfig()
        if not isinstance(self.config, MVPConfig):
            raise TypeError(f"config must be MVPConfig, got {type(self.config)}")
        config = self.config
        self.event_embedding = EventEmbedding(config.hidden, dropout=config.dropout)
        self.state_encoder = StateMLP(hidden=config.state_hidden, d_model=config.hidden)
        self.transformer = LocalTransformer(
            d_model=config.hidden,
            n_layers=config.layers,
            n_heads=config.heads,
            ffn_hidden=config.ffn,
            dropout=config.dropout,
            max_positions=config.max_positions,
            rope_base=config.rope_base,
            rmsnorm_eps=config.rmsnorm_eps,
        )
        self.policy_head = PolicyHead(config.hidden, ACTION_SPACE)

    # ------------------------------------------------------------------ forward
    def forward(self, batch: ModelBatch) -> PolicyOutput:
        if not isinstance(batch, ModelBatch):
            raise TypeError(
                f"Policy.forward accepts only ModelBatch, got {type(batch).__name__}"
            )
        # Re-check visibility even if the batch was valid at construction: tensors are
        # mutable, and future encoders may fill them incrementally.
        batch.validate_visibility()
        parameter = self.policy_head.linear.weight
        if batch.padding_mask.device != parameter.device:
            raise ModelBatchError(
                f"batch device {batch.padding_mask.device} != model device "
                f"{parameter.device}; move the batch/model explicitly"
            )
        batch_size, seq_len = batch.batch_size, batch.seq_len

        events = self.event_embedding(
            batch.event_type_id,
            batch.event_actor_rel,
            batch.event_tile_kind,
            batch.event_aka,
            batch.event_tsumogiri,
            batch.event_phase,
            batch.event_kyoku_pos,
            batch.padding_mask,
            check_ids=False,  # ModelBatch already validated every id / range
        )
        # ModelBatch already validated shape and finiteness of state_features.
        state = self.state_encoder(batch.state_features, validate=False)  # [N, d]

        # Inject the decision-time state into exactly that decision's token.
        injection = events.new_zeros(batch_size * seq_len, self.config.hidden).index_add(
            0, batch.flat_decision_offsets, state.to(dtype=events.dtype)
        )
        hidden = events + injection.view(batch_size, seq_len, self.config.hidden)
        hidden = self.transformer(hidden, batch.padding_mask)

        flat = hidden.reshape(batch_size * seq_len, self.config.hidden)
        decision_hidden = flat.index_select(0, batch.flat_decision_offsets)
        logits = self.policy_head(decision_hidden)
        legal = batch.legal_mask
        masked_logits = logits.masked_fill(~legal, float("-inf"))
        logprobs = torch.log_softmax(masked_logits, dim=-1)
        return PolicyOutput(
            logits=logits,
            masked_logits=masked_logits,
            logprobs=logprobs,
            selected_decision_offsets=batch.flat_decision_offsets,
            legal_mask=legal,
        )


def build_model(config: MVPConfig | None = None) -> Policy:
    """Construct an MVP-S :class:`Policy` (random init; no checkpoint loading)."""
    return Policy(config)


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    """Per-top-level-module trainable parameter counts, plus ``total``."""
    breakdown = {
        name: count_parameters(child)
        for name, child in model.named_children()
    }
    breakdown["total"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return breakdown


__all__ = [
    "ACTION_SPACE", "MVPConfig", "MVPConfigError", "PARAMETER_BUDGET_MAX",
    "PARAMETER_BUDGET_MIN", "Policy", "PolicyHead", "PolicyOutput", "STATE_FEATURE_DIM",
    "build_model", "count_parameters", "parameter_breakdown",
]
