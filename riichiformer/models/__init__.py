"""PR-13 MVP-S synthetic model: event embedding, state MLP, local transformer, policy.

This package is self-contained (torch only plus the frozen schema action contract).
It does not import PR-11 ``data`` code, does not read datasets, and does not write
checkpoints.  ``ModelBatch`` is a replaceable synthetic-first interface.
"""

from .batch import (
    EVENT_TENSOR_FIELDS,
    EVENT_TENSOR_VOCABS,
    ModelBatch,
    ModelBatchError,
    MultiDecisionBatch,
    PHASE_INDEX,
    PHASE_LEGAL_INDEX_SETS,
    PHASE_ORDER,
)
from .event_embedding import EventEmbedding, EventEmbeddingError
from .local_transformer import (
    CausalSelfAttention,
    LocalTransformer,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerShapeError,
)
from .policy import (
    MVPConfig,
    MVPConfigError,
    PARAMETER_BUDGET_MAX,
    PARAMETER_BUDGET_MIN,
    Policy,
    PolicyHead,
    PolicyOutput,
    build_model,
    count_parameters,
    multi_decision_attention_mask,
    parameter_breakdown,
)
from .state_encoder import (
    STATE_FEATURE_DIM,
    STATE_FEATURE_LAYOUT,
    StateEncoder,
    StateFeatureError,
    StateMLP,
    StateSnapshot,
    state_features_from_snapshot,
    state_features_from_snapshots,
)

__all__ = [
    "CausalSelfAttention", "EVENT_TENSOR_FIELDS", "EVENT_TENSOR_VOCABS",
    "EventEmbedding", "EventEmbeddingError", "LocalTransformer", "MVPConfig",
    "MVPConfigError", "ModelBatch", "ModelBatchError", "MultiDecisionBatch",
    "PARAMETER_BUDGET_MAX",
    "PARAMETER_BUDGET_MIN", "PHASE_INDEX", "PHASE_LEGAL_INDEX_SETS", "PHASE_ORDER",
    "Policy", "PolicyHead", "PolicyOutput", "RMSNorm", "RotaryEmbedding",
    "STATE_FEATURE_DIM", "STATE_FEATURE_LAYOUT", "StateEncoder", "StateFeatureError",
    "StateMLP", "StateSnapshot", "SwiGLU", "TransformerBlock", "TransformerShapeError",
    "build_model", "count_parameters", "multi_decision_attention_mask",
    "parameter_breakdown",
    "state_features_from_snapshot", "state_features_from_snapshots",
]
