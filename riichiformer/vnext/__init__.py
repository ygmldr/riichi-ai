"""RiichiFormer VNext package (PR-02+).

Public contracts live in :mod:`riichiformer.vnext.contracts`; model, encoder,
GRP and runtime modules build on them.  The old MVP-S modules are intentionally
not imported here.
"""

from .contracts import (  # noqa: F401
    GRPBoundary,
    GRP_PERMUTATIONS,
    MODEL_ARCH_ID,
    PT_TABLE_5D,
    PT_TABLE_7D,
    RLDecisionRow,
    VNextConfig,
    VNextDecisionIndex,
    VNextModelBatch,
    VNextPolicyOutput,
    VNextTargets,
    VNextVisibleState,
    contract_hash,
    contract_manifest,
)

__all__ = [
    "GRPBoundary", "GRP_PERMUTATIONS", "MODEL_ARCH_ID", "PT_TABLE_5D",
    "PT_TABLE_7D", "RLDecisionRow", "VNextConfig", "VNextDecisionIndex",
    "VNextModelBatch", "VNextPolicyOutput", "VNextTargets", "VNextVisibleState",
    "contract_hash", "contract_manifest",
]
