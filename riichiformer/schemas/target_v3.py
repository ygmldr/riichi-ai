"""Label-side types for schema v3 (PR-08).

This module is intentionally separate from :mod:`riichiformer.schemas.decision_v3`: the
runtime/online inference path imports only the input module, so a forward pass cannot even
see ``DecisionTarget``.  Targets are validated against the input's legal mask *after* both
have been built (PR-10 owns that construction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .action_v3 import ACTION_SPACE, ActionSchemaError, SCHEMA_VERSION, decode_action


class TargetSchemaError(ValueError):
    """A DecisionTarget violation."""


class TargetNotInMask(TargetSchemaError):
    """A supervised expert action is outside the decision's legal mask."""


@dataclass(frozen=True)
class DecisionTarget:
    """Supervision label for one ``DecisionInput``; never a forward input field."""

    expert_action: int | None
    supervision_valid: bool
    censor_reason: str | None = None
    rank_final: int | None = None
    oracle_labels: Mapping[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise TargetSchemaError(
                f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}"
            )
        if not isinstance(self.supervision_valid, bool):
            raise TargetSchemaError("supervision_valid must be bool")
        if self.expert_action is not None:
            if (not isinstance(self.expert_action, int)
                    or isinstance(self.expert_action, bool)
                    or not 0 <= self.expert_action < ACTION_SPACE):
                raise TargetSchemaError(
                    f"expert_action {self.expert_action!r} outside 0..{ACTION_SPACE - 1}"
                )
        if self.supervision_valid:
            if self.expert_action is None:
                raise TargetSchemaError("supervised target needs an expert_action")
            if self.censor_reason is not None:
                raise TargetSchemaError("supervised target cannot carry censor_reason")
        elif self.censor_reason is None:
            raise TargetSchemaError("unsupervised target needs a censor_reason")
        if self.censor_reason is not None and not isinstance(self.censor_reason, str):
            raise TargetSchemaError("censor_reason must be str or None")
        if self.rank_final is not None:
            if (not isinstance(self.rank_final, int)
                    or isinstance(self.rank_final, bool)
                    or not 1 <= self.rank_final <= 4):
                raise TargetSchemaError("rank_final must be 1..4 or None")
        if self.oracle_labels is not None and not isinstance(self.oracle_labels, Mapping):
            raise TargetSchemaError("oracle_labels must be a mapping or None")

    def validate_against_input(self, decision_input) -> None:
        """Check the label against the already-built input (mask + phase)."""
        if not self.supervision_valid:
            return
        action = self.expert_action
        if action is None:  # pragma: no cover - __post_init__ already enforces
            raise TargetSchemaError("supervised target has no expert_action")
        mask = getattr(decision_input, "legal_mask", None)
        phase = getattr(decision_input, "phase", None)
        if mask is None or phase is None:
            raise TargetSchemaError("decision_input is missing legal_mask/phase")
        if not (0 <= action < len(mask)) or not mask[action]:
            raise TargetNotInMask(
                f"expert_action {action} is not legal in the decision mask"
            )
        try:
            decode_action(action, phase)
        except ActionSchemaError as exc:
            raise TargetNotInMask(str(exc)) from exc

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "expert_action": self.expert_action,
            "supervision_valid": self.supervision_valid,
            "censor_reason": self.censor_reason,
            "rank_final": self.rank_final,
            "oracle_labels": dict(self.oracle_labels) if self.oracle_labels else None,
        }


__all__ = ["DecisionTarget", "TargetNotInMask", "TargetSchemaError"]
