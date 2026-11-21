"""VNext runtime adapter: current-kyoku resets, prefix decisions, KAN_SELECTION.

The adapter owns one POV's event stream for one kyoku.  A ``start_kyoku`` (new
``kyoku_idx``) resets the event history, while global scores remain in the
``DecisionInput`` state as usual.  Two decisions with the same visible prefix
(for example ``SELF_TURN`` and ``KAN_SELECTION``) are two different decision
rows sharing the same K/V prefix.

This is a cacheless reference adapter: it rebuilds the stream tensors from the
visible prefix each call.  It is deliberately slower than a production KV-cache
agent, but it is the parity authority used by tests.  A future cache must
produce the same logits within the pre-registered tolerance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch

from ..schemas.action_v3 import ACTION_SPACE, ActionPhase
from ..schemas.decision_v3 import DecisionInput
from .contracts import VNextConfig, VNextModelBatch
from .model import VNextPolicy
from .observable import VNextEventStream, encode_stream


class VNextRuntimeError(RuntimeError):
    pass


class VNextPolicyAdapter:
    """Deterministic argmax policy over a VNext model and current-kyoku stream."""

    def __init__(self, model: VNextPolicy, *, match_id: str = "runtime",
                 pov: int = 0, name: str = "vnext") -> None:
        if not isinstance(model, VNextPolicy):
            raise VNextRuntimeError("model must be VNextPolicy")
        if pov not in (0, 1, 2, 3):
            raise VNextRuntimeError("pov must be 0..3")
        self.model = model.eval()
        self.match_id = str(match_id)
        self.pov = int(pov)
        self.name = str(name)
        self.stream: VNextEventStream | None = None
        self.decision_count = 0

    def reset_match(self) -> None:
        self.stream = None
        self.decision_count = 0

    def reset_kyoku(self) -> None:
        # The next decision's current kyoku will create a fresh BOS stream.
        self.stream = None

    def _ensure_stream(self, decision: DecisionInput) -> VNextEventStream:
        current = [event for event in decision.events if int(event.kyoku_idx) >= 0]
        if not current:
            raise VNextRuntimeError("decision has no current-kyoku event")
        kyoku_id = int(current[-1].kyoku_idx)
        if self.stream is None or self.stream.kyoku_id != kyoku_id:
            self.stream = VNextEventStream(
                match_id=self.match_id, kyoku_id=kyoku_id, pov=self.pov)
        return self.stream

    def observe(self, decision: DecisionInput, *,
                decision_id: str | None = None) -> int:
        if not isinstance(decision, DecisionInput):
            raise VNextRuntimeError("expected DecisionInput")
        stream = self._ensure_stream(decision)
        self.decision_count += 1
        stream.add_decision(
            decision,
            decision_id=decision_id or f"d{self.decision_count}",
            decision_order=self.decision_count - 1,
        )
        return self.decision_count - 1

    def act(self, decision: DecisionInput, *,
            decision_id: str | None = None) -> int:
        self.observe(decision, decision_id=decision_id)
        batch = self.stream.to_batch()
        with torch.inference_mode():
            output = self.model(batch)
        action = int(output.masked_logits[-1].argmax(dim=-1))
        if not 0 <= action < ACTION_SPACE or not bool(decision.legal_mask[action]):
            raise VNextRuntimeError(
                f"model returned illegal action {action} for phase "
                f"{decision.phase.value}")
        return action

    def act_batch(self, batch: VNextModelBatch) -> torch.Tensor:
        with torch.inference_mode():
            output = self.model(batch)
        return output.masked_logits.argmax(dim=-1)

    def select_kan(self, decision: DecisionInput, *, kan_mask: tuple[bool, ...],
                   decision_id: str | None = None) -> int:
        """Second policy call for a multi-kan opportunity (KAN_SELECTION)."""
        if decision.phase is not ActionPhase.KAN_SELECTION:
            raise VNextRuntimeError(
                "select_kan requires a DecisionInput already in KAN_SELECTION")
        if len(kan_mask) != ACTION_SPACE or not any(kan_mask):
            raise VNextRuntimeError("kan_mask must be a non-empty 46-mask")
        if any(kan_mask[i] for i in range(37, ACTION_SPACE)):
            raise VNextRuntimeError("KAN_SELECTION only permits tile actions 0..36")
        action = self.act(decision, decision_id=decision_id)
        if not kan_mask[action]:
            raise VNextRuntimeError(f"selected kan action {action} is not in kan_mask")
        return action

    def checkpoint_metadata(self, path: str | Path) -> dict:
        path = Path(path)
        if not path.is_file():
            raise VNextRuntimeError(f"checkpoint not found: {path}")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_vnext_checkpoint(path: str | Path, *,
                          expected_config_hash: str | None = None,
                          expected_contract_hash: str | None = None) -> VNextPolicy:
    """Load a VNext checkpoint payload; refuse an old MVP-S/checkpoint ABI."""
    import json
    path = Path(path)
    if not path.is_file():
        raise VNextRuntimeError(f"checkpoint not found: {path}")
    sidecar = Path(str(path) + ".json")
    if not sidecar.is_file():
        raise VNextRuntimeError("checkpoint sidecar is missing")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get("sha256") != actual:
        raise VNextRuntimeError("checkpoint file SHA-256 does not match sidecar")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "rf-checkpoint/1":
        raise VNextRuntimeError("not an rf-checkpoint/1 payload")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise VNextRuntimeError("checkpoint has no model_config")
    if "d_model" not in model_config or "cross_attn_layers" not in model_config:
        raise VNextRuntimeError("checkpoint is not a VNext model config")
    config = VNextConfig.from_mapping(dict(model_config))
    from ..train.checkpoint import config_hash
    actual_hash = config_hash(dict(model_config))
    if expected_config_hash is not None and actual_hash != expected_config_hash:
        raise VNextRuntimeError("checkpoint model_config_hash mismatch")
    stored_contract = (payload.get("hashes") or {}).get("vnext_contract_hash")
    if expected_contract_hash is not None \
            and stored_contract != expected_contract_hash:
        raise VNextRuntimeError("checkpoint VNext contract hash mismatch")
    payload_config_hash = payload.get("model_config_hash")
    if payload_config_hash is not None and payload_config_hash != actual_hash:
        raise VNextRuntimeError("checkpoint model_config_hash is inconsistent")
    model = VNextPolicy(config)
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise VNextRuntimeError("checkpoint has no model state")
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "VNextPolicyAdapter", "VNextRuntimeError", "load_vnext_checkpoint",
]
