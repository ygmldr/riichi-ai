"""Independent 24-way GRP (PR-06), kept separate from the Actor.

The GRP predicts the joint final-rank permutation of four seats from the public
kyoku-start score state.  It is a two-layer GRU over ``[grand_kyoku, honba,
kyotaku, score0..score3]`` (7 inputs), followed by a 24-way linear head.  The
permutation table and PT tables are imported from
:mod:`riichiformer.vnext.contracts`; this module never imports the Actor model.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contracts import (
    GRP_PERMUTATIONS,
    PT_TABLE_5D,
    PT_TABLE_5D_ID,
    PT_TABLES,
    ContractError,
    GRPBoundary,
    expected_pt_from_rank_probs,
    permutation_index,
    phi_from_marginals,
    rank_marginals,
)


@dataclass(frozen=True)
class GRPConfig:
    input_size: int = 7
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.0
    permutations: int = 24

    def __post_init__(self) -> None:
        if self.input_size != 7:
            raise ContractError("GRP input contract is exactly 7 features")
        if self.layers < 1 or self.hidden < 1:
            raise ContractError("GRP hidden/layers must be positive")
        if self.permutations != 24:
            raise ContractError("GRP must output 24 joint permutations")
        if not 0.0 <= self.dropout < 1.0:
            raise ContractError("GRP dropout must be in [0,1)")


class GRP24(nn.Module):
    """Public-score GRU -> 24 joint final-rank logits."""

    def __init__(self, config: GRPConfig | None = None) -> None:
        super().__init__()
        self.config = config or GRPConfig()
        c = self.config
        self.gru = nn.GRU(
            input_size=c.input_size, hidden_size=c.hidden,
            num_layers=c.layers, batch_first=True,
            dropout=c.dropout if c.layers > 1 else 0.0)
        self.head = nn.Linear(c.hidden, c.permutations)

    def forward(self, boundaries: Tensor,
                lengths: Tensor | None = None) -> Tensor:
        """Return ``[B, 24]`` logits; ``boundaries`` is ``[B,T,7]``."""
        if boundaries.dim() == 2:
            boundaries = boundaries.unsqueeze(1)
        if boundaries.dim() != 3 or boundaries.size(-1) != 7:
            raise ContractError("GRP boundaries must be [B,T,7]")
        if not torch.isfinite(boundaries).all():
            raise ContractError("GRP boundaries contain NaN/Inf")
        if lengths is None:
            out, _ = self.gru(boundaries)
            last = out[:, -1]
        else:
            if lengths.shape != (boundaries.size(0),):
                raise ContractError("GRP lengths must be [B]")
            if bool((lengths < 1).any()) or bool((lengths > boundaries.size(1)).any()):
                raise ContractError("GRP lengths outside sequence")
            packed = nn.utils.rnn.pack_padded_sequence(
                boundaries, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=boundaries.size(1))
            last = out[torch.arange(out.size(0)), lengths - 1]
        return self.head(last)

    def probability(self, boundaries: Tensor,
                    lengths: Tensor | None = None) -> Tensor:
        return torch.softmax(self.forward(boundaries, lengths), dim=-1)

    def marginals(self, boundaries: Tensor,
                  lengths: Tensor | None = None) -> Tensor:
        return rank_marginals(self.probability(boundaries, lengths))

    def phi(self, boundaries: Tensor, *, pt_table_id: str = PT_TABLE_5D_ID,
            lengths: Tensor | None = None) -> Tensor:
        if pt_table_id not in PT_TABLES:
            raise ContractError(f"unknown pt_table_id {pt_table_id!r}")
        return phi_from_marginals(
            self.marginals(boundaries, lengths), PT_TABLES[pt_table_id])


def boundaries_tensor(boundaries: Sequence[GRPBoundary]) -> Tensor:
    if not boundaries:
        raise ContractError("need at least one GRPBoundary")
    return torch.tensor([boundary.input_vector for boundary in boundaries],
                        dtype=torch.float32)


def grp_loss(logits: Tensor, rank_by_abs_seat: Tensor,
             weights: Tensor | None = None) -> Tensor:
    """Cross entropy over the 24 permutations, with optional per-row weights."""
    if logits.dim() != 2 or logits.size(1) != 24:
        raise ContractError("GRP logits must be [N,24]")
    if rank_by_abs_seat.dim() != 2 or rank_by_abs_seat.size(1) != 4:
        raise ContractError("rank_by_abs_seat must be [N,4]")
    n = logits.size(0)
    if rank_by_abs_seat.size(0) != n:
        raise ContractError("rank/logit length mismatch")
    targets = torch.tensor(
        [permutation_index(row.tolist()) for row in rank_by_abs_seat],
        dtype=torch.long, device=logits.device)
    losses = F.cross_entropy(logits, targets, reduction="none")
    if weights is None:
        return losses.mean()
    if weights.shape != (n,):
        raise ContractError("weights must be [N]")
    total = weights.sum()
    if float(total) <= 0:
        raise ContractError("weights must sum to > 0")
    return (losses * weights).sum() / total


def permute_input_scores(scores: Sequence[int], order: Sequence[int]) -> tuple[int, ...]:
    """Reorder absolute-seat scores for a seat permutation ``new[i]=old[order[i]]``."""
    if len(scores) != 4 or sorted(int(v) for v in order) != [0, 1, 2, 3]:
        raise ContractError("scores must have 4 entries and order a permutation")
    return tuple(int(scores[int(order[i])]) for i in range(4))


def _permuted_index(old_perm: Sequence[int], order: Sequence[int]) -> int:
    new_perm = tuple(int(old_perm[int(order[i])]) for i in range(4))
    return permutation_index(new_perm)


def permute_prob_perm(prob_perm: Tensor, order: Sequence[int]) -> Tensor:
    """Apply a seat permutation to a ``[...,24]`` joint probability vector."""
    if prob_perm.size(-1) != 24:
        raise ContractError("prob_perm last dimension must be 24")
    if sorted(int(v) for v in order) != [0, 1, 2, 3]:
        raise ContractError("order must be a permutation of 0..3")
    out = torch.empty_like(prob_perm)
    for old_index, old_perm in enumerate(GRP_PERMUTATIONS):
        out[..., _permuted_index(old_perm, order)] = prob_perm[..., old_index]
    return out


def permutation_consistency_error(model: GRP24, boundaries: Tensor,
                                  order: Sequence[int]) -> float:
    """Diagnostic max difference after score/seat reordering.

    A plain GRU is not permutation-equivariant, so this is not a pass/fail
    gate; use it only to inspect drift before adding symmetry training.
    """
    if boundaries.dim() != 3:
        raise ContractError("boundaries must be [B,T,7]")
    if sorted(int(v) for v in order) != [0, 1, 2, 3]:
        raise ContractError("order must be a permutation of 0..3")
    permuted = boundaries.clone()
    permuted[..., 3:] = boundaries[..., 3:][..., list(order)]
    with torch.no_grad():
        original_prob = model.probability(boundaries)
        permuted_prob = model.probability(permuted)
        # A permutation of the seats is not a permutation of the GRU input order
        # in general; the expected joint distribution transforms by the same rank
        # relabeling.  Compare the transformed original distribution.
        expected = permute_prob_perm(original_prob, order)
        return float((permuted_prob - expected).abs().max())


@dataclass(frozen=True)
class CalibrationReport:
    nll: float
    brier: float
    ece: float
    count: int


def calibration_report(prob_perm: Tensor, rank_by_abs_seat: Tensor,
                       bins: int = 10) -> CalibrationReport:
    """Multiclass NLL/Brier/ECE over the 24 permutation labels."""
    if prob_perm.dim() != 2 or prob_perm.size(1) != 24:
        raise ContractError("prob_perm must be [N,24]")
    if rank_by_abs_seat.shape != (prob_perm.size(0), 4):
        raise ContractError("rank labels must be [N,4]")
    probs = prob_perm.float()
    if bool((probs < 0).any()) or not bool(
            torch.allclose(probs.sum(-1), torch.ones(probs.size(0)),
                           atol=1e-5, rtol=1e-5)):
        raise ContractError("prob_perm rows must be probability distributions")
    targets = [permutation_index(row.tolist()) for row in rank_by_abs_seat]
    idx = torch.tensor(targets, device=probs.device)
    p_true = probs[torch.arange(probs.size(0)), idx]
    nll = float(-torch.log(p_true.clamp_min(1e-12)).mean())
    one_hot = F.one_hot(idx, num_classes=24).to(probs.dtype)
    brier = float(((probs - one_hot) ** 2).sum(-1).mean())
    confidence, prediction = probs.max(-1)
    correct = (prediction == idx).to(probs.dtype)
    ece = 0.0
    for lower in torch.linspace(0, 1, bins + 1)[:-1].tolist():
        upper = lower + 1.0 / bins
        selected = (confidence >= lower) & (
            confidence < upper if upper < 1.0 else confidence <= upper)
        if bool(selected.any()):
            gap = abs(float(confidence[selected].mean()) - float(correct[selected].mean()))
            ece += gap * float(selected.float().mean())
    return CalibrationReport(nll=nll, brier=brier, ece=ece, count=probs.size(0))


def grp_state_sha(model: GRP24) -> str:
    """Hash parameter names, shapes and bytes; stable across save/load."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def grp_config_hash(config: GRPConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_grp_checkpoint(path: str | Path, model: GRP24, *,
                        normalizer_hash: str, rules_hash: str,
                        permutation_map_hash: str) -> dict:
    """Save an independent GRP artifact; no Actor state is written."""
    if not isinstance(model, GRP24):
        raise ContractError("model must be GRP24")
    for name, value in (("normalizer_hash", normalizer_hash),
                        ("rules_hash", rules_hash),
                        ("permutation_map_hash", permutation_map_hash)):
        if not isinstance(value, str) or len(value) != 64:
            raise ContractError(f"{name} must be a 64-char sha256")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "rf-grp-checkpoint/1",
        "config": asdict(model.config),
        "config_hash": grp_config_hash(model.config),
        "state": model.state_dict(),
        "state_sha256": grp_state_sha(model),
        "normalizer_hash": normalizer_hash,
        "rules_hash": rules_hash,
        "permutation_map_hash": permutation_map_hash,
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    os.replace(tmp, path)
    manifest = {key: value for key, value in payload.items() if key != "state"}
    manifest["sha256"] = digest
    Path(str(path) + ".json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_grp_checkpoint(path: str | Path) -> tuple[GRP24, dict]:
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"GRP checkpoint not found: {path}")
    sidecar = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if sidecar.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ContractError("GRP checkpoint file SHA-256 does not match sidecar")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "rf-grp-checkpoint/1":
        raise ContractError("unknown GRP checkpoint format")
    config = GRPConfig(**payload["config"])
    if grp_config_hash(config) != payload.get("config_hash"):
        raise ContractError("GRP config hash mismatch")
    model = GRP24(config)
    model.load_state_dict(payload["state"], strict=True)
    if grp_state_sha(model) != payload.get("state_sha256"):
        raise ContractError("GRP state hash mismatch")
    return model, sidecar


__all__ = [
    "CalibrationReport", "GRP24", "GRPConfig", "boundaries_tensor",
    "calibration_report", "grp_config_hash", "grp_loss", "grp_state_sha",
    "load_grp_checkpoint", "permutation_consistency_error",
    "permute_input_scores", "permute_prob_perm", "save_grp_checkpoint",
]
