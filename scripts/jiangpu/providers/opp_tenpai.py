"""H1 provider: `opp_tenpai_prob` (three opponents) from public information only.

Frozen model: `runs/exp/EX1/headA/headA.pt` (Route A, H1 head). The provider loads the checkpoint
through the CLI's `--head-checkpoint` seam; without it the metric is explicitly unavailable (never
a hardcoded path, never a silent zero).

NO-REVEAL: features come from public observation planes via
`jiangpu.opp_tenpai_features` (t12 context + t13 safety). No oracle label, no opponent concealed
hand, no true tile order. `basis="public_info_predictor"`, `no_hidden_info=true`.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from jiangpu import opp_tenpai_features as features

PROVIDER_NAME = "opp_tenpai"
PROVIDER_VERSION = 1
UNIT = "probability"
NO_HIDDEN_INFO = True
BASIS = "public_info_predictor"
N_OPPS = 3
EVAL_BATCH = 4096

_MODEL_CACHE: dict[tuple, tuple] = {}


class OppTenpaiHead(nn.Module):
    """Trunk + H1 head only; the other t14 heads are ignored at load time."""

    def __init__(self, in_dim: int = features.FEATURE_DIM, hidden: int = 256, layers: int = 2):
        super().__init__()
        trunk: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            trunk += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.trunk = nn.Sequential(*trunk)
        self.opp_tenpai = nn.Linear(d, N_OPPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.opp_tenpai(self.trunk(x))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta(checkpoint_sha256):
    return {
        "provider": PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "unit": UNIT,
        "no_hidden_info": NO_HIDDEN_INFO,
        "basis": BASIS,
    }


def _unavailable(reason, checkpoint_sha256=None):
    return {**_meta(checkpoint_sha256), "unavailable": True, "reason": str(reason)}


def _load(checkpoint_path):
    path = Path(checkpoint_path)
    key = (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    spec = ckpt.get("feature_spec") or {}
    if int(spec.get("dim", -1)) != features.FEATURE_DIM:
        raise ValueError(f"checkpoint feature dim {spec.get('dim')} != {features.FEATURE_DIM}")
    model = OppTenpaiHead()
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        raise ValueError(f"missing H1/trunk weights: {sorted(missing)}")
    allowed_extra = ("deal_in.", "wait_membership.", "exp_occ.", "exp_val.")
    bad_extra = [k for k in unexpected if not k.startswith(allowed_extra)]
    if bad_extra:
        raise ValueError(f"unexpected checkpoint keys: {sorted(bad_extra)[:5]}")
    mu = np.asarray(ckpt["mu"], dtype=np.float32)
    sd = np.asarray(ckpt["sd"], dtype=np.float32)
    if mu.shape != (features.FEATURE_DIM,) or sd.shape != (features.FEATURE_DIM,):
        raise ValueError(f"checkpoint normalization shapes {mu.shape}/{sd.shape}")
    model.eval()
    value = (model, mu, sd, _sha256(path))
    _MODEL_CACHE[key] = value
    return value


def compute(decisions, checkpoint_path=None, device="cpu", batch=EVAL_BATCH):
    """Return {decision.index: payload}; explicit `unavailable` when the head cannot run."""
    decisions = list(decisions)
    if not decisions:
        return {}
    if not checkpoint_path:
        reason = "未提供 --head-checkpoint:H1 头不可用(不硬编码路径、不填 0)"
        return {dec.index: _unavailable(reason) for dec in decisions}
    try:
        model, mu, sd, sha = _load(checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        reason = f"H1 检查点加载失败: {type(exc).__name__}: {exc}"
        return {dec.index: _unavailable(reason) for dec in decisions}
    try:
        x = features.features_for_decisions(decisions)
    except Exception as exc:  # noqa: BLE001
        reason = f"H1 公开特征构造失败: {type(exc).__name__}: {exc}"
        return {dec.index: _unavailable(reason, sha) for dec in decisions}

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    model = model.to(dev)
    mt = torch.from_numpy(mu).to(dev)
    st = torch.from_numpy(sd).to(dev)
    probs: list[np.ndarray] = []
    with torch.inference_mode():
        for lo in range(0, x.shape[0], batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[lo:lo + batch])).to(dev).float()
            xb = (xb - mt) / st
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs, axis=0) if probs else np.zeros((0, N_OPPS), dtype=np.float32)
    if p.shape != (len(decisions), N_OPPS):
        reason = f"H1 输出形状 {p.shape} != ({len(decisions)}, {N_OPPS})"
        return {dec.index: _unavailable(reason, sha) for dec in decisions}
    out = {}
    meta = _meta(sha)
    for dec, row in zip(decisions, p):
        out[dec.index] = {**meta, "opp_tenpai_prob": [round(float(v), 6) for v in row]}
    return out
