"""3a: Q 值相对分。

铁律(D1): Q 只作**相对排序**。我们自己的 A2b 审计测出 Q 局内排序 Spearman ρ=0.145、
τ=0.05 时 ECE 0.1897;Mortal 官方 FAQ 亦载"非最优动作的 Q 不可靠"。
故本 provider 只输出 q_rel(q_i - max_j q_j)与 confidence,绝不输出绝对期望值/概率。
confidence 阈值必须来自校准产物(D7);未提供时如实输出 "uncalibrated"。
"""
import numpy as np
import torch

PROVIDER_NAME = "q_ev"
PROVIDER_VERSION = 1
UNIT = "pt_ev"


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def compute(bundle, decisions, batch_size=64, device=None, calibration=None):
    device = device or bundle.device
    out = {}
    for chunk in _chunks(decisions, batch_size):
        obs = torch.as_tensor(np.stack([d.obs for d in chunk]), dtype=torch.float32, device=device)
        mask = torch.as_tensor(np.stack([d.mask for d in chunk]), dtype=torch.bool, device=device)
        with torch.inference_mode():
            phi = bundle.brain(obs, events=None)
            q, _v = bundle.dqn(phi, mask)
        q_np = q.detach().to("cpu").numpy()
        for row, dec in zip(q_np, chunk):
            legal = [int(i) for i, ok in enumerate(dec.mask.tolist()) if ok]
            vals = [float(row[i]) for i in legal]
            best = max(vals)
            q_rel = {str(i): round(v - best, 6) for i, v in zip(legal, vals)}
            gap = float(row[dec.logged_action] - best)
            if calibration and "gap_threshold" in calibration:
                conf = "low" if abs(gap) < float(calibration["gap_threshold"]) else "high"
            else:
                conf = "uncalibrated"
            out[dec.index] = {
                "provider": PROVIDER_NAME,
                "provider_version": PROVIDER_VERSION,
                "checkpoint_sha256": bundle.sha256,
                "unit": UNIT,
                "q_rel": q_rel,
                "logged_action_q_rel": round(gap, 6),
                "confidence": conf,
            }
    return out
