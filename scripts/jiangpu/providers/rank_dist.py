"""3c: GRP 状态级顺位分布。

语义限制(spec §9 Q1): GRP 只吃**状态**,不吃动作 ⇒ 给出"该决策点的期望最终顺位分布",
不是"打出某张牌之后的分布"。动作级反事实需新建动作条件化模型,不在 M1/M2 范围。
"""
import numpy as np
import torch

PROVIDER_NAME = "rank_dist"
PROVIDER_VERSION = 1
UNIT = "probability"
KIND = "state_level"


def _grp_matrices(bundle, raw_text):
    from libriichi.dataset import Grp
    grp_ds = Grp.load_log(raw_text)
    feature = grp_ds.take_feature()          # (K,7) float64,消费型,只调一次
    seq = [torch.as_tensor(feature[:i + 1], dtype=torch.float64, device=bundle.device)
           for i in range(len(feature))]
    with torch.inference_mode():
        logits = bundle.grp(seq)
        matrix = bundle.grp.calc_matrix(logits)     # (K,4,4)
    return matrix.detach().to("cpu").double().numpy()


def _degrade(decisions, bundle, reason):
    return {d.index: {
        "provider": PROVIDER_NAME, "provider_version": PROVIDER_VERSION,
        "checkpoint_sha256": bundle.sha256, "unit": UNIT,
        "unavailable": True, "reason": reason,
    } for d in decisions}


def compute(bundle, record, decisions):
    try:
        matrices = _grp_matrices(bundle, record.raw_text)
    except Exception as exc:                      # noqa: BLE001 显式降级,不静默
        return _degrade(decisions, bundle, f"GRP 失败: {exc}")

    out = {}
    for dec in decisions:
        k = int(dec.at_kyoku)
        if k < 0 or k >= len(matrices):
            out[dec.index] = {
                "provider": PROVIDER_NAME, "provider_version": PROVIDER_VERSION,
                "checkpoint_sha256": bundle.sha256, "unit": UNIT,
                "unavailable": True,
                "reason": f"at_kyoku={k} 超出 GRP 局数 {len(matrices)}",
            }
            continue
        mat = matrices[k]
        norm = mat / mat.sum(axis=1, keepdims=True)
        ranks = np.arange(1, 5, dtype=np.float64)
        expected = float((norm * ranks).sum(axis=1)[dec.player_id])
        out[dec.index] = {
            "provider": PROVIDER_NAME,
            "provider_version": PROVIDER_VERSION,
            "checkpoint_sha256": bundle.sha256,
            "unit": UNIT,
            "kind": KIND,
            "matrix": [[round(float(v), 6) for v in row] for row in norm],
            "expected_rank": round(expected, 6),
            "subject_player": int(dec.player_id),
        }
    return out
