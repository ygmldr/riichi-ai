"""3f: 悪手度(regret)。

定义: delta_rel = q_rel(实际打出的动作) ≤ 0,越负越差。
置信(承 D1 / spec §8 风险): Q 的局内排序能力弱(ρ=0.145),故必须带置信标记;
阈值来自校准产物,缺失时输出 "uncalibrated",**不得**默认声称可信。
"""
PROVIDER_NAME = "regret"
PROVIDER_VERSION = 1
UNIT = "pt_ev"


def compute(decisions, q_payloads, calibration=None):
    # D7: 阈值**必须**来自校准产物;缺失时不得用任何硬编码兜底值产出确定性判定
    threshold = None
    confidence = "uncalibrated"
    if calibration and "regret_blunder_threshold" in calibration:
        threshold = float(calibration["regret_blunder_threshold"])
        confidence = "calibrated"

    out = {}
    for dec in decisions:
        payload = q_payloads.get(dec.index)
        if payload is None or payload.get("unavailable"):
            out[dec.index] = {
                "provider": PROVIDER_NAME, "provider_version": PROVIDER_VERSION,
                "checkpoint_sha256": None, "unit": UNIT,
                "unavailable": True, "reason": "缺少 q_ev",
            }
            continue
        delta = float(payload.get("logged_action_q_rel", 0.0))
        out[dec.index] = {
            "provider": PROVIDER_NAME,
            "provider_version": PROVIDER_VERSION,
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "unit": UNIT,
            "delta_rel": round(min(delta, 0.0), 6),
            # D7: 未校准 ⇒ is_blunder 为 None(不产出确定性判定),而非硬编码阈值
            "is_blunder": (None if threshold is None else bool(delta <= threshold)),
            "confidence": confidence,
        }
    return out
