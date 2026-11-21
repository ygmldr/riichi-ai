"""门 1: 语义一致性 —— 日志里已存的 meta.q_values vs 引擎自推理的 Q。

还原规则(libriichi agent/mortal.rs gen_meta):
  q_values 是**压缩**数组: 按动作索引**升序**取所有 mask_bits 置位的动作;
  mask_bits 是 u64,bit i 置位表示动作 i 合法。
容差: 参照日志由 GPU + enable_amp 生成,故**禁止假设完全相等**;
  必须实测标定并写进 runs/exp/JIANGPU/gate1_tolerance.json。
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from eval_report import SEAT_OF_SPLIT  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TOLERANCE_PATH = REPO_ROOT / "runs/exp/JIANGPU/gate1_tolerance.json"
M3_GAMES_DIR = REPO_ROOT / "runs/exp/M3/games"
META_EVENT_TYPES = ("dahai", "chi", "pon", "daiminkan", "kakan", "ankan", "reach")


def seat_of_split(split: str) -> int:
    return int(SEAT_OF_SPLIT[split])


def decode_meta_q(meta):
    q_values = meta.get("q_values")
    mask_bits = meta.get("mask_bits")
    if not q_values or not isinstance(mask_bits, int):
        return {}
    legal = [i for i in range(64) if (mask_bits >> i) & 1]
    if len(legal) != len(q_values):
        raise ValueError(
            f"mask_bits 置位数 {len(legal)} 与 q_values 长度 {len(q_values)} 不一致")
    return {i: float(v) for i, v in zip(legal, q_values)}


def _meta_events(record, pov):
    """返回该席位**带 meta 的反应事件**里的 meta 字典。

    关键(实测,r2 修正): q_values/mask_bits 嵌在 `event["meta"]` **内层**,不在事件顶层。
    若把整个事件当 meta 传给 decode_meta_q,会永远得到 {} ⇒ 门 1 恒不过(这正是 r1→r2 的首个 BLOCKER)。
    """
    out = []
    for ev in record.events:
        meta = ev.get("meta")
        if (ev.get("type") in META_EVENT_TYPES
                and ev.get("actor") == pov
                and isinstance(meta, dict)
                and "q_values" in meta):
            out.append(meta)
    return out


def mask_bits_of(mask) -> int:
    """把 46 维布尔掩码重新压回 u64 位图(与日志 meta.mask_bits 同构)。"""
    import numpy as np
    bits = 0
    for i, ok in enumerate(np.asarray(mask).tolist()):
        if ok:
            bits |= (1 << i)
    return bits


def compare(record_path, checkpoint_path, max_decisions=200, device="cuda:0"):
    """把日志的 meta.q_values 与"同一检查点重推理"的 Q 逐决策比对。

    对齐方式(必须如此,**不能**假设一一对应):
      日志只在 dahai/chi/pon/kan/reach 上带 meta;pass/hora/ryukyoku/quick-eval 都不带。
      故某席位 **决策数 > meta 事件数**(实测 256 vs 193)。本函数用 meta.mask_bits 与
      决策的 mask 位图做**保序子序列对齐**,仅当位图相等时才算一对。
    """
    import numpy as np
    import torch

    from jiangpu.models import load_bundle
    from jiangpu.record import read_record
    from jiangpu.replay import iter_decisions

    record = read_record(record_path)
    pov = seat_of_split(record.split)
    decisions = iter_decisions(record, pov=pov)
    metas = _meta_events(record, pov)

    result = {
        "record": str(record_path),
        "checkpoint": str(checkpoint_path),
        "split": record.split,
        "pov": pov,
        "n_decisions": len(decisions),
        "n_meta_events": len(metas),
        "n_aligned": 0,
        "alignment_rate": None,
        "n_compared": 0,
        "argmax_match_rate": None,
        "max_abs_dq": None,
        "mean_abs_dq": None,
    }

    aligned = []
    cursor = 0
    for meta in metas:
        logged = decode_meta_q(meta)
        if not logged:
            continue
        bits = int(meta["mask_bits"])
        while cursor < len(decisions) and mask_bits_of(decisions[cursor].mask) != bits:
            cursor += 1
        if cursor >= len(decisions):
            break
        aligned.append((decisions[cursor], logged))
        cursor += 1

    result["n_aligned"] = len(aligned)
    result["alignment_rate"] = round(len(aligned) / len(metas), 6) if metas else None
    if not metas or len(aligned) != len(metas):
        result["reason"] = (f"仅 {len(aligned)}/{len(metas)} 个 meta 事件能在决策序列上对齐;"
                            "重放状态与生成日志时不一致(不静默通过)")
        return result

    bundle = load_bundle(checkpoint_path, device=device)
    n_match = 0
    deltas = []
    for dec, logged in aligned[:max_decisions]:
        obs = torch.as_tensor(dec.obs[None], dtype=torch.float32, device=device)
        mask = torch.as_tensor(dec.mask[None], dtype=torch.bool, device=device)
        with torch.inference_mode():
            phi = bundle.brain(obs, events=None)
            q, _v = bundle.dqn(phi, mask)
        q_row = q.detach().to("cpu").numpy()[0]
        legal = sorted(logged)
        mine = np.array([q_row[i] for i in legal], dtype=np.float64)
        theirs = np.array([logged[i] for i in legal], dtype=np.float64)
        deltas.append(float(np.max(np.abs(mine - theirs))))
        if int(mine.argmax()) == int(theirs.argmax()):
            n_match += 1
        result["n_compared"] += 1

    if deltas:
        result["argmax_match_rate"] = round(n_match / len(deltas), 6)
        result["max_abs_dq"] = round(max(deltas), 6)
        result["mean_abs_dq"] = round(sum(deltas) / len(deltas), 6)
    return result


def calibrate_and_record(record_path, checkpoint_path, max_decisions=200, device="cuda:0"):
    result = compare(record_path, checkpoint_path,
                     max_decisions=max_decisions, device=device)
    if result.get("max_abs_dq") is not None:
        result["tolerance"] = round(result["max_abs_dq"] * 10, 6)
        result["tolerance_rule"] = "10 × 观测 max|dQ|（首次标定,须人工复核后固化）"
    TOLERANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOLERANCE_PATH.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    return result
