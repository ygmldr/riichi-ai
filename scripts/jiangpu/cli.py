"""单元 6: 编排 + 写盘 + 校验 + 退出码。

确定性(门 4)手段:
  - 文档内不写时间戳/主机名;路径按传入值原样记录
  - 浮点统一 round 到 6 位(contract.round_floats)
  - 序列化一律 sort_keys=True
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analyze_alllast import analyze_stream, decode_events  # noqa: E402  既有函数复用
from eval_report import SEAT_OF_SPLIT  # noqa: E402  <-- Task 8 必须在此导入(否则 build_document 直接 NameError)
from jiangpu import aggregate  # noqa: E402
from jiangpu import contract  # noqa: E402
from jiangpu.models import load_bundle  # noqa: E402
from jiangpu.providers import q_ev, rank_dist, regret, sp_table  # noqa: E402
from jiangpu.record import read_record  # noqa: E402
from jiangpu.replay import iter_decisions  # noqa: E402


def _source_tag(record_path):
    p = str(record_path)
    if "runs/1v3/" in p:
        return "frozen"
    if "runs/exp/" in p:
        return "experiment"
    return "unknown"


def build_document(record_path, checkpoint_path, grp_dir=None, device="cpu",
                   max_decisions=None, calibration=None):
    record = read_record(record_path)
    subject = int(SEAT_OF_SPLIT[record.split])          # challenger 席位
    # r2 修正: 只分析该席位的决策。否则 judgement_score/摘要 会把四家的决策混在一起,
    # 而 subject_player/actual_rank 只描述一家 ⇒ 语义不一致。
    decisions = iter_decisions(record, pov=subject)
    if max_decisions is not None:
        decisions = decisions[:int(max_decisions)]

    bundle = load_bundle(checkpoint_path, grp_dir=grp_dir, device=device)

    q_payloads = q_ev.compute(bundle, decisions, device=device, calibration=calibration)
    sp_payloads = sp_table.compute(decisions)
    rd_payloads = rank_dist.compute(bundle, record, decisions)

    game = analyze_stream(record.path, decode_events(record.path))
    final_scores = game["final_scores"]

    regret_payloads = regret.compute(decisions, q_payloads, calibration=calibration)
    agg = aggregate.build(decisions, q_payloads, regret_payloads, rd_payloads,
                          record, final_scores=final_scores, subject_player=subject)

    unavailable = []
    out_decisions = []
    for dec in decisions:
        metrics = {}
        for name, payloads in (("q_ev", q_payloads), ("sp_table", sp_payloads),
                              ("rank_dist", rd_payloads), ("regret", regret_payloads)):
            payload = payloads.get(dec.index)
            if payload is None:
                unavailable.append({"metric": f"{name}@decision{dec.index}",
                                    "reason": "provider 未返回该决策点"})
                continue
            if payload.get("unavailable"):
                unavailable.append({"metric": f"{name}@decision{dec.index}",
                                    "reason": str(payload.get("reason", "unknown"))})
                continue
            metrics[name] = payload
        out_decisions.append({
            "index": dec.index,
            "player_id": dec.player_id,
            "at_kyoku": dec.at_kyoku,
            "at_turn": dec.at_turn,
            "shanten": dec.shanten,
            "logged_action": dec.logged_action,
            "legal_actions": [int(i) for i, ok in enumerate(dec.mask.tolist()) if ok],
            "public": dec.public,
            "metrics": metrics,
        })

    doc = {
        "schema_version": contract.SCHEMA_VERSION,
        "engine_version": contract.ENGINE_VERSION,
        "checkpoint": {"path": str(checkpoint_path), "sha256": bundle.sha256,
                       "steps": bundle.steps},
        "record": {"path": str(record_path), "source": _source_tag(record_path),
                   "names": record.names, "seed": record.seed, "key": record.key,
                   "split": record.split},
        "decisions": out_decisions,
        "kyoku_summaries": agg["kyoku_summaries"],
        "game_summary": agg["game_summary"],
        "key_moments": agg["key_moments"],
        "unavailable": unavailable + agg["unavailable"],
    }
    return contract.round_floats(doc)


def main(argv=None):
    ap = argparse.ArgumentParser(description="讲谱分析引擎(离线,单份牌谱)")
    ap.add_argument("--record", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--grp-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    calibration = None
    if args.calibration:
        calibration = json.loads(pathlib.Path(args.calibration).read_text(encoding="utf-8"))

    try:
        doc = build_document(args.record, args.checkpoint, grp_dir=args.grp_dir,
                             device=args.device, max_decisions=args.max_decisions,
                             calibration=calibration)
    except Exception as exc:                        # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    errors = contract.validate(doc)
    if errors:
        for err in errors:
            print(f"SCHEMA: {err}", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"OK: {out} ({len(doc['decisions'])} decisions, "
          f"{len(doc['unavailable'])} unavailable)")
    return 0
