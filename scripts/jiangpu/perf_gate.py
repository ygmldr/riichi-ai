"""门 3: 性能门。一局约 270 决策,须在可接受的离线批量预算内。

预算沿用 tenhou-integration 计划的 P95 < 500 ms/决策
(libriichi 实测 p50 16.9-20.9 ms / p95 31-69 ms,CPU 单线程)。
"""
import time

P95_BUDGET_MS = 500.0


def _p95(values):
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[idx]


def run(record_path, checkpoint_path, device="cpu", limit=None):
    from jiangpu.models import load_bundle
    from jiangpu.providers import q_ev
    from jiangpu.record import read_record
    from jiangpu.replay import iter_decisions

    decisions = iter_decisions(read_record(record_path))
    if limit is not None:
        decisions = decisions[:int(limit)]
    bundle = load_bundle(checkpoint_path, device=device)

    per_decision = []
    t0 = time.perf_counter()
    for dec in decisions:
        t1 = time.perf_counter()
        q_ev.compute(bundle, [dec], device=device)
        per_decision.append((time.perf_counter() - t1) * 1000.0)
    total = time.perf_counter() - t0

    return {
        "n_decisions": len(decisions),
        "seconds": round(total, 6),
        "ms_per_decision": round(total * 1000.0 / max(1, len(decisions)), 6),
        "p95_ms": round(_p95(per_decision), 6) if per_decision else None,
    }
