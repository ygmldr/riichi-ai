#!/usr/bin/env python3
"""Read-only VNext Census over an authenticated rf-shard/3 shard.

Uses the shard's SQLite rows directly (no event decoding): phase/action/rank
counts and prefix-length quantiles are enough for the PR-11 pilot Census.
This is not a full-corpus Census and never trains or writes a shard.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.loader import ShardReader  # noqa: E402
from riichiformer.policy.config import load_policy_config  # noqa: E402


def pct(values: list[int], q: float) -> int | None:
    if not values:
        return None
    values.sort()
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    policy = load_policy_config(args.policy_config)
    reader = ShardReader(
        args.shard, purpose="audit",
        expected_policy_hash=policy.policy_hash,
        allowed_local_sources=policy.allowed_local_sources)
    try:
        conn = reader._conn
        try:
            conn.execute("PRAGMA temp_store=MEMORY")
            games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            decisions = conn.execute("SELECT COUNT(*) FROM inputs").fetchone()[0]
            supervised = conn.execute(
                "SELECT COUNT(*) FROM targets WHERE supervision_valid=1"
            ).fetchone()[0]
            povs = conn.execute(
                "SELECT DISTINCT pov FROM inputs ORDER BY pov").fetchall()
            phases = dict(conn.execute(
                "SELECT phase, COUNT(*) FROM inputs GROUP BY phase").fetchall())
            actions = dict(conn.execute(
                "SELECT expert_action, COUNT(*) FROM targets GROUP BY expert_action"
            ).fetchall())
            ranks = dict(conn.execute(
                "SELECT rank_final, COUNT(*) FROM targets GROUP BY rank_final"
            ).fetchall())
            lengths = [int(row[0]) + 1 for row in conn.execute(
                "SELECT before_event_idx FROM inputs")]
        finally:
            pass
        report = {
            "shard": str(Path(args.shard).resolve()),
            "shard_sha256": reader.manifest.get("shard_sha256"),
            "purpose": reader.metadata.purpose,
            "source_id": reader.metadata.source_id,
            "research_policy": reader.metadata.research_policy,
            "research_policy_hash": reader.metadata.research_policy_hash,
            "games": int(games),
            "povs": [int(row[0]) for row in povs],
            "decisions": int(decisions),
            "supervised": int(supervised),
            "phases": {str(k): int(v) for k, v in sorted(phases.items())},
            "actions": {str(k): int(v) for k, v in sorted(
                actions.items(), key=lambda kv: int(kv[0]) if kv[0] is not None
                else 999)},
            "ranks": {str(k): int(v) for k, v in sorted(
                ranks.items(), key=lambda kv: int(kv[0]) if kv[0] is not None
                else 999)},
            "prefix_tokens": {
                "n": len(lengths),
                "p50": pct(lengths, 0.50),
                "p90": pct(lengths, 0.90),
                "p99": pct(lengths, 0.99),
                "max": max(lengths) if lengths else None,
            },
            "note": "read-only pilot census; not full-corpus and not a strength result",
        }
        text = json.dumps(report, indent=2, sort_keys=True)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        return 0
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
