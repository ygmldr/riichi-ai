#!/usr/bin/env python3
"""RiichiFormer V2 PR-05: build/verify the frozen split registry.

  # build both protocols from the PR-02 audit inventory
  .venv/bin/python scripts/rf_build_split.py build \
      --inventory artifacts/riichiformer/audit/source_audit.parquet \
      --out artifacts/riichiformer/split

  # verify train/val/select/confirm sets are disjoint (per protocol)
  .venv/bin/python scripts/rf_build_split.py assert-no-overlap \
      --manifest artifacts/riichiformer/split/split_manifest_L.parquet \
      --manifest artifacts/riichiformer/split/split_manifest_R.parquet

Outputs: split_manifest_L.parquet, split_manifest_R.parquet, split_summary.json and
docs/riichiformer/SPLIT_REGISTRY.md.  Rows are streamed; the registry hash uses an
order-independent XOR of per-game SHA-256 digests per split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from riichiformer.data import split as sp  # noqa: E402

DEFAULT_INVENTORY = "artifacts/riichiformer/audit/source_audit.parquet"
DEFAULT_OUT = "artifacts/riichiformer/split"
DEFAULT_DOC = "docs/riichiformer/SPLIT_REGISTRY.md"
BATCH = 200_000


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_inventory(path: Path, columns):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=BATCH, columns=columns):
        for row in batch.to_pylist():
            if row.get("kind", "leaf") != "leaf" or row.get("invalid_reason"):
                continue
            yield row


def _count_ids(path: Path, columns=("relpath",)) -> Counter:
    counts = Counter()
    for row in _iter_inventory(path, ["kind", "invalid_reason", *columns]):
        gid, _ = sp.game_id_from_relpath(row["relpath"])
        counts[gid] += 1
    return counts


def _split_digests(acc) -> dict:
    out = {}
    for split, record in sorted(acc.items()):
        out[split] = {
            "count": record["count"],
            "xor_sha256": hashlib.sha256(record["xor"].to_bytes(16, "big")).hexdigest(),
        }
    return out


def _registry_hash(protocol: str, cfg, counts: dict, digests: dict) -> str:
    payload = {
        "split_version": cfg.version,
        "protocol": protocol,
        "mode": cfg.mode,
        "hash_seed": cfg.hash_seed if cfg.mode == "hash_fallback" else None,
        "counts": counts,
        "split_digests": digests,
        "rights": sp._rights_status_map(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_one(inventory: Path, out: Path, protocol: str, mode: str, id_counts: Counter):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = sp.SplitConfig(protocol=protocol, mode=mode)
    path = out / f"split_manifest_{protocol}.parquet"
    schema = sp.manifest_schema()
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    accumulator = defaultdict(lambda: {"count": 0, "xor": 0})
    counts = Counter()
    year_split = Counter()
    blocked = 0
    train_allowed = 0
    dup_rows = 0

    def flush(rows):
        nonlocal blocked, train_allowed, dup_rows
        table_rows = []
        for row in rows:
            accumulator[row["split"]]["count"] += 1
            accumulator[row["split"]]["xor"] ^= int.from_bytes(
                hashlib.sha256(row["game_id"].encode("utf-8")).digest()[:16], "big"
            )
            counts[row["split"]] += 1
            year_split[(row["year"], row["split"])] += 1
            blocked += 1 if row["blocked_source"] else 0
            train_allowed += 1 if row["train_allowed"] else 0
            dup_rows += 1 if row["duplicate_group_size"] > 1 else 0
            table_rows.append(row)
        table = pa.Table.from_pydict(
            {name: [row.get(name) for row in table_rows] for name in schema.names},
            schema=schema,
        )
        writer.write_table(table)

    batch = []
    pending = []

    def process(pending_rows):
        rows = sp.build_split_rows(pending_rows, cfg)
        for row in rows:
            row["duplicate_group_size"] = id_counts.get(row["game_id"], 1)
        flush(rows)

    for item in _iter_inventory(inventory, ["root", "relpath", "year", "source_id", "kind", "invalid_reason"]):
        pending.append(item)
        if len(pending) >= BATCH:
            process(pending)
            pending = []
    if pending:
        process(pending)
    writer.close()

    digests = _split_digests(accumulator)
    registry_hash = _registry_hash(protocol, cfg, dict(sorted(counts.items())), digests)
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "protocol": protocol,
        "mode": mode,
        "manifest": str(path),
        "manifest_sha256": manifest_sha,
        "rows": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
        "blocked_source_rows": blocked,
        "train_allowed_rows": train_allowed,
        "duplicate_rows": dup_rows,
        "registry_hash": registry_hash,
        "split_digests": digests,
        "year_split": {f"{year}|{split}": n for (year, split), n in sorted(year_split.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))},
    }


def write_doc(path: Path, summaries: list[dict], inventory: Path):
    lines = [
        "# RiichiFormer V2 — SPLIT REGISTRY (PR-05)",
        "",
        f"- 生成时间：{_utc_now()}",
        f"- inventory：`{inventory}`（SHA-256 `{hashlib.sha256(inventory.read_bytes()).hexdigest()}`）",
        f"- schema：`{sp.SPLIT_VERSION}`；一行 = 一个 `game_id`（四 POV 与全部标签共享同一 split）",
        "- **所有来源当前 `unknown` → `train_allowed=false`；本文件不授权训练。**",
        "",
    ]
    for summary in summaries:
        lines += [
            f"## Protocol {summary['protocol']}（mode={summary['mode']}）",
            "",
            f"- rows：**{summary['rows']:,}**",
            f"- `registry_hash`：`{summary['registry_hash']}`",
            f"- `manifest_sha256`：`{summary['manifest_sha256']}`",
            f"- blocked_source rows：{summary['blocked_source_rows']:,}；train_allowed rows：{summary['train_allowed_rows']:,}",
            "",
            "| split | rows |",
            "|---|---:|",
        ]
        for split, count in summary["counts"].items():
            lines.append(f"| `{split}` | {count:,} |")
        lines.append("")
    lines += [
        "## 规则",
        "",
        "- **L（复现历史）**：train 2009–2022；val_holdout 2023；2024–2026 = `external_diagnostic`",
        "  （旧 EX1 已使用，**不是 pristine confirmation**）。",
        "- **R（近期训练）**：train 2009–2024；val 2025 H1；select 2025 H2；",
        "  2026 = `confirm_blocked_previously_used`（本地仅 1 月，且 2024–2026 已被 EX1 消费）。",
        "- month 来自 Tenhou 文件名 `YYYYMMDDHH…`；缺失 = `undated`，不猜。",
        "- 非 Tenhou 文件名 = `unknown_id`，fail closed；不进入 train。",
        "- `hash_fallback` 仅用于无日期语料，**不是时间外**；必须显式选择并记录。",
        "- 每个协议内 `game_id` 必须唯一 split；`--assert-no-overlap` 验证。",
        "",
        "> 待办：全量 canonical dedup（PR-03 `full_corpus=false`）尚未完成；本 registry 的重复检测",
        "> 目前依赖 Tenhou log id 唯一性，SHA 级重复覆盖需全量 hash 后才能宣称。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def command_build(args, repo: Path) -> int:
    inventory = _resolve(repo, args.inventory)
    out = _resolve(repo, args.out)
    doc = _resolve(repo, args.doc)
    if not inventory.is_file():
        print(f"rf_build_split: inventory not found: {inventory}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    protocols = [args.protocol] if args.protocol in sp.PROTOCOLS else list(sp.PROTOCOLS)
    print("counting duplicate game ids...", file=sys.stderr, flush=True)
    id_counts = _count_ids(inventory, columns=("relpath",))
    duplicates = sum(1 for count in id_counts.values() if count > 1)
    summaries = []
    for protocol in protocols:
        print(f"building protocol {protocol}...", file=sys.stderr, flush=True)
        summaries.append(build_one(inventory, out, protocol, args.mode, id_counts))
    summary = {
        "schema": "rf-split-summary/1",
        "generated_at_utc": _utc_now(),
        "inventory": str(inventory),
        "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "protocols": summaries,
        "duplicate_game_ids": duplicates,
        "training_blocked": all(s["train_allowed_rows"] == 0 for s in summaries),
        "note": "PR-03 full canonical index is still full_corpus=false; dedup by log id only",
    }
    (out / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_doc(doc, summaries, inventory)
    for s in summaries:
        print(f"  {s['protocol']}: rows={s['rows']} counts={s['counts']} hash={s['registry_hash'][:16]}…")
    print(f"  summary={out / 'split_summary.json'} doc={doc}")
    return 0


def command_assert(args, repo: Path) -> int:
    manifests = [_resolve(repo, value) for value in args.manifest]
    if not manifests:
        print("rf_build_split: --manifest is required", file=sys.stderr)
        return 2
    failed = False
    for path in manifests:
        try:
            report = sp.assert_manifest_no_overlap(path)
        except Exception as exc:  # noqa: BLE001
            print(f"ASSERT FAILED {path}: {exc}", file=sys.stderr)
            failed = True
            continue
        print(f"OK {path}: rows={report['rows']} counts={report['counts']} "
              f"duplicate_game_ids={report['duplicate_game_ids']}")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RiichiFormer V2 split registry (PR-05)")
    sub = ap.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--inventory", default=DEFAULT_INVENTORY)
    build.add_argument("--out", default=DEFAULT_OUT)
    build.add_argument("--doc", default=DEFAULT_DOC)
    build.add_argument("--protocol", choices=["L", "R", "both"], default="both")
    build.add_argument("--mode", choices=["calendar", "hash_fallback"], default="calendar")
    build.add_argument("--repo-root", default=str(REPO_ROOT))

    check = sub.add_parser("assert-no-overlap")
    check.add_argument("--manifest", nargs="+", required=True)
    check.add_argument("--repo-root", default=str(REPO_ROOT))

    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    if args.command == "build":
        return command_build(args, repo)
    return command_assert(args, repo)


if __name__ == "__main__":
    sys.exit(main())
