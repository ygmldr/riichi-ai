#!/usr/bin/env python3
"""RiichiFormer V2 PR-03: build a canonical game index (stratified sample or full).

Reads the PR-02 audit inventory (``source_audit.parquet``), selects a deterministic
hash-stratified sample (default 1000, ``--sample``), canonicalises each game, groups
duplicates and writes:

  canonical_log_index.parquet   one row per canonicalised game (PR-03 schema)
  sample_1000.jsonl             human-inspectable list (paths + hashes + reasons)
  summary.json                  counts, seed, audit hash, full_corpus flag

``--all`` processes every audited leaf; that is a multi-hour job on 2.5 M local logs
and is **not** run by default.  A sample index is explicitly marked
``full_corpus=false`` and must not be used as the PR-05 split registry.

Safety: read-only on data, no network, no repair of invalid games.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data import canonicalize as canon  # noqa: E402
from riichiformer.data import dedup  # noqa: E402

DEFAULT_AUDIT = "artifacts/riichiformer/audit/source_audit.parquet"
DEFAULT_OUT = "artifacts/riichiformer/canonical"


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path)


def load_audit_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(f"audit parquet not found: {path} (run PR-02 first)")
    columns = ["root", "relpath", "year", "source_id", "kind", "compression", "invalid_reason"]
    table = pq.read_table(path, columns=columns)
    rows = table.to_pylist()
    return [row for row in rows if row.get("kind") == "leaf"]


def canonicalize_selected(rows: list[dict]):
    records = []
    for row in rows:
        path = Path(row["root"]) / row["relpath"]
        try:
            record = canon.canonicalize_file(
                path, relpath=row["relpath"], source_id=row.get("source_id"),
                year=row.get("year"),
            )
        except canon.CanonicalizeError as exc:
            raw = None
            try:
                raw = path.read_bytes()
            except OSError:
                pass
            record = canon.invalid_record(row["relpath"], raw, exc,
                                          source_id=row.get("source_id"),
                                          year=row.get("year"))
        records.append(record)
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RiichiFormer V2 canonical log index (PR-03)")
    ap.add_argument("--audit-parquet", default=DEFAULT_AUDIT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sample", type=int, default=1000,
                    help="stratified sample size (0 or --all for every audited leaf)")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--all", action="store_true", help="canonicalise every audited leaf (slow)")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    audit_path = _resolve(repo, args.audit_parquet)
    out_dir = _resolve(repo, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = load_audit_rows(audit_path)
    except Exception as exc:  # noqa: BLE001
        print(f"rf_build_canonical: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    rows = [row for row in rows if not row.get("invalid_reason")]
    full = bool(args.all or args.sample == 0)
    if not full:
        rows = dedup.stratified_sample(rows, max_total=args.sample, seed=args.seed)
    if not rows:
        print("rf_build_canonical: no audited leaf rows selected", file=sys.stderr)
        return 2

    records = canonicalize_selected(rows)
    index_rows = dedup.index_rows(records)
    index_path = out_dir / "canonical_log_index.parquet"
    dedup.write_index_parquet(index_rows, index_path)

    sample_path = out_dir / f"sample_{'all' if full else args.sample}.jsonl"
    with open(sample_path, "w", encoding="utf-8") as fh:
        for row in index_rows:
            fh.write(json.dumps({
                "game_id": row["game_id"],
                "relpath": row["relpath"],
                "year": row["year"],
                "compression": row.get("compression"),
                "canonical_sha256": row["canonical_sha256"],
                "duplicate_group": row["duplicate_group"],
                "duplicate_count": row["duplicate_count"],
                "is_representative": row["is_representative"],
                "collision_review": row["collision_review"],
                "invalid_reason": row["invalid_reason"],
            }, ensure_ascii=False, sort_keys=True) + "\n")

    summary = dedup.summarize_index(index_rows)
    summary.update({
        "audit_parquet": str(audit_path),
        "audit_parquet_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "out": str(out_dir),
        "index_parquet": str(index_path),
        "index_parquet_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "full_corpus": full,
        "sample_size": None if full else args.sample,
        "sample_seed": None if full else args.seed,
        "canonical_version": canon.CANONICAL_VERSION,
        "note": ("sample index: authoritative for canonicalisation/dedup tests only; "
                 "PR-05 needs --all"),
    })
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"rf_build_canonical: rows={summary['rows']} valid={summary['valid']} "
          f"invalid={summary['invalid']} groups={summary['duplicate_groups']} "
          f"full_corpus={full}")
    print(f"  index={index_path} sha256={summary['index_parquet_sha256']}")
    print(f"  summary={out_dir / 'summary.json'}")
    return 0 if summary["valid"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
