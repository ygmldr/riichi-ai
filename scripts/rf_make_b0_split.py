#!/usr/bin/env python3
"""Create a deterministic, reproducible B0 sample from an H-02 split-v2 manifest.

The builder's ``--limit`` takes the first representative train rows in game-id order,
which is biased toward old logs.  B0 needs a fixed sample that is independent of that
ordering and reproducible from a seed.  This script selects ``--sample-size``
representative, non-quarantine train rows by

    sha256(f"{sample_seed}:{game_id}")

and writes a parquet with the **same schema** as the source split manifest.  The
sampled manifest can be passed to ``scripts/rf_build_shards.py`` unchanged; its SHA-256
is recorded as the checkpoint/shard split provenance.

This is a sampling utility, not a training-rights decision: the source rows keep their
original rights_status/train_allowed fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = (REPO_ROOT / "artifacts/riichiformer/h02/split_full"
                 / "split_manifest_L_v2.parquet")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, game_id: str) -> bytes:
    return hashlib.sha256(f"{int(seed)}:{game_id}".encode("utf-8")).digest()


def select_sample(table: pa.Table, *, size: int, seed: int) -> pa.Table:
    mask = pc.and_(
        pc.equal(table["is_representative"], pa.scalar(True)),
        pc.and_(
            pc.equal(table["quarantine"], pa.scalar(False)),
            pc.equal(table["split"], pa.scalar("train")),
        ),
    )
    filtered = table.filter(mask)
    if filtered.num_rows < size:
        raise SystemExit(
            f"only {filtered.num_rows} eligible train rows, requested {size}")
    ids = filtered["game_id"].to_pylist()
    order = sorted(range(len(ids)), key=lambda i: stable_key(seed, ids[i]))[:size]
    return filtered.take(pa.array(order, type=pa.int64()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=20261121)
    args = parser.parse_args(argv)
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")
    source = Path(args.split_manifest)
    if not source.is_file():
        raise SystemExit(f"split manifest not found: {source}")
    table = pq.read_table(source)
    sample = select_sample(table, size=args.sample_size, seed=args.sample_seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sample, out)
    months = sample["month"].to_pylist()
    years = sample["year"].to_pylist()
    sources = sample["source_id"].to_pylist()
    summary = {
        "format": "rf-b0-sample/1",
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": file_sha256(source),
        "sample_manifest": str(out.resolve()),
        "sample_manifest_sha256": file_sha256(out),
        "sample_seed": int(args.sample_seed),
        "sample_size": int(args.sample_size),
        "selected_rows": sample.num_rows,
        "source_rows": table.num_rows,
        "eligible_train_rows": int(
            pc.sum(pc.and_(
                pc.equal(table["is_representative"], pa.scalar(True)),
                pc.and_(pc.equal(table["quarantine"], pa.scalar(False)),
                        pc.equal(table["split"], pa.scalar("train"))),
            )).as_py()),
        "sources": {name: sources.count(name) for name in sorted(set(sources))},
        "years": {str(year): years.count(year) for year in sorted(set(years))},
        "months": {str(month): months.count(month) for month in sorted(set(months))},
        "warning": (
            "fixed B0 sample only; rows keep their original rights_status/train_allowed "
            "and must not be treated as a third-party licence"
        ),
    }
    summary_path = Path(str(out) + ".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
