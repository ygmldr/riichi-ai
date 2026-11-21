#!/usr/bin/env python3
"""EX1 corpus sampler: deterministic three-way split + engine-readable materialization.

t3 of `.omo/plans/2026-09-23-explainability-metrics-and-validity.md`; contract frozen in
`scripts/exp_ex1_prereg.md` §2 (targets 400/400/800, floors 200/200/400, seed 13, split
rotates a/b/c/d) and §9 (shortfall: record actuals, never substitute training-year files).

  1. enumerate `datasets/extracted/{2024,2025,2026}/**/*.mjson` (the human holdout; any
     other year is refused);
  2. draw `calib_sub`/`select_sub`/`report_sub` as three sequential reservoir samples
     (seed 13, canonical `exp_a2_probe.reservoir_sample`) ⇒ pairwise disjoint;
  3. materialize each pick as `<seed>_<key>_<split>.json.gz` under
     `runs/exp/EX1/corpus/<subset>/`. **C-1**: `read_record` accepts only canonical names
     (`jiangpu/record.py:36-41`, `eval_report.py:53`) while the pool is `*.mjson`, so a
     direct feed raises `RecordError`. `(seed, key, split)` is deterministic (four
     consecutive picks share a seed, split rotates to balance seat coverage); plaintext
     sources are gzipped with `mtime=0`;
  4. smoke-read every copy with the engine's `read_record` (0 failures allowed) and write
     `runs/exp/EX1/corpus_index.json` (per-file source path/sha256 + target path/sha256,
     per-subset year distribution + shortfall, PREREG sha256).

Determinism: same seed + same pool ⇒ byte-identical index (no timestamps, no host data,
`sort_keys=True`, sorted pool). PREREG re-hashed at end-of-run; drift ⇒ FATAL (M8).

Usage: `source scripts/activate_env.sh && python scripts/exp_ex1_corpus.py`
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))        # flat `scripts/` imports
sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))  # libriichi (sampler import)
sys.path.insert(0, str(REPO / "vendor/mortal"))

from eval_report import FILE_RE, SEAT_OF_SPLIT  # noqa: E402
from exp_ex1_audit import assert_prereg_unchanged, load_prereg  # noqa: E402
from jiangpu.record import RecordError, read_record  # noqa: E402

PREREG_PATH: Final = REPO / "scripts/exp_ex1_prereg.md"
DATASETS_DIR: Final = REPO / "datasets/extracted"
OUT_DIR: Final = REPO / "runs/exp/EX1"

SEED: Final = 13
ALLOWED_YEARS: Final = (2024, 2025, 2026)
SUBSETS: Final = ("calib_sub", "select_sub", "report_sub")
TARGETS: Final = {"calib_sub": 400, "select_sub": 400, "report_sub": 800}
FLOORS: Final = {"calib_sub": 200, "select_sub": 200, "report_sub": 400}
ROLES: Final = {
    "calib_sub": "only subset on which any threshold/parameter is fitted (PREREG §2, MA-3/B1)",
    "select_sub": "head-to-head selection only (t16)",
    "report_sub": "final reporting; evaluated exactly once per head route (t16)",
}
# Distinct seed blocks per subset: (seed, key) groups hold 4 files, one per split.
SEED_BASE: Final = {"calib_sub": 900_000_000, "select_sub": 901_000_000, "report_sub": 902_000_000}
CANON_KEY: Final = 0x455831  # "EX1"
SPLITS: Final = ("a", "b", "c", "d")
GZIP_MAGIC: Final = b"\x1f\x8b"


@dataclass(frozen=True, slots=True, order=True)
class PoolFile:
    """One holdout source file; ordered by path so reservoir_sample's sort stays path-stable."""

    path: Path
    year: int


def _rel(path: Path) -> str:
    """Repo-relative display path (falls back to absolute for out-of-repo overrides)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def enumerate_pool(datasets_dir: Path, years: tuple[int, ...]) -> list[PoolFile]:
    """Sorted holdout pool for `years`; refuses any year outside the frozen holdout."""
    refused = [y for y in years if y not in ALLOWED_YEARS]
    if refused:
        raise SystemExit(
            f"FATAL: refusing non-holdout years {refused}; the frozen EX1 holdout is "
            f"{list(ALLOWED_YEARS)} (training years must never enter the corpus, PREREG §2)"
        )
    pool: list[PoolFile] = []
    for year in years:
        root = datasets_dir / str(year)
        found = sorted(glob.glob(str(root / "**" / "*.mjson"), recursive=True))
        if not found:
            raise SystemExit(f"FATAL: no .mjson found under {root}")
        pool.extend(PoolFile(Path(p), year) for p in found)
    pool.sort()
    return pool


def allocate_counts(n_pool: int) -> dict[str, int]:
    """Apportion min(n_pool, 1600) files: floors first, surplus proportional to targets.

    Integer-exact largest-remainder apportionment, ties by `SUBSETS` order; a pool below the
    floors is apportioned proportional to the targets and the shortfall is recorded (§9.6).
    """
    total_target = sum(TARGETS.values())
    total_floor = sum(FLOORS.values())
    n = min(n_pool, total_target)
    if n >= total_floor:
        base = dict(FLOORS)
        weights = {s: TARGETS[s] - FLOORS[s] for s in SUBSETS}
        surplus = n - total_floor
    else:
        base = dict.fromkeys(SUBSETS, 0)
        weights = dict(TARGETS)
        surplus = n
    total_w = sum(weights.values())
    quotas = {s: surplus * weights[s] for s in SUBSETS}
    counts = {s: quotas[s] // total_w for s in SUBSETS}
    leftover = surplus - sum(counts.values())
    order = sorted(SUBSETS, key=lambda s: (-(quotas[s] % total_w), SUBSETS.index(s)))
    for name in order[:leftover]:
        counts[name] += 1
    return {s: base[s] + counts[s] for s in SUBSETS}


def materialize(src: Path, dst: Path) -> tuple[str, str]:
    """Copy `src` to `dst` as gzip (plaintext ⇒ mtime=0); returns source/target sha256."""
    data = src.read_bytes()
    payload = data if data[:2] == GZIP_MAGIC else gzip.compress(data, compresslevel=6, mtime=0)
    dst.write_bytes(payload)
    return hashlib.sha256(data).hexdigest(), hashlib.sha256(payload).hexdigest()


def build_subset(name: str, picked: list[PoolFile], corpus_dir: Path) -> dict:
    """Materialize one subset under canonical names and smoke-read every copy."""
    subset_dir = corpus_dir / name
    subset_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(subset_dir.glob("*.json.gz")):
        stale.unlink()  # exact re-materialization: canonical payloads inside our own dir only
    files: list[dict] = []
    for i, src in enumerate(picked):
        seed = SEED_BASE[name] + i // 4
        split = SPLITS[i % 4]
        dst = subset_dir / f"{seed}_{CANON_KEY}_{split}.json.gz"
        src_sha, dst_sha = materialize(src.path, dst)
        files.append({
            "seed": seed,
            "key": CANON_KEY,
            "split": split,
            "year": src.year,
            "source_path": _rel(src.path),
            "source_sha256": src_sha,
            "target_path": _rel(dst),
            "target_sha256": dst_sha,
        })

    splits = sorted({f["split"] for f in files})
    if len(files) >= len(SPLITS) and splits != sorted(SPLITS):
        raise SystemExit(f"FATAL: {name}: split coverage {splits} != {sorted(SPLITS)}")
    for f in files:
        if FILE_RE.match(Path(f["target_path"]).name) is None:
            raise SystemExit(f"FATAL: {name}: target name violates FILE_RE: {f['target_path']}")
        if f["split"] not in SEAT_OF_SPLIT:
            raise SystemExit(f"FATAL: {name}: split {f['split']!r} has no SEAT_OF_SPLIT mapping")
        try:
            rec = read_record(str(REPO / f["target_path"]))
        except RecordError as exc:
            raise SystemExit(f"FATAL: {name}: materialized copy unreadable: {exc}") from exc
        if (rec.seed, rec.key, rec.split) != (f["seed"], f["key"], f["split"]):
            raise SystemExit(
                f"FATAL: {name}: identity mismatch for {f['target_path']}: "
                f"({rec.seed}, {rec.key}, {rec.split}) != ({f['seed']}, {f['key']}, {f['split']})"
            )

    year_counts = Counter(f["year"] for f in files)
    return {
        "role": ROLES[name],
        "target": TARGETS[name],
        "floor": FLOORS[name],
        "count": len(files),
        "shortfall": max(0, TARGETS[name] - len(files)),
        "floor_met": len(files) >= FLOORS[name],
        "year_distribution": {str(y): year_counts[y] for y in sorted(year_counts)},
        "splits": splits,
        "subset_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "files": files,
    }


def assert_invariants(subsets: dict[str, dict]) -> None:
    """Pairwise-disjoint subsets, all sources in the frozen holdout years (PREREG §2/§9)."""
    for i, a in enumerate(SUBSETS):
        for b in SUBSETS[i + 1:]:
            srcs = {f["source_path"] for f in subsets[a]["files"]}
            overlap = srcs & {f["source_path"] for f in subsets[b]["files"]}
            if overlap:
                raise SystemExit(f"FATAL: {a} ∩ {b} = {len(overlap)} files (must be empty)")
    allowed = {str(y) for y in ALLOWED_YEARS}
    for name in SUBSETS:
        outside = set(subsets[name]["year_distribution"]) - allowed
        if outside:
            raise SystemExit(f"FATAL: {name} contains non-holdout years {sorted(outside)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EX1 holdout corpus sampler (t3)")
    ap.add_argument("--seed", type=int, default=SEED, help="sampling seed (PREREG §2 pins 13)")
    ap.add_argument("--years", type=int, nargs="+", default=list(ALLOWED_YEARS),
                    help="holdout years (frozen: 2024 2025 2026)")
    ap.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR, help="read-only .mjson root")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="artifact dir (runs/exp/EX1)")
    ap.add_argument("--prereg", type=Path, default=PREREG_PATH, help="frozen pre-registration")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed < 0:
        raise SystemExit("FATAL: --seed must be a non-negative integer (PREREG §2 pins 13)")
    datasets_dir = Path(args.datasets_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    corpus_dir = out_dir / "corpus"
    prereg_path = Path(args.prereg).resolve()

    sha_start = load_prereg(prereg_path)

    # Canonical sampler from exp_a2_probe:238-251 (single implementation). That module
    # imports torch/libriichi, so pin the `mortal` namespace package before its own
    # directory joins sys.path (t1 notepad gotcha: otherwise `from event_data import ...`
    # explodes inside mortal/model.py).
    import mortal  # noqa: F401, PLC0415
    sys.path.append(str(REPO / "vendor/mortal/mortal"))
    from exp_a2_probe import reservoir_sample  # noqa: PLC0415

    pool = enumerate_pool(datasets_dir, tuple(args.years))
    allocation = allocate_counts(len(pool))
    print(f"pool: {len(pool):,} files over years {list(args.years)}; allocation {allocation}")

    remaining = list(pool)
    subsets: dict[str, dict] = {}
    for name in SUBSETS:
        picked = reservoir_sample(remaining, allocation[name], args.seed)
        picked_set = set(picked)
        remaining = [p for p in remaining if p not in picked_set]
        subsets[name] = build_subset(name, picked, corpus_dir)
        s = subsets[name]
        print(f"{name}: {s['count']} files (target {s['target']}, floor {s['floor']}, "
              f"shortfall {s['shortfall']}) years {s['year_distribution']}")

    assert_invariants(subsets)
    sha_end = assert_prereg_unchanged(prereg_path, sha_start)

    pool_years = Counter(p.year for p in pool)
    index = {
        "schema": "ex1-corpus-index/v1",
        "seed": args.seed,
        "source_years": list(args.years),
        "pool": {
            "n_files": len(pool),
            "by_year": {str(y): pool_years[y] for y in args.years},
        },
        "prereg": {
            "path": _rel(prereg_path),
            "sha256_start": sha_start,
            "sha256_end": sha_end,
            "immutable_during_run": sha_start == sha_end,
        },
        "subsets": subsets,
        "checks": {
            "pairwise_disjoint": True,
            "holdout_only": True,
            "read_record_checked": sum(s["count"] for s in subsets.values()),
            "read_record_failures": 0,
            "splits": list(SPLITS),
        },
    }
    out_path = out_dir / "corpus_index.json"
    out_path.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] wrote {_rel(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
