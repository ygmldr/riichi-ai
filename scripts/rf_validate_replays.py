#!/usr/bin/env python3
"""RiichiFormer V2 PR-04: replay validation against the native libriichi engine.

Reads a manifest (PR-02 audit parquet by default), selects a deterministic stratified
sample (or ``--all``), replays every game with ``GameplayLoader``, and checks that every
logged action is inside its 46-wide legal mask.  Failures are written with decision
index and reason; nothing is repaired or skipped silently.

  .venv/bin/python scripts/rf_validate_replays.py \
      --manifest artifacts/riichiformer/audit/source_audit.parquet \
      --out artifacts/riichiformer/replay --sample 1000 --workers 4

Exit codes: 0 = PASS (all sampled games legal), 1 = FAIL (illegal/load errors),
2 = setup error (manifest/libriichi missing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "vendor" / "mortal" / "target" / "release"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from riichiformer.data import canonicalize, dedup, replay  # noqa: E402

DEFAULT_MANIFEST = "artifacts/riichiformer/audit/source_audit.parquet"
DEFAULT_OUT = "artifacts/riichiformer/replay"
DEFAULT_DATA_ROOT = "datasets"


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_manifest(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path} (run PR-02 first)")
    table = pq.read_table(path)
    rows = table.to_pylist()
    return [row for row in rows if row.get("kind", "leaf") == "leaf"]


def read_game_text(path: Path) -> str:
    raw = path.read_bytes()
    compression = canonicalize.detect_compression_bytes(raw)
    payload = canonicalize.decode_log_bytes(raw, compression)
    return payload.decode("utf-8")


def validate_one(row: dict, data_root: Path) -> dict:
    path = Path(row.get("root") or data_root) / row["relpath"]
    base = {
        "relpath": row["relpath"],
        "path": str(path),
        "year": row.get("year"),
        "source_id": row.get("source_id"),
    }
    try:
        text = read_game_text(path)
    except (OSError, UnicodeDecodeError, canonicalize.CanonicalizeError) as exc:
        base.update({"ok": False, "error_kind": "read", "error": f"{type(exc).__name__}: {exc}",
                     "decisions": 0, "illegal": 0})
        return base
    result = replay.replay_game(text, oracle=False)
    summary = replay.summarize_replay(result)
    base.update(summary)
    return base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RiichiFormer V2 native replay validation (PR-04)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--all", action="store_true", help="validate every manifest row (slow)")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    args = ap.parse_args(argv)

    try:
        replay.load_libriichi()
    except replay.ReplayUnavailable as exc:
        print(f"rf_validate_replays: {exc}", file=sys.stderr)
        return 2

    repo = Path(args.repo_root).resolve()
    manifest_path = _resolve(repo, args.manifest)
    data_root = _resolve(repo, args.data_root)
    out_dir = _resolve(repo, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = load_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001
        print(f"rf_validate_replays: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    rows = [row for row in rows if not row.get("invalid_reason")]
    full = bool(args.all)
    if not full:
        rows = dedup.stratified_sample(rows, max_total=args.sample, seed=args.seed)
    if not rows:
        print("rf_validate_replays: no manifest rows selected", file=sys.stderr)
        return 2

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, result in enumerate(pool.map(lambda row: validate_one(row, data_root), rows), 1):
            results.append(result)
            if index % 200 == 0:
                print(f"  {index}/{len(rows)} games", file=sys.stderr, flush=True)

    invalid = [item for item in results if not item["ok"]]
    decisions = sum(item.get("decisions", 0) for item in results)
    illegal = sum(item.get("illegal", 0) for item in results)
    per_year = defaultdict(lambda: {"games": 0, "ok": 0, "decisions": 0, "illegal": 0})
    for item in results:
        row = per_year[str(item.get("year"))]
        row["games"] += 1
        row["decisions"] += item.get("decisions", 0)
        row["illegal"] += item.get("illegal", 0)
        row["ok"] += 1 if item["ok"] else 0
    error_kinds = Counter(item.get("error_kind") for item in invalid if item.get("error_kind"))
    status = "PASS" if not invalid else "FAIL"
    report = {
        "schema": "rf-replay-report/1",
        "status": status,
        "generated_at_utc": _utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "data_root": str(data_root),
        "full_corpus": full,
        "sample_size": None if full else args.sample,
        "sample_seed": None if full else args.seed,
        "workers": args.workers,
        "games_attempted": len(results),
        "games_ok": len(results) - len(invalid),
        "games_invalid": len(invalid),
        "decisions": decisions,
        "illegal_actions": illegal,
        "error_kinds": dict(error_kinds),
        "per_year": dict(sorted(per_year.items())),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "git_sha": _git_sha(repo),
        "limitations": [
            "sample report is not a full-corpus guarantee; use --all for that",
            "oracle=False (invisible obs not computed); actions/masks are engine-authored",
        ],
    }
    (out_dir / "replay_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with open(out_dir / "invalid_games.jsonl", "w", encoding="utf-8") as fh:
        for item in invalid:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"rf_validate_replays: status={status} games={len(results)} ok={report['games_ok']} "
          f"invalid={len(invalid)} decisions={decisions} illegal={illegal}")
    print(f"  error_kinds={report['error_kinds']}")
    print(f"  report={out_dir / 'replay_report.json'}")
    return 0 if status == "PASS" else 1


def _git_sha(repo: Path):
    try:
        import subprocess

        proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else None
    except OSError:
        return None


if __name__ == "__main__":
    sys.exit(main())
