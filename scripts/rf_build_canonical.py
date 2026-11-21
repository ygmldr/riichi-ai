#!/usr/bin/env python3
"""RiichiFormer H-02: sharded, checkpointed canonical index builder.

Replaces the PR-03 in-memory ``--all`` path.  The PR-02 audit inventory is streamed in
Parquet batches; selected leaves are canonicalised in deterministic shards (year +
``--chunk-rows`` rows) and committed to one SQLite checkpoint per shard.  A resume
re-processes only the shard that was in flight when the process died; re-running a
completed build rewrites the same artifacts byte-for-byte.

Outputs (under ``--out``, default ``artifacts/riichiformer/h02/canonical``):

  canonical_index.sqlite        durable checkpoint (shards/games/components + meta)
  canonical_log_index.parquet  one row per selected leaf (PR-03 columns + H-02 fields)
  duplicate_components.parquet one row per component with >1 file
  quarantine.jsonl             every quarantined row with its reason
  sample_<N>.jsonl             sample mode only (human-inspectable)
  manifest.json                deterministic manifest (counts, hashes, versions)
  summary.json                 byte-identical copy of the manifest (back-compat name)
  run_log.jsonl                append-only volatile per-shard timing/RSS log

Safety: input audit/data are opened read-only; ``--out`` may not point inside
``datasets/`` or any existing PR-02/03/05 artifact directory.  Nothing is pushed.

Examples::

  # 10k stratified fixture (no full run)
  .venv/bin/python scripts/rf_build_canonical.py --sample 10000 \
      --out artifacts/riichiformer/h02/bench_10k

  # full 2.51M run (only after Lead approves the exclusive-IO window)
  .venv/bin/python scripts/rf_build_canonical.py --all \
      --out artifacts/riichiformer/h02/canonical
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data import canonicalize as canon  # noqa: E402
from riichiformer.data import dedup  # noqa: E402

DEFAULT_AUDIT = "artifacts/riichiformer/audit/source_audit.parquet"
DEFAULT_OUT = "artifacts/riichiformer/h02/canonical"
AUDIT_COLUMNS = [
    "root", "relpath", "year", "source_id", "kind", "compression", "bytes",
    "sha256", "invalid_reason",
]
ARTIFACT_NAMES = [
    "canonical_log_index.parquet",
    "duplicate_components.parquet",
    "quarantine.jsonl",
    "manifest.json",
    "summary.json",
]
PROTECTED_ARTIFACT_DIRS = ("audit", "canonical", "split", "bench", "quality", "replay",
                           "drift", "git_safeguard", "audit_logs")


class BuildError(RuntimeError):
    """Invalid configuration / unsafe output path; always fail closed."""


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _unique_paths(paths) -> list[Path]:
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = Path(path).resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _git_checkout_roots(start: Path) -> list[Path]:
    """Checkout roots visible from ``start`` (main checkout + every linked worktree).

    ``git rev-parse --git-common-dir`` points at the main checkout's ``.git`` even when
    invoked from a linked worktree, so the main root is always discovered; the worktree
    list additionally protects sibling worktrees that may own PR artifacts.
    """
    roots: list[Path] = []
    try:
        proc = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--git-common-dir", "--show-toplevel"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode == 0:
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                common = Path(lines[0])
                common = common if common.is_absolute() else (start / common)
                common = common.resolve()
                roots.append(common.parent if common.name == ".git" else common)
            for line in lines[1:]:
                path = Path(line)
                roots.append((path if path.is_absolute() else start / path).resolve())
        proc = subprocess.run(
            ["git", "-C", str(start), "worktree", "list", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("worktree "):
                    roots.append(Path(line[len("worktree "):].strip()).resolve())
    except (OSError, subprocess.SubprocessError):
        pass
    return roots


def repository_roots(repo: Path, extra_roots=()) -> list[Path]:
    """Every checkout root whose ``artifacts/riichiformer/*`` must stay read-only."""
    candidates = [Path(repo), REPO_ROOT, *extra_roots]
    candidates.extend(_git_checkout_roots(Path(repo)))
    return _unique_paths(candidates)


def _protected_artifact_name(path: Path) -> str | None:
    """Detect ``artifacts/riichiformer/<protected>`` in any checkout, without git."""
    parts = path.parts
    for index in range(len(parts) - 2):
        if (parts[index] == "artifacts" and parts[index + 1] == "riichiformer"
                and parts[index + 2] in PROTECTED_ARTIFACT_DIRS):
            return parts[index + 2]
    return None


def assert_safe_out(out_dir: Path, repo: Path, protected_roots=()) -> None:
    resolved = out_dir.resolve()
    # Repository-independent guard: an absolute --out under *any* checkout's
    # artifacts/riichiformer/{audit,canonical,split,...} is refused even when the
    # caller passes a worktree as --repo-root (main checkout not on that path).
    artifact_name = _protected_artifact_name(resolved)
    if artifact_name is not None:
        raise BuildError(
            "refusing to write under protected PR artifact directory "
            f"artifacts/riichiformer/{artifact_name}: {resolved}; "
            "write under artifacts/riichiformer/h02/ instead"
        )
    for root in repository_roots(repo, protected_roots):
        datasets = (root / "datasets").resolve()
        if resolved == datasets or _is_relative_to(resolved, datasets):
            raise BuildError(f"refusing to write inside read-only datasets/: {resolved}")
        for name in PROTECTED_ARTIFACT_DIRS:
            protected = (root / "artifacts" / "riichiformer" / name).resolve()
            if resolved == protected or _is_relative_to(resolved, protected):
                raise BuildError(
                    f"refusing to overwrite existing PR artifact directory {protected}; "
                    "write under artifacts/riichiformer/h02/ instead"
                )


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _maxrss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def _atomic_write_text(path: Path, text: str) -> None:
    with dedup.atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")


def _append_run_log(out_dir: Path, record: dict) -> None:
    record = dict(record)
    record.setdefault("ts", _utc_now())
    with open(out_dir / "run_log.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def iter_audit_leaves(path: Path, batch_size: int = 65536):
    """Stream every leaf row of the PR-02 audit parquet (read-only)."""
    import pyarrow.parquet as pq

    if not path.is_file():
        raise BuildError(f"audit parquet not found: {path} (run PR-02 first)")
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=AUDIT_COLUMNS):
        for row in batch.to_pylist():
            if row.get("kind") != "leaf":
                continue
            yield row


def _audit_sha256(path: Path) -> str:
    return dedup.sha256_file(path)


def select_sample_rows(audit_path: Path, sample: int, seed: int) -> list[dict]:
    """Deterministic stratified sample with bounded memory (three passes worst case)."""
    if sample <= 0:
        raise BuildError("--sample must be > 0 (use --all for the full corpus)")
    rows = _ReiterableLeaves(audit_path)
    selected = dedup.select_streamed(rows, sample, seed=seed)
    if len(selected) < sample:
        raise BuildError(
            f"only {len(selected)} audited leaves available for --sample {sample}"
        )
    return selected


class _ReiterableLeaves:
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        return iter_audit_leaves(self.path)


def canonicalize_audit_row(row: dict) -> dict:
    """Canonicalise one audit row, turning every failure into an invalid record."""
    relpath = row["relpath"]
    source_id = row.get("source_id")
    year = row.get("year")
    audit_reason = row.get("invalid_reason")
    if audit_reason:
        record = canon.invalid_record(
            relpath, None, canon.CanonicalizeError("audit_invalid", audit_reason),
            source_id=source_id, year=year,
        )
        record["compression"] = row.get("compression")
        record["bytes"] = row.get("bytes")
        record["original_sha256"] = row.get("sha256")
        record["root"] = row.get("root")
        return record

    path = Path(row["root"]) / relpath
    try:
        record = canon.canonicalize_file(
            path, relpath=relpath, source_id=source_id, year=year
        )
    except canon.CanonicalizeError as exc:
        raw = None
        try:
            raw = path.read_bytes()
        except OSError:
            pass
        record = canon.invalid_record(relpath, raw, exc, source_id=source_id, year=year)
        if raw is None:
            record["original_sha256"] = row.get("sha256")
            record["bytes"] = row.get("bytes")
            record["compression"] = row.get("compression")
    record["root"] = row.get("root")
    return record


def iter_shards(rows, chunk_rows: int):
    """Deterministic year/chunk shards from a row stream.

    Shard ids are ``<year>:<chunk index>`` where the chunk index is the order in which
    that year's rows are encountered.  Resume therefore only needs the same inventory
    hash and ``--chunk-rows`` (both recorded in the store meta).
    """
    if chunk_rows <= 0:
        raise BuildError("--chunk-rows must be > 0")
    buffers: dict[int, list[dict]] = defaultdict(list)
    counters: dict[int, int] = defaultdict(int)
    for row in rows:
        year = int(row.get("year") or 0)
        buffers[year].append(row)
        if len(buffers[year]) >= chunk_rows:
            index = counters[year]
            counters[year] += 1
            yield f"{year}:{index:05d}", year, buffers[year]
            buffers[year] = []
    for year in sorted(buffers):
        if buffers[year]:
            index = counters[year]
            counters[year] += 1
            yield f"{year}:{index:05d}", year, buffers[year]
            buffers[year] = []


def _load_config(audit_sha: str, args) -> dict:
    full = bool(args.all or args.sample == 0)
    return {
        "builder": "rf_build_canonical/h02",
        "store_version": dedup.STORE_VERSION,
        "index_version": dedup.INDEX_VERSION,
        "canonical_version": canon.CANONICAL_VERSION,
        "rules_version": canon.RULES_VERSION,
        "audit_parquet_sha256": audit_sha,
        "full_corpus": full,
        "sample_size": None if full else int(args.sample),
        "sample_seed": None if full else int(args.seed),
        "chunk_rows": int(args.chunk_rows),
    }


def _restart(out_dir: Path) -> None:
    names = ["canonical_index.sqlite", "canonical_index.sqlite-wal",
             "canonical_index.sqlite-shm", "run_log.jsonl", *ARTIFACT_NAMES]
    names += [path.name for path in out_dir.glob("sample_*.jsonl")]
    for name in names:
        target = (out_dir / name)
        if target.parent.resolve() != out_dir.resolve():
            raise BuildError(f"refusing to remove unexpected path {target}")
        if target.exists():
            target.unlink()


def build(args, repo: Path) -> int:
    audit_path = _resolve(repo, args.audit_parquet)
    out_dir = _resolve(repo, args.out)
    assert_safe_out(out_dir, repo, getattr(args, "protected_root", ()))
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_sha = _audit_sha256(audit_path)
    config = _load_config(audit_sha, args)
    config_json = json.dumps(config, sort_keys=True)

    if args.restart:
        _restart(out_dir)

    store = dedup.CanonicalStore(out_dir / "canonical_index.sqlite", create=True)
    try:
        previous = store.get_meta_json("config")
        if previous is None:
            store.set_meta({"config": config, "created_utc": _utc_now()})
        elif previous != config:
            raise BuildError(
                "canonical store was built with a different config/inventory; "
                f"stored={json.dumps(previous, sort_keys=True)} "
                f"requested={config_json}. Re-run with --restart to rebuild."
            )

        full = config["full_corpus"]
        if full:
            rows = iter_audit_leaves(audit_path)
        else:
            rows = select_sample_rows(audit_path, int(args.sample), int(args.seed))
            _append_run_log(out_dir, {
                "event": "sample_selected", "sample": int(args.sample),
                "seed": int(args.seed), "rows": len(rows),
            })
            print(f"rf_build_canonical: sample={args.sample} selected={len(rows)} "
                  f"seed={args.seed}")

        shard_status = store.shard_status()
        processed = skipped = total_shards = total_rows = 0
        stopped_early = False
        t_build = time.time()
        for shard_id, year, shard_rows in iter_shards(rows, int(args.chunk_rows)):
            total_shards += 1
            total_rows += len(shard_rows)
            previous_shard = shard_status.get(shard_id)
            if previous_shard is not None:
                if previous_shard["rows"] != len(shard_rows):
                    raise BuildError(
                        f"shard {shard_id} has {previous_shard['rows']} rows in the "
                        f"checkpoint but {len(shard_rows)} rows were selected now; "
                        "the inventory or chunking changed -- use --restart"
                    )
                skipped += 1
                continue
            if args.max_shards and processed >= args.max_shards:
                stopped_early = True
                break
            t0 = time.time()
            records = [canonicalize_audit_row(row) for row in shard_rows]
            info = store.add_shard(shard_id, year, records, round(time.time() - t0, 3))
            processed += 1
            elapsed = time.time() - t_build
            rate = processed / elapsed if elapsed > 0 else 0.0
            _append_run_log(out_dir, {
                "event": "shard", "shard_id": shard_id, "year": year,
                "rows": info["rows"], "valid": info["valid"], "invalid": info["invalid"],
                "seconds": info["seconds"], "shards_processed": processed,
                "maxrss_mb": _maxrss_mb(),
            })
            print(f"  shard {shard_id}: rows={info['rows']} invalid={info['invalid']} "
                  f"{info['seconds']:.2f}s ({rate:.1f} shard/s, rss={_maxrss_mb()} MB)")

        if stopped_early:
            print(f"rf_build_canonical: stopped after {processed} new shard(s) "
                  f"(--max-shards); re-run without --max-shards to finish")
            _append_run_log(out_dir, {
                "event": "partial", "new_shards": processed, "done_shards": skipped,
                "total_shards_seen": total_shards,
            })
            return 0

        expected = store.get_meta("expected_shards")
        if expected is not None and int(expected) != total_shards:
            raise BuildError(
                f"checkpoint expects {expected} shards but the inventory now yields "
                f"{total_shards}; use --restart"
            )
        store.set_meta({
            "scan_complete": "1",
            "expected_shards": str(total_shards),
            "selected_rows": str(total_rows),
            "shards_new": str(processed),
            "shards_skipped": str(skipped),
        })

        t_final = time.time()
        finalize = store.finalize_components()
        _append_run_log(out_dir, {
            "event": "finalize", "seconds": round(time.time() - t_final, 3),
            **finalize, "maxrss_mb": _maxrss_mb(),
        })

        summary = store.summary()
        if summary["rows"] != total_rows:
            raise BuildError(
                f"store has {summary['rows']} rows but {total_rows} were selected"
            )

        index_info = dedup.write_store_index_parquet(
            store, out_dir / "canonical_log_index.parquet"
        )
        components_info = dedup.write_store_components_parquet(
            store, out_dir / "duplicate_components.parquet"
        )
        quarantine_info = dedup.write_store_quarantine_jsonl(
            store, out_dir / "quarantine.jsonl"
        )
        sample_info = None
        if not full:
            sample_name = f"sample_{int(args.sample)}.jsonl"
            sample_info = dedup.write_index_jsonl(
                store.iter_index_rows(), out_dir / sample_name
            )

        artifacts = {
            "canonical_log_index.parquet": {
                "rows": index_info["rows"],
                "sha256": dedup.sha256_file(out_dir / "canonical_log_index.parquet"),
                "logical_sha256": index_info["logical_sha256"],
            },
            "duplicate_components.parquet": {
                "rows": components_info["rows"],
                "sha256": dedup.sha256_file(out_dir / "duplicate_components.parquet"),
                "logical_sha256": components_info["logical_sha256"],
            },
            "quarantine.jsonl": {
                "rows": quarantine_info["rows"],
                "sha256": dedup.sha256_file(out_dir / "quarantine.jsonl"),
                "logical_sha256": quarantine_info["logical_sha256"],
            },
        }
        if sample_info is not None:
            sample_name = f"sample_{int(args.sample)}.jsonl"
            artifacts[sample_name] = {
                "rows": sample_info["rows"],
                "sha256": dedup.sha256_file(out_dir / sample_name),
                "logical_sha256": sample_info["logical_sha256"],
            }

        manifest = {
            "canonical_version": canon.CANONICAL_VERSION,
            "rules_version": canon.RULES_VERSION,
            "index_version": dedup.INDEX_VERSION,
            "store_version": dedup.STORE_VERSION,
            "builder": "rf_build_canonical/h02",
            "audit_parquet": str(audit_path),
            "audit_parquet_sha256": audit_sha,
            "full_corpus": full,
            "sample_size": config["sample_size"],
            "sample_seed": config["sample_seed"],
            "chunk_rows": config["chunk_rows"],
            "selected_rows": total_rows,
            "shards": total_shards,
            "rows": summary["rows"],
            "valid": summary["valid"],
            "invalid": summary["invalid"],
            "invalid_reasons": summary["invalid_reasons"],
            "components": summary["components"],
            "duplicate_groups": summary["duplicate_groups"],
            "duplicate_components": summary["duplicate_components"],
            "duplicate_rows": summary["duplicate_rows"],
            "conflict_components": summary["conflict_components"],
            "incomplete_components": summary["incomplete_components"],
            "cross_key_duplicate_rows": summary["cross_key_duplicate_rows"],
            "representatives": summary["representatives"],
            "quarantine_rows": summary["quarantine_rows"],
            "quarantine_components": summary["quarantine_components"],
            "years": summary["years"],
            "sources": summary["sources"],
            "artifacts": artifacts,
            "note": (
                "full-corpus canonical index; duplicate components are connected by "
                "game_id and canonical SHA. quarantined components must not enter any "
                "training split. sample indexes are not authoritative for dedup."
                if full else
                "sample index: canonicalisation/dedup fixtures only; NOT authoritative "
                "for corpus-wide duplicate counts (run --all under an approved IO window)."
            ),
        }
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(out_dir / "manifest.json", manifest_text)
        _atomic_write_text(out_dir / "summary.json", manifest_text)

        elapsed_total = time.time() - t_build
        _append_run_log(out_dir, {
            "event": "complete", "seconds": round(elapsed_total, 3),
            "rows": summary["rows"], "shards": total_shards,
            "new_shards": processed, "skipped_shards": skipped,
            "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
            "maxrss_mb": _maxrss_mb(),
        })
        print(
            f"rf_build_canonical: rows={summary['rows']} valid={summary['valid']} "
            f"invalid={summary['invalid']} dup_components={summary['duplicate_components']} "
            f"quarantine_rows={summary['quarantine_rows']} full_corpus={full} "
            f"shards={total_shards} ({processed} new, {skipped} resumed) "
            f"elapsed={elapsed_total:.1f}s maxrss={_maxrss_mb()}MB"
        )
        print(f"  index={out_dir / 'canonical_log_index.parquet'} "
              f"sha256={artifacts['canonical_log_index.parquet']['sha256']}")
        print(f"  manifest={out_dir / 'manifest.json'}")
        return 0 if summary["valid"] > 0 else 1
    finally:
        store.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="RiichiFormer H-02 sharded canonical index (streaming, resumable)"
    )
    ap.add_argument("--audit-parquet", default=DEFAULT_AUDIT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sample", type=int, default=1000,
                    help="stratified sample size (0 or --all for every audited leaf)")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--all", action="store_true",
                    help="canonicalise every audited leaf (full 2.51M run; IO window)")
    ap.add_argument("--chunk-rows", type=int, default=25000,
                    help="rows per checkpoint shard (resume granularity)")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="stop after N new shards (resume tests / staged runs)")
    ap.add_argument("--restart", action="store_true",
                    help="delete the checkpoint and known artifacts, then rebuild")
    ap.add_argument("--protected-root", action="append", default=[],
                    help="extra checkout root whose artifacts/riichiformer/* is read-only "
                         "(repeatable; git roots are discovered automatically)")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    try:
        return build(args, repo)
    except (BuildError, dedup.DedupError, FileNotFoundError) as exc:
        print(f"rf_build_canonical: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
