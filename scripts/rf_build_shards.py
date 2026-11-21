#!/usr/bin/env python3
"""PR-12 shard builder: H-02 canonical/split-v2 -> rf-shard/3.

Read-only over canonical/split/raw logs.  Default is a bounded 100k dry-run;
writing is explicit.  The default research profile is the strict
``licensed_only`` gate.  ``operator_private_research`` is an explicit operator
opt-in that additionally requires ``--policy-config`` and checks that the
policy allowlist covers every selected source; it records the policy version +
canonical hash while leaving the real rights status untouched.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.decision import DecisionBuildConfig, build_decisions  # noqa: E402
from riichiformer.policy.config import (  # noqa: E402
    PROFILES, PolicyConfig, PolicyConfigError, load_policy_config,
)
from riichiformer.policy.research_policy import (  # noqa: E402
    PRIVATE_PROFILE, ResearchPolicyError,
)
from riichiformer.data.shard import (  # noqa: E402
    ACTION_HASH, FROZEN_SCHEMA, NORMALIZER_HASH, SCHEMA_HASH, ShardMetadata,
    ShardRightsError, ShardWriter, make_game_record, sample_hashes,
)
from riichiformer.schemas.decision_v3 import RulesetId, SourceKind  # noqa: E402

DEFAULT_CANONICAL = REPO_ROOT / "artifacts/riichiformer/h02/bench_100k"
DEFAULT_SPLIT = REPO_ROOT / "artifacts/riichiformer/h02/split_100k/split_manifest_L_v2.parquet"
DEFAULT_RIGHTS = REPO_ROOT / "configs/riichiformer/source_rights.csv"
DEFAULT_LIMIT = 100_000
#: Conservative measured estimate from the 10-game audit shard (9.0 MB / 10 games).
ESTIMATED_BYTES_PER_GAME = 1_000_000
PROTECTED_ARTIFACT_DIRS = ("audit", "canonical", "split", "bench", "quality",
                           "replay", "drift", "git_safeguard", "audit_logs")
PROTECTED_ROOT_DIRS = ("datasets", "runs", "checkpoints")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Main checkout + every linked worktree visible from ``start``."""
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
    """Every checkout root whose datasets/runs/checkpoints/PR artifacts are read-only."""
    package_root = Path(__file__).resolve().parent.parent
    candidates = [Path(repo), package_root, *extra_roots]
    candidates.extend(_git_checkout_roots(Path(repo)))
    return _unique_paths(candidates)


def _unsafe_segment(path: Path) -> str | None:
    parts = path.parts
    for index in range(len(parts) - 2):
        if (parts[index] == "artifacts" and parts[index + 1] == "riichiformer"
                and parts[index + 2] in PROTECTED_ARTIFACT_DIRS):
            return f"artifacts/riichiformer/{parts[index + 2]}"
    for part in parts:
        if part in PROTECTED_ROOT_DIRS:
            return part
    return None


def assert_safe_shard_out(out: str | Path, repo: Path, extra_roots=()) -> None:
    """Refuse writes under datasets/runs/checkpoints or any checkout's PR artifacts.

    Repository-independent segment check catches an absolute ``--out`` aimed at the
    main checkout even when invoked from a linked worktree; the multi-root loop also
    covers datasets/runs/checkpoints below every discovered checkout root.
    """
    resolved = Path(out).resolve()
    segment = _unsafe_segment(resolved)
    if segment is not None:
        raise ShardRightsError(
            f"refusing to write shard under read-only {segment}: {resolved}; "
            "choose an output under /tmp or a dedicated non-artifact directory")
    for root in repository_roots(repo, extra_roots):
        for name in PROTECTED_ROOT_DIRS:
            protected = (root / name).resolve()
            if resolved == protected or _is_relative_to(resolved, protected):
                raise ShardRightsError(
                    f"refusing to write shard inside read-only {protected}")
        for name in PROTECTED_ARTIFACT_DIRS:
            protected = (root / "artifacts" / "riichiformer" / name).resolve()
            if resolved == protected or _is_relative_to(resolved, protected):
                raise ShardRightsError(
                    f"refusing to overwrite existing PR artifact directory {protected}; "
                    "choose a dedicated shard output directory")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _free_bytes(out: Path) -> int:
    parent = Path(out).resolve().parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return shutil.disk_usage(parent).free


def load_rights(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["source_id"]: row for row in csv.DictReader(handle)}


def source_rights_allowed(rights: dict, source_id: str) -> bool:
    row = rights.get(source_id)
    if not row:
        return False
    return (row.get("train_allowed") == "ALLOWED"
            and row.get("status") == "ALLOWED")


def select_rows(manifest_path: Path, protocol: str, limit: int,
                purpose: str, rights: dict,
                research_policy: str = "licensed_only") -> list[dict]:
    """Stream the split-v2 manifest; select representatives only.

    ``is_representative=True`` already means one row per H-02 component; the duplicate
    members carry ``is_representative=False``.  Do *not* additionally filter on
    ``component_size==1`` (that would drop every representative of a multi-file
    component).  A training build also requires the explicit train partition.
    """
    import pyarrow.parquet as pq

    columns = [
        "protocol", "is_representative", "quarantine", "component_size",
        "train_allowed", "rights_status", "source_id", "game_id", "component_id",
        "canonical_sha256", "root", "relpath", "split", "split_version",
    ]
    rows: list[dict] = []
    if limit is not None and limit <= 0:
        raise ShardRightsError(f"limit must be positive, got {limit!r}")
    parquet = pq.ParquetFile(manifest_path)
    stop = False
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        data = batch.to_pydict()
        for index in range(batch.num_rows):
            row = {name: data[name][index] for name in columns}
            if protocol and row.get("protocol") != protocol:
                continue
            if not row.get("is_representative", False):
                continue
            if row.get("quarantine", False):
                continue
            if purpose == "train":
                if row.get("split") != "train":
                    if research_policy == "operator_private_research":
                        continue  # Keep all holdout/select/confirm partitions isolated.
                    raise ShardRightsError(
                        f"training shard selected non-train split {row.get('split')!r} "
                        f"for {row.get('game_id')}")
                if row.get("source_id") not in rights:
                    raise ShardRightsError("training source missing from provenance registry")
                if research_policy == "licensed_only":
                    if not row.get("train_allowed", False):
                        raise ShardRightsError(
                            f"training shard selected blocked row {row.get('game_id')} "
                            f"(source={row.get('source_id')}, train_allowed=False)")
                    if str(row.get("rights_status", "")).upper() != "ALLOWED":
                        raise ShardRightsError(
                            f"training shard selected non-ALLOWED rights row {row.get('game_id')}")
                    if not source_rights_allowed(rights, row.get("source_id", "")):
                        raise ShardRightsError(
                            f"training source {row.get('source_id')!r} is not ALLOWED in "
                            "source_rights.csv")
            rows.append(row)
            if len(rows) >= limit:
                stop = True
                break
        if stop:
            break
    rows.sort(key=lambda row: row["game_id"])
    return rows

def assert_policy_covers_sources(policy: PolicyConfig, sources: Iterable[str]) -> None:
    """Fail closed when a private build selects a source outside the allowlist."""
    uncovered = sorted(set(sources) - set(policy.allowed_local_sources))
    if uncovered:
        raise ShardRightsError(
            f"policy allowlist does not cover training source(s) {uncovered}; "
            f"profile={policy.profile} policy_hash={policy.policy_hash}; add the "
            "source to allowed_local_sources or use --research-policy licensed_only")


def resolve_policy(research_policy: str, policy_config: str | Path | None
                   ) -> tuple[str, PolicyConfig | None]:
    """Resolve the explicit profile + optional config; private never defaults open."""
    if research_policy not in PROFILES:
        raise ShardRightsError(
            f"unknown research policy {research_policy!r}; choose one of {PROFILES}")
    policy = None
    if policy_config:
        policy = load_policy_config(policy_config)
    if policy is not None and policy.profile != research_policy:
        raise ResearchPolicyError(
            f"--research-policy {research_policy!r} does not match policy config "
            f"profile {policy.profile!r}")
    if research_policy == PRIVATE_PROFILE and policy is None:
        raise ResearchPolicyError(
            "operator_private_research requires an explicit --policy-config; "
            "the former fail-open default is removed")
    return research_policy, policy


def build_metadata(rows: list[dict], canonical_dir: Path, split_manifest: Path,
                   purpose: str, research_policy: str = "licensed_only",
                   policy: PolicyConfig | None = None) -> ShardMetadata:
    canonical_manifest = canonical_dir / "manifest.json"
    canonical = read_json(canonical_manifest)
    canonical_index_sha = canonical["artifacts"]["canonical_log_index.parquet"]["sha256"]
    canonical_manifest_sha = file_sha(canonical_manifest)
    split_sha = file_sha(split_manifest)
    sources = sorted({row["source_id"] for row in rows})
    statuses = {str(row.get("rights_status", "UNKNOWN")).upper() for row in rows}
    train_ok = all(bool(row.get("train_allowed", False)) for row in rows) \
        and statuses == {"ALLOWED"}
    if research_policy not in PROFILES:
        raise ShardRightsError("unknown research policy")
    if policy is not None:
        if not isinstance(policy, PolicyConfig):
            raise ShardRightsError("policy must be a PolicyConfig")
        if policy.profile != research_policy:
            raise ResearchPolicyError(
                f"research_policy {research_policy!r} != policy config profile "
                f"{policy.profile!r}")
    if research_policy == PRIVATE_PROFILE:
        if policy is None:
            raise ResearchPolicyError(
                "operator_private_research requires an explicit PolicyConfig")
        assert_policy_covers_sources(policy, sources)
    if purpose == "train" and research_policy == "licensed_only" and not train_ok:
        raise ShardRightsError("licensed training shard rows are not all ALLOWED")
    research_policy_hash = ""
    policy_version = ""
    if research_policy == PRIVATE_PROFILE:
        research_policy_hash = policy.policy_hash
        policy_version = policy.policy_version
    if len(sources) == 1:
        source_id = sources[0]
    else:
        source_id = "mixed:" + "+".join(sources)
    rights_status = "ALLOWED" if train_ok else (
        "RESTRICTED" if "RESTRICTED" in statuses else "UNKNOWN")
    train_eligible = purpose == "train" and (
        train_ok if research_policy == "licensed_only" else True)
    return ShardMetadata(
        purpose=purpose,
        source_id=source_id,
        rights_status=rights_status,
        external_license_status=rights_status,
        train_allowed=train_ok,
        train_eligible_under_policy=train_eligible,
        research_policy=research_policy,
        research_policy_hash=research_policy_hash,
        policy_version=policy_version,
        canonical_manifest_sha256=canonical_manifest_sha,
        canonical_index_sha256=canonical_index_sha,
        split_manifest_sha256=split_sha,
        split_version=rows[0].get("split_version", "unknown"),
        schema_version=FROZEN_SCHEMA["schema_version"],
        schema_hash=SCHEMA_HASH,
        action_map_hash=ACTION_HASH,
        normalizer_hash=NORMALIZER_HASH,
        ruleset_id=RulesetId.TENHOU_4P_RANKED.value,
        source_kind=SourceKind.HUMAN_LOG.value,
    )


def check_canonical_db(path: Path, rows: list[dict]) -> dict:
    """Read-only sanity check that selected representatives exist in the canonical DB."""
    import sqlite3
    if not path.is_file():
        raise ShardRightsError(f"canonical DB not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        if "store_version" not in meta:
            raise ShardRightsError("canonical DB has no store_version meta")
        for row in rows:
            found = conn.execute(
                "SELECT 1 FROM games WHERE game_id=? AND canonical_sha256=? LIMIT 1",
                (row["game_id"], row["canonical_sha256"])).fetchone()
            if found is None:
                raise ShardRightsError(
                    f"selected game {row['game_id']} is absent from canonical DB")
    finally:
        conn.close()
    return meta


def read_events(root: str, relpath: str) -> list[dict]:
    path = Path(root) / relpath
    data = path.read_bytes()
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def run(args) -> dict:
    research_policy = getattr(args, "research_policy", "licensed_only")
    policy = None
    research_policy, policy = resolve_policy(
        research_policy, getattr(args, "policy_config", None))
    assert_safe_shard_out(Path(args.out),
                          Path(getattr(args, "repo_root", REPO_ROOT)),
                          getattr(args, "protected_root", ()))
    dry_run = bool(args.dry_run or not args.write)
    limit = args.limit
    if args.all:
        if not args.io_window_approval:
            raise ShardRightsError("--all requires --io-window-approval (Lead-approved window)")
        if args.purpose != "train":
            raise ShardRightsError("--all is only supported for --purpose train")
        limit = 10**9
    elif limit > DEFAULT_LIMIT:
        raise ShardRightsError(f"limit {limit} exceeds bounded dry-run {DEFAULT_LIMIT}; use --all")
    rights = load_rights(Path(args.source_rights))
    rows = select_rows(Path(args.split_manifest), args.protocol, limit,
                       args.purpose, rights, research_policy=research_policy)
    if not rows:
        raise ShardRightsError("no representative rows selected")
    metadata = build_metadata(rows, Path(args.canonical_dir),
                              Path(args.split_manifest), args.purpose,
                              research_policy=research_policy, policy=policy)
    canonical_db = Path(args.canonical_db) if args.canonical_db else (
        Path(args.canonical_dir) / "canonical_index.sqlite")
    db_meta = check_canonical_db(canonical_db, rows) if args.check_canonical_db else {}
    estimated_bytes = len(rows) * ESTIMATED_BYTES_PER_GAME
    free_bytes = _free_bytes(Path(args.out))
    if not dry_run and estimated_bytes > free_bytes:
        raise ShardRightsError(
            f"insufficient disk: estimated {estimated_bytes} bytes for {len(rows)} "
            f"games, only {free_bytes} free under {Path(args.out).resolve().parent}")
    print(f"PR12_PLAN purpose={args.purpose} rows={len(rows)} "
          f"sources={metadata.source_id} rights={metadata.rights_status} "
          f"train_allowed={metadata.train_allowed} research_policy={research_policy} "
          f"policy_version={metadata.policy_version} "
          f"policy_hash={metadata.research_policy_hash} "
          f"train_eligible_under_policy={metadata.train_eligible_under_policy} "
          f"dry_run={dry_run} estimated_bytes={estimated_bytes} "
          f"free_bytes={free_bytes} canonical={db_meta.get('store_version', 'unchecked')}")
    if dry_run:
        return {"selected": len(rows), "written": 0, "dry_run": True,
                "rights_status": metadata.rights_status,
                "estimated_bytes": estimated_bytes}
    writer = ShardWriter(args.out, metadata, resume=args.resume)
    written = 0
    skipped = 0
    try:
        for index, row in enumerate(rows):
            try:
                raw = read_events(row["root"], row["relpath"])
                samples = build_decisions(
                    raw, DecisionBuildConfig(
                        ruleset_id=RulesetId.TENHOU_4P_RANKED,
                        source_kind=SourceKind.HUMAN_LOG))
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"PR12_SKIP game={row['game_id']} {type(exc).__name__}: {exc}")
                continue
            if not samples:
                skipped += 1
                continue
            record = make_game_record(
                raw, samples, game_id=row["game_id"],
                component_id=row["component_id"],
                component_sha256=row["canonical_sha256"],
                split=row["split"], source_id=row["source_id"],
                train_allowed=bool(row.get("train_allowed", False)),
                rights_status=str(row.get("rights_status", "UNKNOWN")))
            writer.add_game(record)
            written += 1
            if (index + 1) % 100 == 0:
                print(f"PR12_PROGRESS games={written} selected={index + 1}")
        manifest = writer.finalize()
    finally:
        if not writer._closed:
            writer.close()
    hashes = sample_hashes(args.out)
    print(f"PR12_DONE games={manifest['game_count']} decisions={manifest['decision_count']} "
          f"written={written} skipped={skipped} shard_sha256={manifest['shard_sha256']} "
          f"sample_hashes={hashes}")
    return {"manifest": manifest, "written": written, "skipped": skipped,
            "sample_hashes": hashes, "metadata": metadata.as_dict(),
            "estimated_bytes": estimated_bytes}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", default=str(DEFAULT_CANONICAL))
    parser.add_argument("--canonical-db", default=None)
    parser.add_argument("--check-canonical-db", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT))
    parser.add_argument("--protocol", default="L", choices=("L", "R"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--purpose", default="diagnostic",
                        choices=("train", "audit", "diagnostic"))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--io-window-approval", default=None)
    parser.add_argument("--source-rights", default=str(DEFAULT_RIGHTS))
    parser.add_argument("--research-policy",
                        choices=PROFILES,
                        default="licensed_only",
                        help="licensed_only (strict default) or "
                             "operator_private_research (requires --policy-config "
                             "and records the real rights status unchanged)")
    parser.add_argument("--policy-config", default=None,
                        help="strict YAML/JSON PolicyConfig; required for "
                             "operator_private_research and hashed into metadata")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate/report only (default unless --write)")
    parser.add_argument("--write", action="store_true",
                        help="actually write the shard (default is dry-run)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--repo-root", default=str(REPO_ROOT),
                        help="checkout root used for the read-only write guard")
    parser.add_argument("--protected-root", action="append", default=[],
                        help="extra checkout root whose data/artifact dirs are read-only")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ShardRightsError, ResearchPolicyError, PolicyConfigError) as exc:
        print(f"PR12_RIGHTS_REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"PR12_FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
