"""Read-only yearly file audit for RiichiFormer V2 (PR-02).

The audit never downloads, normalises, moves or deletes a file.  It only:

* walks local roots (directories and/or ``.zip`` archives);
* classifies each game log by **magic bytes** (gzip / plain JSON), never by extension;
* records per-file size / mtime / year / source and streams them to Parquet;
* validates every zip's central directory + CRC via ``ZipFile.testzip()``;
* deep-parses a deterministic per-year sample (``start_kyoku`` count and event types);
* writes ``yearly_counts.csv``, ``bad_files.jsonl``, ``samples.jsonl``,
  ``source_rights.csv`` and a human report.

Important limitation, stated in the report: CRC/JSON validation for *every* leaf file is
deferred to PR-03/PR-04.  PR-02 detects corruption in zips (all members), sampled logs,
and files whose magic cannot be classified; it does not claim 100% JSON validity.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import json
import os
import platform
import re
import socket
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from riichiformer.schemas.provenance import (
    AUDIT_SCHEMA_VERSION,
    YEAR_MAX,
    YEAR_MIN,
    RightsStatus,
    SourceId,
    default_source_rights,
    rights_block_reasons,
    source_for_year,
)

LOG_SUFFIXES = (".mjson", ".mjson.gz", ".json", ".json.gz")
ZIP_MAGIC = b"PK\x03\x04"
ZIP_EMPTY_MAGIC = b"PK\x05\x06"


def utc_iso(ts: float | None = None) -> str:
    dt = datetime.fromtimestamp(ts, timezone.utc) if ts is not None else datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def sniff_magic(head: bytes) -> str:
    """Classify a file/stream by its first bytes (extension-independent)."""
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(ZIP_MAGIC) or head.startswith(ZIP_EMPTY_MAGIC):
        return "zip"
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"{"):
        return "json"
    if not head:
        return "empty"
    return "unknown"


def detect_compression(path: Path) -> str:
    try:
        with open(path, "rb") as fh:
            return sniff_magic(fh.read(16))
    except OSError:
        return "unreadable"


def year_from_relpath(relpath: str):
    """Derive a supported year only from an explicit directory name or filename prefix.

    Never regex-search arbitrary path text: Tenhou log names contain hexadecimal
    suffixes (e.g. ``...-e2011cd6.mjson``) that would otherwise be misread as 2011.
    """
    parts = Path(relpath).parts
    for part in reversed(parts[:-1]):
        if re.fullmatch(r"20\d{2}", part):
            year = int(part)
            if YEAR_MIN <= year <= YEAR_MAX:
                return year
    base = Path(relpath).name
    match = re.match(r"(?<!\d)(20\d{2})(?!\d)", base)
    if match:
        year = int(match.group(1))
        if YEAR_MIN <= year <= YEAR_MAX:
            return year
    return None


def is_log_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(LOG_SUFFIXES)


def decode_log_bytes(raw: bytes, compression: str) -> bytes:
    if compression == "gzip":
        return gzip.decompress(raw)
    return raw


def parse_mjai_sample(raw: bytes) -> dict:
    """Strict line-by-line NDJSON parse; returns counts, never raises on bad logs."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return {"ok": False, "error": f"utf8_decode: {exc}", "events": 0, "kyoku": 0,
                "has_start_game": False, "has_end_game": False, "types": {}}
    types: Counter = Counter()
    bad_lines = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        if isinstance(event, dict) and isinstance(event.get("type"), str):
            types[event["type"]] += 1
    known = {"start_game", "start_kyoku", "end_kyoku", "end_game"}
    has_required = known.issubset(types)
    return {
        "ok": bad_lines == 0 and bool(types) and has_required,
        "error": None if bad_lines == 0 else f"bad_json_lines:{bad_lines}",
        "events": sum(types.values()),
        "kyoku": int(types.get("start_kyoku", 0)),
        "has_start_game": "start_game" in types,
        "has_end_game": "end_game" in types,
        "types": dict(sorted(types.items())),
        "bad_lines": bad_lines,
    }


def validate_zip(path: Path) -> dict:
    """Read the central directory and verify every member CRC (``testzip``)."""
    rec = {
        "container": "zip",
        "member_count": 0,
        "log_member_count": 0,
        "log_member_bytes": 0,
        "valid": False,
        "bad_member": None,
        "error": None,
    }
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            rec["member_count"] = len(names)
            for name in names:
                if is_log_name(name):
                    rec["log_member_count"] += 1
                    try:
                        rec["log_member_bytes"] += zf.getinfo(name).file_size
                    except KeyError:  # pragma: no cover - inconsistent zip
                        pass
            bad = zf.testzip()
            if bad is None:
                rec["valid"] = True
            else:
                rec["bad_member"] = bad
                rec["error"] = f"zip_crc:{bad}"
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def sample_zip_members(path: Path, count: int = 3):
    """Parse a few members of a valid zip; returns (samples, bad) without extraction."""
    samples, bad = [], []
    try:
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if is_log_name(n))
            for name in names[:count]:
                try:
                    raw = zf.read(name)
                except (KeyError, OSError, RuntimeError) as exc:
                    bad.append({"path": f"{path.name}!{name}", "reason": f"zip_read:{exc}"})
                    continue
                compression = sniff_magic(raw[:16])
                try:
                    payload = decode_log_bytes(raw, compression)
                except (OSError, EOFError) as exc:
                    bad.append({"path": f"{path.name}!{name}", "reason": f"gzip_decode:{exc}"})
                    continue
                parsed = parse_mjai_sample(payload)
                samples.append({
                    "path": f"{path.name}!{name}",
                    "compression": compression,
                    "bytes": len(raw),
                    "parsed": parsed,
                })
    except zipfile.BadZipFile:
        pass
    return samples, bad


@dataclass
class AuditStats:
    started: float = field(default_factory=time.time)
    leaf_files: int = 0
    zip_files: int = 0
    zip_members: int = 0
    total_bytes: int = 0
    skipped_files: int = 0
    by_year: dict = field(default_factory=dict)
    bad_files: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    sampled_by_year: Counter = field(default_factory=Counter)
    sampled_ok_by_year: Counter = field(default_factory=Counter)
    sampled_kyoku_by_year: Counter = field(default_factory=Counter)

    def year_row(self, year, source_id: str) -> dict:
        key = (year, source_id)
        if key not in self.by_year:
            self.by_year[key] = {
                "year": year,
                "source_id": source_id,
                "files": 0,
                "bytes": 0,
                "gzip": 0,
                "plaintext_json": 0,
                "empty": 0,
                "other": 0,
                "invalid": 0,
                "zip_containers": 0,
                "zip_members": 0,
                "zip_member_bytes": 0,
                "sampled": 0,
                "sampled_ok": 0,
                "sampled_kyoku": 0,
                "sample_errors": 0,
            }
        return self.by_year[key]


def _iter_entries(root: Path):
    """Yield (kind, path) for leaf logs and zip containers, deterministic order."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            if name.lower().endswith(".zip"):
                yield "zip", path
            elif is_log_name(name):
                yield "log", path


def _parquet_schema():
    import pyarrow as pa

    return pa.schema([
        ("root", pa.string()),
        ("relpath", pa.string()),
        ("year", pa.int16()),
        ("source_id", pa.string()),
        ("kind", pa.string()),
        ("compression", pa.string()),
        ("bytes", pa.int64()),
        ("mtime_utc", pa.string()),
        ("sha256", pa.string()),
        ("member_count", pa.int64()),
        ("invalid_reason", pa.string()),
    ])


class _ParquetSink:
    def __init__(self, path: Path):
        import pyarrow.parquet as pq

        self.path = path
        self.schema = _parquet_schema()
        self.batch = []
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.rows = 0

    def add(self, row: dict) -> None:
        self.batch.append(row)
        if len(self.batch) >= 50_000:
            self.flush()

    def flush(self) -> None:
        import pyarrow as pa

        if not self.batch:
            return
        columns = {name: [row.get(name) for row in self.batch] for name in self.schema.names}
        table = pa.Table.from_pydict(columns, schema=self.schema)
        self.writer.write_table(table)
        self.rows += len(self.batch)
        self.batch.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def _empty_parquet(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(_parquet_schema().empty_table(), path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _git_sha(repo: Path):
    try:
        proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else None
    except OSError:
        return None


def run_audit(roots: list[Path], out_dir: Path, report_path: Path | None = None,
              sample_per_year: int = 200, hash_mode: str = "sample",
              repo: Path | None = None) -> dict:
    """Audit ``roots`` and write parquet/csv/jsonl/report; returns the summary dict.

    Rows are streamed to Parquet, so 2.5 M local logs do not require 2.5 M dicts in RAM.
    Deep samples are the first ``sample_per_year`` files per year in deterministic
    (sorted) traversal order.
    """
    if sample_per_year < 0:
        raise ValueError("sample_per_year must be >= 0")
    if hash_mode not in ("sample", "all", "none"):
        raise ValueError("hash_mode must be sample|all|none")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = AuditStats()
    sink = _ParquetSink(out_dir / "source_audit.parquet")
    sampled_count: Counter = Counter()
    missing_roots = []

    for root in roots:
        root = Path(root)
        if not root.exists():
            missing_roots.append(str(root))
            continue
        root_label = str(root)
        if root.is_file():
            if root.suffix.lower() == ".zip":
                _audit_zip(root, root_label, stats, sink)
            else:
                stats.skipped_files += 1
            continue
        for kind, path in _iter_entries(root):
            if kind == "zip":
                _audit_zip(path, root_label, stats, sink)
            else:
                _audit_leaf(path, root, root_label, stats, sink,
                            sample_per_year, sampled_count, hash_mode)

    sink.close()
    if sink.rows == 0:
        _empty_parquet(out_dir / "source_audit.parquet")

    yearly_rows = sorted(stats.by_year.values(),
                         key=lambda r: (r["year"] if r["year"] is not None else 9999, r["source_id"]))
    yearly_path = out_dir / "yearly_counts.csv"
    _write_csv(yearly_path, yearly_rows, [
        "year", "source_id", "files", "bytes", "gzip", "plaintext_json", "empty", "other",
        "invalid", "zip_containers", "zip_members", "zip_member_bytes",
        "sampled", "sampled_ok", "sampled_kyoku", "sample_errors",
    ])
    _write_jsonl(out_dir / "bad_files.jsonl", stats.bad_files)
    _write_jsonl(out_dir / "samples.jsonl", stats.samples)

    rights = default_source_rights()
    rights_path = out_dir / "source_rights.csv"
    rights_rows = [row.as_row() for row in rights]
    _write_csv(rights_path, rights_rows, list(rights_rows[0].keys()))

    block_reasons = rights_block_reasons(rights)
    if stats.leaf_files == 0 and stats.zip_members == 0:
        status = "EMPTY"
        block_reasons.append("no_game_logs_found")
    else:
        status = "PASS"
    summary = {
        "schema": AUDIT_SCHEMA_VERSION,
        "status": status,
        "mode": "scan-only",
        "roots": [str(r) for r in roots],
        "missing_roots": missing_roots,
        "started_at_utc": utc_iso(stats.started),
        "finished_at_utc": utc_iso(),
        "duration_seconds": round(time.time() - stats.started, 2),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git_sha": _git_sha(repo) if repo else None,
        "leaf_files": stats.leaf_files,
        "zip_files": stats.zip_files,
        "zip_members": stats.zip_members,
        "total_leaf_bytes": stats.total_bytes,
        "skipped_files": stats.skipped_files,
        "parquet_rows": sink.rows,
        "sample_per_year": sample_per_year,
        "hash_mode": hash_mode,
        "sampled": len(stats.samples),
        "bad_files": len(stats.bad_files),
        "yearly_rows": yearly_rows,
        "training_blocked": bool(block_reasons),
        "block_reasons": block_reasons,
        "limitations": [
            "full JSON/CRC validation of every leaf file is deferred to PR-03/PR-04",
            "zip member CRC is validated for every archive (ZipFile.testzip)",
            "year is derived from local path; missing timestamps are not guessed",
            "no network; no identity/name extraction",
        ],
        "outputs": {
            "source_audit.parquet": str(out_dir / "source_audit.parquet"),
            "yearly_counts.csv": str(yearly_path),
            "bad_files.jsonl": str(out_dir / "bad_files.jsonl"),
            "samples.jsonl": str(out_dir / "samples.jsonl"),
            "source_rights.csv": str(rights_path),
        },
    }
    (out_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(summary), encoding="utf-8")
    return summary


def _audit_leaf(path: Path, root: Path, root_label: str, stats: AuditStats,
                sink: "_ParquetSink", sample_per_year: int, sampled_count: Counter,
                hash_mode: str) -> None:
    rel = str(path.relative_to(root))
    year = year_from_relpath(rel)
    if year is None:
        year = year_from_relpath(str(root)) or year_from_relpath(str(path))
    source = source_for_year(year).value
    try:
        st = path.stat()
    except OSError as exc:
        stats.bad_files.append({"path": f"{root_label}/{rel}", "year": year,
                                "reason": f"stat:{exc}"})
        return
    compression = detect_compression(path)
    row = {
        "root": root_label,
        "relpath": rel,
        "year": year,
        "source_id": source,
        "kind": "leaf",
        "compression": compression,
        "bytes": st.st_size,
        "mtime_utc": utc_iso(st.st_mtime),
        "sha256": None,
        "member_count": None,
        "invalid_reason": None,
    }
    yrow = stats.year_row(year, source)
    stats.leaf_files += 1
    stats.total_bytes += st.st_size
    yrow["files"] += 1
    yrow["bytes"] += st.st_size
    if compression == "gzip":
        yrow["gzip"] += 1
    elif compression == "json":
        yrow["plaintext_json"] += 1
    elif compression == "empty":
        yrow["empty"] += 1
        row["invalid_reason"] = "empty_file"
        stats.bad_files.append({"path": f"{root_label}/{rel}", "year": year,
                                "reason": "empty_file"})
    else:
        yrow["other"] += 1
        row["invalid_reason"] = f"unknown_magic:{compression}"
        stats.bad_files.append({"path": f"{root_label}/{rel}", "year": year,
                                "reason": f"unknown_magic:{compression}"})
    if row["invalid_reason"]:
        yrow["invalid"] += 1

    do_sample = year is not None and sampled_count[year] < sample_per_year
    if do_sample:
        sampled_count[year] += 1
        try:
            raw = path.read_bytes()
        except OSError as exc:
            stats.bad_files.append({"path": str(path), "year": year,
                                    "reason": f"sample_read:{exc}"})
        else:
            if hash_mode in ("sample", "all"):
                row["sha256"] = hashlib.sha256(raw).hexdigest()
            sample_rec = {"path": str(path), "year": year, "compression": compression}
            try:
                parsed = parse_mjai_sample(decode_log_bytes(raw, compression))
            except (OSError, EOFError, gzip.BadGzipFile) as exc:
                parsed = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "events": 0, "kyoku": 0}
                stats.bad_files.append({"path": str(path), "year": year,
                                        "reason": f"sample_decode:{exc}"})
            sample_rec.update({"ok": parsed["ok"], "events": parsed.get("events", 0),
                               "kyoku": parsed.get("kyoku", 0),
                               "error": parsed.get("error")})
            stats.samples.append(sample_rec)
            yrow["sampled"] += 1
            if sample_rec["ok"]:
                yrow["sampled_ok"] += 1
                yrow["sampled_kyoku"] += sample_rec["kyoku"]
            else:
                yrow["sample_errors"] += 1
                if sample_rec.get("error"):
                    stats.bad_files.append({"path": str(path), "year": year,
                                            "reason": f"sample_parse:{sample_rec['error']}"})
    elif hash_mode == "all":
        try:
            row["sha256"] = _sha256_file(path)
        except OSError as exc:
            row["invalid_reason"] = f"hash:{exc}"

    sink.add(row)


def _zip_year(path: Path):
    return year_from_relpath(path.name)


def _audit_zip(path: Path, root_label: str, stats: AuditStats, sink: "_ParquetSink") -> None:
    year = _zip_year(path)
    source = source_for_year(year).value
    stats.zip_files += 1
    try:
        st = path.stat()
    except OSError as exc:
        stats.bad_files.append({"path": root_label, "year": year, "reason": f"stat:{exc}"})
        return
    validation = validate_zip(path)
    stats.zip_members += validation["member_count"]
    row = {
        "root": root_label,
        "relpath": path.name,
        "year": year,
        "source_id": source,
        "kind": "zip",
        "compression": "zip",
        "bytes": st.st_size,
        "mtime_utc": utc_iso(st.st_mtime),
        "sha256": None,
        "member_count": validation["member_count"],
        "invalid_reason": validation["error"],
    }
    yrow = stats.year_row(year, source)
    yrow["zip_containers"] += 1
    yrow["zip_members"] += validation["member_count"]
    yrow["zip_member_bytes"] += validation["log_member_bytes"]
    if not validation["valid"]:
        stats.bad_files.append({"path": root_label, "year": year,
                                "reason": validation["error"] or "invalid_zip"})
    sink.add(row)

    samples, bad = sample_zip_members(path)
    for sample in samples:
        parsed = sample["parsed"]
        stats.samples.append({
            "path": sample["path"], "year": year, "compression": sample["compression"],
            "ok": parsed["ok"], "events": parsed["events"], "kyoku": parsed["kyoku"],
            "error": parsed["error"], "container": "zip",
        })
        yrow["sampled"] += 1
        if parsed["ok"]:
            yrow["sampled_ok"] += 1
            yrow["sampled_kyoku"] += parsed["kyoku"]
        else:
            yrow["sample_errors"] += 1
    for item in bad:
        stats.bad_files.append({"path": item["path"], "year": year, "reason": item["reason"]})


def render_report(summary: dict) -> str:
    lines = []
    lines.append("# RiichiFormer V2 — DATA_REPORT (PR-02 audit)")
    lines.append("")
    lines.append(f"- 状态：**{summary['status']}**；training_blocked=**{summary['training_blocked']}**")
    lines.append(f"- 生成时间：{summary['finished_at_utc']}（UTC）；耗时 {summary['duration_seconds']} s")
    lines.append(f"- git HEAD：`{summary['git_sha']}`")
    lines.append(f"- roots：{', '.join(f'`{r}`' for r in summary['roots'])}")
    lines.append(f"- 扫描：{summary['leaf_files']} 个叶子日志、{summary['zip_files']} 个 zip"
                 f"（{summary['zip_members']} members）、{summary['sampled']} 个深度抽样")
    lines.append(f"- bad/yellow：{summary['bad_files']} 条（见 `bad_files.jsonl`）")
    lines.append("")
    lines.append("## 来源权利")
    lines.append("")
    lines.append("**所有来源均为 `unknown`（Tenhou terms 未离线核实）→ 新训练数据 BLOCKED。**")
    lines.append("")
    lines.append("| source | rights | allows_new_training | terms |")
    lines.append("|---|---|---|---|")
    for row in default_source_rights():
        lines.append(f"| `{row.source_id.value}` | `{row.rights_status.value}` | "
                     f"{row.allows_new_training} | {row.terms_url} |")
    lines.append("")
    lines.append("## 年度覆盖")
    lines.append("")
    lines.append("| year | source | files | gzip | plaintext | other | bytes | zip members | sampled | kyoku(sampled) |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["yearly_rows"]:
        lines.append(f"| {row['year'] if row['year'] is not None else 'unknown'} | "
                     f"`{row['source_id']}` | {row['files']} | {row['gzip']} | "
                     f"{row['plaintext_json']} | {row['other']} | {row['bytes']} | "
                     f"{row['zip_members']} | {row['sampled']} | {row['sampled_kyoku']} |")
    lines.append("")
    lines.append("## 限制")
    lines.append("")
    for item in summary["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("> 本报告只有本地只读证据；没有下载、没有改写数据、没有推断玩家身份。"
                 "机器可读输出在 `artifacts/riichiformer/audit/`。")
    lines.append("")
    return "\n".join(lines)
