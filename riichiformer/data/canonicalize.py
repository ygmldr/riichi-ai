"""Lossless MJAI canonicalisation for RiichiFormer V2 (PR-03).

What "canonical" means here:

* decode gzip or plaintext, then **strictly** parse one JSON object per non-empty line;
* preserve event order and every field/value (no recomputation, no repair, no hidden
  tile recovery, no future labels invented);
* serialise with sorted object keys, no extra whitespace and UTF-8 (``ensure_ascii=False``)
  so the same events in gzip or plaintext get the **same** canonical SHA;
* derive ``game_id`` from the Tenhou log id when present, else from the canonical SHA;
* reject incomplete games (missing/duplicated ``start_game``/``end_game``, nested or
  unpaired ``start_kyoku``/``end_kyoku``).

The canonical form deliberately keeps unknown event types and unknown fields: this is a
lossless re-encoding, not a schema migration.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

CANONICAL_VERSION = "rf-canon/1"
TENHOU_LOG_ID_RE = re.compile(r"^(?P<id>\d{10}gm-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{8})$")
LOG_SUFFIXES = (".mjson.gz", ".mjson", ".json.gz", ".json")


class CanonicalizeError(ValueError):
    """A game that cannot be canonicalised losslessly; a reason is always recorded."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def detect_compression_bytes(raw: bytes) -> str:
    if raw.startswith(b"\x1f\x8b"):
        return "gzip"
    stripped = raw.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"{"):
        return "json"
    if not raw:
        return "empty"
    return "unknown"


def decode_log_bytes(raw: bytes, compression: str | None = None) -> bytes:
    compression = compression or detect_compression_bytes(raw)
    if compression == "gzip":
        try:
            return gzip.decompress(raw)
        except (OSError, EOFError) as exc:
            raise CanonicalizeError("gzip_decode", f"{type(exc).__name__}: {exc}") from exc
    if compression == "json":
        return raw
    raise CanonicalizeError(f"unsupported_compression:{compression}")


def parse_events(text: bytes | str) -> list[dict]:
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalizeError("utf8_decode", str(exc)) from exc
    events = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CanonicalizeError("bad_json", f"line {lineno}: {exc}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise CanonicalizeError("bad_event", f"line {lineno}: not an object with string type")
        events.append(event)
    if not events:
        raise CanonicalizeError("empty_log")
    return events


def validate_sequence(events: list[dict]) -> int:
    """Validate start/end pairing; returns the number of ``start_kyoku``."""
    types = [e["type"] for e in events]
    if types[0] != "start_game":
        raise CanonicalizeError("missing_start_game", f"first event is {types[0]!r}")
    if types.count("start_game") != 1:
        raise CanonicalizeError("duplicate_start_game", f"{types.count('start_game')} occurrences")
    if types[-1] != "end_game":
        raise CanonicalizeError("missing_end_game", f"last event is {types[-1]!r}")
    if types.count("end_game") != 1:
        raise CanonicalizeError("duplicate_end_game", f"{types.count('end_game')} occurrences")

    open_kyoku = False
    kyoku_count = 0
    for index, event_type in enumerate(types[1:-1], start=1):
        if event_type == "start_kyoku":
            if open_kyoku:
                raise CanonicalizeError("nested_start_kyoku", f"event index {index}")
            open_kyoku = True
            kyoku_count += 1
        elif event_type == "end_kyoku":
            if not open_kyoku:
                raise CanonicalizeError("unpaired_end_kyoku", f"event index {index}")
            open_kyoku = False
        elif event_type in ("start_game", "end_game"):
            raise CanonicalizeError("duplicate_boundary", f"event index {index}: {event_type}")
        elif not open_kyoku:
            raise CanonicalizeError("event_outside_kyoku", f"event index {index}: {event_type}")
    if open_kyoku:
        raise CanonicalizeError("unclosed_kyoku", "missing end_kyoku")
    if kyoku_count == 0:
        raise CanonicalizeError("no_kyoku")
    return kyoku_count


def canonical_text(events: list[dict]) -> str:
    return "".join(
        json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for event in events
    )


def platform_key_from_name(name: str):
    stem = Path(name).name
    for suffix in LOG_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    match = TENHOU_LOG_ID_RE.match(stem)
    return match.group("id") if match else None


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonicalize_bytes(raw: bytes, relpath: str, compression: str | None = None,
                       source_id: str | None = None, year: int | None = None) -> dict:
    """Canonicalise one game. Raises ``CanonicalizeError`` (never silently repairs)."""
    compression = compression or detect_compression_bytes(raw)
    payload = decode_log_bytes(raw, compression)
    events = parse_events(payload)
    kyoku_count = validate_sequence(events)
    text = canonical_text(events)
    canonical_sha = _sha256_bytes(text.encode("utf-8"))
    original_sha = _sha256_bytes(raw)
    platform_key = platform_key_from_name(relpath)
    game_id = f"tenhou:{platform_key}" if platform_key else f"sha256:{canonical_sha}"
    return {
        "canonical_version": CANONICAL_VERSION,
        "game_id": game_id,
        "platform_game_key": platform_key,
        "source_id": source_id,
        "year": year,
        "relpath": relpath,
        "compression": compression,
        "bytes": len(raw),
        "original_sha256": original_sha,
        "canonical_sha256": canonical_sha,
        "events": len(events),
        "kyoku_count": kyoku_count,
        "start_game": True,
        "end_game": True,
        "invalid_reason": None,
        "canonical_text_sha256": _sha256_bytes(text.encode("utf-8")),
    }


def canonicalize_file(path: Path, relpath: str | None = None,
                      source_id: str | None = None, year: int | None = None) -> dict:
    relpath = relpath or str(path)
    compression = detect_compression_bytes(_read_head(path))
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CanonicalizeError("read_error", f"{type(exc).__name__}: {exc}") from exc
    rec = canonicalize_bytes(raw, relpath=relpath, compression=compression,
                             source_id=source_id, year=year)
    rec["mtime_utc"] = None  # filled by the index builder if needed
    return rec


def _read_head(path: Path, size: int = 16) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(size)
    except OSError:
        return b""


def invalid_record(relpath: str, raw: bytes | None, error: CanonicalizeError,
                   source_id: str | None = None, year: int | None = None) -> dict:
    """Index row for a game that must be excluded (reason preserved verbatim)."""
    raw = raw or b""
    return {
        "canonical_version": CANONICAL_VERSION,
        "game_id": None,
        "platform_game_key": platform_key_from_name(relpath),
        "source_id": source_id,
        "year": year,
        "relpath": relpath,
        "compression": detect_compression_bytes(raw),
        "bytes": len(raw),
        "original_sha256": _sha256_bytes(raw) if raw else None,
        "canonical_sha256": None,
        "events": 0,
        "kyoku_count": 0,
        "start_game": False,
        "end_game": False,
        "invalid_reason": f"{error.reason}: {error.detail}" if error.detail else error.reason,
        "canonical_text_sha256": None,
    }
