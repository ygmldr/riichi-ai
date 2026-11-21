#!/usr/bin/env python3
"""Fail-closed source-rights gate for RiichiFormer (H-03).

Reads a reviewed ``source_rights.csv`` registry and answers exactly one
question: may *this* source be used for *this* purpose?

Design rules
------------
* Only a purpose cell whose value is exactly ``ALLOWED`` can pass.
* An ``ALLOWED`` row must carry a specific evidence URL, a verbatim quote,
  a retrieval date, a non-empty privacy note, and a reviewer that is not
  marked as an AI/draft/pending review.  The reviewer check is deliberate:
  this agent may collect evidence, but it may not approve a source.
* ``UNKNOWN`` and ``RESTRICTED`` always deny.
* An unknown ``source_id`` is an error (exit 2), not an implicit deny; a
  known default that is missing from the registry is also an error.
* A malformed registry is an error (exit 2); it never silently degrades to
  "allow" or "deny everything but keep going".
* ``status`` is the row-level review verdict: ``ALLOWED`` iff at least one
  purpose is ``ALLOWED``; otherwise ``RESTRICTED`` if any purpose is
  ``RESTRICTED``, else ``UNKNOWN``.  ``check()`` revalidates this even for a
  hand-built row, and a purpose can only pass when ``status=ALLOWED``.

Exit codes
----------
0  requested source/purpose is ALLOWED with complete evidence
1  source is known but the requested purpose is not ALLOWED
2  malformed CSV, unknown source, or invalid CLI usage

This script is read-only: it performs no downloads, writes no files, and
does not import training code.  It is an operational safety rule, **not a
legal opinion**; see docs/riichiformer/v3/SOURCE_RIGHTS_REVIEW.md.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import io
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "configs" / "riichiformer" / "source_rights.csv"

PURPOSES = ("storage", "train", "redistribute")
PURPOSE_COLUMNS = {
    "storage": "storage_allowed",
    "train": "train_allowed",
    "redistribute": "redistribute_allowed",
}
STATUS_VALUES = frozenset({"ALLOWED", "UNKNOWN", "RESTRICTED"})
REQUIRED_COLUMNS = (
    "source_id",
    "origin_url",
    "retrieved_at",
    "terms_url",
    "evidence_url",
    "evidence_quote",
    "storage_allowed",
    "train_allowed",
    "redistribute_allowed",
    "privacy_notes",
    "status",
    "reviewer",
    "notes",
)
# Any of these markers means the row is not a human approval.  Keep the list
# deliberately broad: the cost of a false positive is one more review, while
# the cost of a false negative is an unauthorised training run.
PENDING_REVIEWER_MARKERS = (
    "pending",
    "tbd",
    "todo",
    "unreviewed",
    "not reviewed",
    "ai agent",
    "agent draft",
    "draft",
    "n/a",
)
# Short markers are matched as tokens, not substrings: a human reviewer named
# "Kaito" must not be rejected merely because it contains "ai".  The token
# rules cover plain AI/LLM/GPT/ChatGPT/bot/automated (case-insensitive) and the
# common compact forms that a substring search would miss.
PENDING_REVIEWER_TOKENS = (
    "ai",
    "llm",
    "gpt",
    "chatgpt",
    "chatbot",
    "bot",
    "automated",
    "openai",
)
MIN_EVIDENCE_QUOTE_CHARS = 20
KNOWN_SOURCE_IDS = frozenset({
    "tenhou_raw_logs",
    "tenhou_old",
    "tenhou_recent",
    "tenhou_to_mjai_dataset",
    "tenhou_to_mjai_code",
    "mortal_code",
    "teacher_local",
    "akochan_code",
    "selfplay_own",
    "majsoul_optional",
    "third_party_mirror_hf",
})

EXIT_ALLOWED = 0
EXIT_DENIED = 1
EXIT_INVALID = 2


class RegistryError(Exception):
    """The registry itself is unusable; fail closed with exit code 2."""


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _valid_iso_date(value: str) -> bool:
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        return False
    try:
        _dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _status_is_consistent(row: dict[str, str]) -> bool:
    values = [row[PURPOSE_COLUMNS[p]] for p in PURPOSES]
    if any(v not in STATUS_VALUES for v in values):
        return False
    return row["status"] == _expected_status(values)


def _expected_status(values: list[str]) -> str:
    """Row-level verdict: any ALLOWED makes the row ALLOWED, else restrictions win."""
    if "ALLOWED" in values:
        return "ALLOWED"
    if "RESTRICTED" in values:
        return "RESTRICTED"
    return "UNKNOWN"


def _validate_evidence(row: dict[str, str], lineno: int, context: str) -> None:
    """An ALLOWED cell must be backed by scope-specific, checkable evidence."""
    problems: list[str] = []
    if not _is_http_url(row["evidence_url"]):
        problems.append("evidence_url must be an http(s) URL")
    if not row["terms_url"] or row["terms_url"].upper() == "UNKNOWN":
        problems.append("terms_url must be a concrete URL (not UNKNOWN)")
    elif not (_is_http_url(row["terms_url"]) or row["terms_url"].startswith("local:")):
        problems.append("terms_url must be http(s) or local:")
    if len(row["evidence_quote"]) < MIN_EVIDENCE_QUOTE_CHARS:
        problems.append(
            f"evidence_quote must be >= {MIN_EVIDENCE_QUOTE_CHARS} characters "
            "(verbatim quote, not a placeholder)"
        )
    if not _valid_iso_date(row["retrieved_at"]):
        problems.append("retrieved_at must be an ISO date (YYYY-MM-DD)")
    elif _dt.date.fromisoformat(row["retrieved_at"]) > _dt.datetime.now(
        _dt.timezone.utc
    ).date():
        problems.append("retrieved_at is in the future (UTC); evidence cannot be dated ahead")
    if not row["privacy_notes"]:
        problems.append("privacy_notes must be non-empty")
    reviewer = row["reviewer"].casefold()
    if not reviewer:
        problems.append("reviewer must be non-empty")
    else:
        for marker in PENDING_REVIEWER_MARKERS:
            if marker in reviewer:
                problems.append(
                    f"reviewer contains pending marker {marker!r}; "
                    "ALLOWED requires a human reviewer, not an agent draft"
                )
                break
        else:
            for token in PENDING_REVIEWER_TOKENS:
                if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", reviewer):
                    problems.append(
                        f"reviewer contains AI/automation marker {token!r}; "
                        "ALLOWED requires a human reviewer"
                    )
                    break
    if problems:
        raise RegistryError(
            f"{context}, line {lineno}: ALLOWED lacks evidence: " + "; ".join(problems)
        )


def _validate_row(row: dict[str, str], lineno: int) -> None:
    if not row["source_id"]:
        raise RegistryError(f"line {lineno}: empty source_id")
    if any(ch.isspace() for ch in row["source_id"]):
        raise RegistryError(f"line {lineno}: source_id must not contain whitespace")
    # Even a blocked row must say where it came from and when it was checked;
    # a blank provenance row is indistinguishable from a typo and is rejected.
    for column in ("origin_url", "retrieved_at", "terms_url", "evidence_quote",
                   "privacy_notes", "reviewer"):
        if not row[column]:
            raise RegistryError(f"line {lineno}: {column} must be non-empty")
    if not _valid_iso_date(row["retrieved_at"]):
        raise RegistryError(
            f"line {lineno}: retrieved_at must be an ISO date (YYYY-MM-DD)"
        )
    if row["status"] not in STATUS_VALUES:
        raise RegistryError(
            f"line {lineno}: status {row['status']!r} is not one of {sorted(STATUS_VALUES)}"
        )
    for purpose in PURPOSES:
        column = PURPOSE_COLUMNS[purpose]
        if row[column] not in STATUS_VALUES:
            raise RegistryError(
                f"line {lineno}: {column} value {row[column]!r} is not one of "
                f"{sorted(STATUS_VALUES)}"
            )
    if not _status_is_consistent(row):
        expected = _expected_status([row[PURPOSE_COLUMNS[p]] for p in PURPOSES])
        raise RegistryError(
            f"line {lineno}: status={row['status']!r} is inconsistent with purpose "
            f"cells; expected {expected!r} (any ALLOWED purpose requires "
            "status=ALLOWED; otherwise restrictions win over UNKNOWN)"
        )
    for purpose in PURPOSES:
        if row[PURPOSE_COLUMNS[purpose]] == "ALLOWED":
            _validate_evidence(row, lineno, f"source {row['source_id']!r}")


def load_registry(path: str | Path) -> dict[str, dict[str, str]]:
    """Parse and validate the registry.  Any defect raises RegistryError."""
    path = Path(path)
    if not path.is_file():
        raise RegistryError(f"registry not found: {path}")

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RegistryError(f"registry {path} is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise RegistryError(f"cannot read registry {path}: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise RegistryError(f"registry {path} is empty (no header)")

    header = list(reader.fieldnames)
    if any(name is None or name != name.strip() for name in header):
        raise RegistryError(f"registry {path} has blank or padded header names")
    if len(set(header)) != len(header):
        raise RegistryError(f"registry {path} has duplicate header names")
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise RegistryError(
            f"registry {path} is missing required columns: {missing}"
        )

    registry: dict[str, dict[str, str]] = {}
    try:
        for lineno, raw in enumerate(reader, start=2):
            if None in raw:
                raise RegistryError(
                    f"registry {path}, line {lineno}: too many fields "
                    "(row has more cells than the header)"
                )
            row = {name: _text(raw.get(name)) for name in header}
            _validate_row(row, lineno)
            source_id = row["source_id"]
            if source_id in registry:
                raise RegistryError(
                    f"registry {path}, line {lineno}: duplicate source_id "
                    f"{source_id!r}; ambiguous rows are rejected"
                )
            registry[source_id] = row
    except csv.Error as exc:
        raise RegistryError(f"registry {path} is not valid CSV: {exc}") from exc

    if not registry:
        raise RegistryError(f"registry {path} has no data rows")

    # A registry that contains some known sources but not all of them is a
    # truncated production registry, not a valid custom fixture.  Reject it
    # globally; a fixture-only registry (no known ids) is still allowed so the
    # gate's own tests can use synthetic sources.
    present_known = set(registry) & KNOWN_SOURCE_IDS
    if present_known and present_known != KNOWN_SOURCE_IDS:
        missing = sorted(KNOWN_SOURCE_IDS - present_known)
        raise RegistryError(
            f"known source(s) missing from the registry: {missing}; "
            "refusing a partial production registry"
        )
    return registry


def check(
    registry: dict[str, dict[str, str]], source: str, purpose: str
) -> tuple[bool, str, dict[str, str] | None]:
    """Return (allowed, reason, row).  Never mutates the registry."""
    if purpose not in PURPOSE_COLUMNS:
        raise RegistryError(f"unknown purpose {purpose!r}")
    row = registry.get(source)
    if row is None:
        if source in KNOWN_SOURCE_IDS:
            raise RegistryError(
                f"known source {source!r} is missing from the registry; fail closed"
            )
        raise RegistryError(f"unknown source {source!r}; not configured = fail closed")

    # load_registry already validates, but check() is also a public entry point.
    # Revalidate the row so a hand-built dict cannot smuggle an ALLOWED purpose
    # past an UNKNOWN/RESTRICTED status.
    try:
        _validate_row(row, 0)
    except RegistryError as exc:
        raise RegistryError(f"source {source!r} row is invalid: {exc}") from exc

    column = PURPOSE_COLUMNS[purpose]
    value = row[column]
    if row["status"] != "ALLOWED":
        return (
            False,
            f"status={row['status']} (need ALLOWED; {column}={value})",
            row,
        )
    if value != "ALLOWED":
        return False, f"{column}={value} (need exactly ALLOWED)", row
    # _validate_row already checked evidence; keep this explicit for callers
    # that bypass load_registry and for a readable reason string.
    try:
        _validate_evidence(row, 0, f"source {source!r}")
    except RegistryError as exc:
        return False, str(exc), row
    return True, "ALLOWED with evidence", row


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    if payload.get("allowed"):
        print(
            "ALLOW source={source} purpose={purpose} status={status} "
            "evidence_url={evidence_url} retrieved_at={retrieved_at}".format(**payload)
        )
    elif payload.get("error"):
        print(
            "ERROR source={source} purpose={purpose} reason={error}".format(**payload),
            file=sys.stderr,
        )
    else:
        print(
            "DENY source={source} purpose={purpose} status={status} reason={reason}".format(
                **payload
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed source-rights gate: exit 0 only when the requested "
            "source/purpose is exactly ALLOWED with evidence."
        )
    )
    parser.add_argument("--source", required=True, help="source_id from the registry")
    parser.add_argument(
        "--purpose", required=True, choices=PURPOSES, help="intended use of the source"
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help=f"registry path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--json", action="store_true", help="machine-readable single-line output"
    )
    args = parser.parse_args(argv)

    base = {"source": args.source, "purpose": args.purpose}
    try:
        registry = load_registry(args.csv)
    except RegistryError as exc:
        payload = dict(base, allowed=False, error=f"malformed_registry: {exc}")
        _emit(payload, args.json)
        return EXIT_INVALID

    try:
        allowed, reason, row = check(registry, args.source, args.purpose)
    except RegistryError as exc:
        payload = dict(base, allowed=False, error=str(exc))
        _emit(payload, args.json)
        return EXIT_INVALID

    assert row is not None
    payload = dict(
        base,
        allowed=allowed,
        reason=reason,
        status=row["status"],
        evidence_url=row["evidence_url"],
        retrieved_at=row["retrieved_at"],
    )
    if not allowed:
        payload["error"] = None
    _emit(payload, args.json)
    return EXIT_ALLOWED if allowed else EXIT_DENIED


if __name__ == "__main__":
    raise SystemExit(main())
