#!/usr/bin/env python3
"""RiichiFormer V2 PR-07R: era / rule drift measurement over the local corpus.

  PYTHONPATH=/path/to/libriichi/release \
  .venv/bin/python scripts/rf_drift.py \
      --manifest /abs/path/source_audit.parquet \
      --out /abs/path/artifacts/riichiformer/drift \
      --report docs/riichiformer/DRIFT_REPORT.md \
      --per-year 400 --seed 20261121 --workers 8

Reads a deterministic per-year sample, parses public events and native replay
decisions, then writes ``drift_cohorts.csv``, ``drift_strata.csv``,
``drift_summary.json`` and ``docs/riichiformer/DRIFT_REPORT.md``.

Safety: read-only, no network, no player identifiers in artefacts (optional
in-memory HMAC only), teacher audit NOT_RUN (no authorised teacher protocol),
all players ``unknown`` provenance.  Failures are counted by stage/reason, never
silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT, REPO_ROOT / "scripts",
              REPO_ROOT / "vendor" / "mortal" / "target" / "release"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from riichiformer.data import canonicalize, drift, split  # noqa: E402

DEFAULT_MANIFEST = "artifacts/riichiformer/audit/source_audit.parquet"
DEFAULT_OUT = "artifacts/riichiformer/drift"
DEFAULT_REPORT = "docs/riichiformer/DRIFT_REPORT.md"

CONFOUNDERS = [
    "platform: the inventory mixes Tenhou archive vintages (`tenhou_old`, "
    "`tenhou_recent`); client/rule changes and archive selection are not adjusted.",
    "rules: per-log rule flags (aka dora, hanchan vs tonpu, renchan/tobi, illegal "
    "action handling) are absent from the inventory and were NOT independently "
    "verified for every log, so rule drift is a candidate explanation, not a "
    "controlled variable.",
    "player pool: player identity is unavailable to this analysis; account rank, "
    "table level, time of day, region and repeated opponents are unknown and are "
    "not conditioned on.",
    "year: windows are fixed calendar partitions; seasonality, archive coverage, "
    "client updates and the January-only 2026 slice are not corrected for.",
    "AI provenance: all players are `unknown`. There is no local confirmed-AI "
    "evidence and no account list is published; absence of evidence is NOT "
    "`confirmed_human`.",
    "selection: the cohort is a deterministic hash-ranked per-year sample, not the "
    "full corpus; confidence intervals reflect sampled games only, not the "
    "data-generating uncertainty of all Tenhou play.",
    "counting: a ron event is a deal-in (multi-ron counts each winner); tsumo "
    "`hora` events carry `target == actor` in this corpus and are excluded from "
    "deal-in counts.",
]


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path)


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_years(text: str) -> set[int]:
    years = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(part))
    return years


def select_sample(manifest: Path, per_year: int, seed: int,
                  years: set[int] | None = None) -> list[dict]:
    """Deterministic per-year sample: rank by SHA-256(seed NUL relpath)."""
    import pyarrow.parquet as pq

    available = set(pq.read_schema(manifest).names)
    columns = [name for name in
               ("root", "relpath", "year", "kind", "invalid_reason", "source_id")
               if name in available]
    rows = pq.read_table(manifest, columns=columns).to_pylist()
    buckets = defaultdict(list)
    for row in rows:
        if row.get("kind", "leaf") != "leaf" or row.get("invalid_reason"):
            continue
        if row.get("year") is None:
            continue
        if years is not None and row["year"] not in years:
            continue
        buckets[row["year"]].append(row)
    selected = []
    for year in sorted(buckets):
        ranked = sorted(
            buckets[year],
            key=lambda row: hashlib.sha256(f"{seed}\x00{row['relpath']}".encode()).hexdigest(),
        )
        selected.extend(ranked[:per_year])
    return selected


def parse_one(row: dict, hmac_key: bytes | None, include_decisions: bool):
    path = Path(row["root"]) / row["relpath"]
    game_id, _ = split.game_id_from_relpath(row["relpath"])
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"read:{type(exc).__name__}"
    try:
        record = drift.parse_game(raw, row["relpath"], game_id, row.get("year"),
                                  hmac_key=hmac_key, include_decisions=include_decisions)
        return record, None
    except canonicalize.CanonicalizeError as exc:
        return None, f"parse:{exc.reason}"
    except Exception as exc:  # noqa: BLE001 - data-quality evidence
        return None, f"parse:{type(exc).__name__}"


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_ci(value, low, high, digits=4):
    if value is None:
        return "n/a"
    return f"{_fmt(value, digits)} [{_fmt(low, digits)}, {_fmt(high, digits)}]"


def _window_rows(cohorts: list[dict]) -> list[dict]:
    by_name = {row["cohort"]: row for row in cohorts
               if row["scope"] == "window" and row["cohort"] in drift.WINDOWS}
    return [by_name[name] for name in drift.WINDOWS if name in by_name]


def _window_strata(strata: list[dict], stratum_type: str) -> list[dict]:
    rows = [row for row in strata if row["stratum_type"] == stratum_type]
    return [row for row in rows if row["stratum"].split("|")[0] in drift.WINDOWS]


def _replay_backend() -> dict:
    try:
        from riichiformer.data import replay

        lib = replay.load_libriichi()
        module = getattr(lib, "__file__", None)
        digest = _sha256(Path(module)) if module and Path(module).is_file() else None
        return {"available": True, "module": module, "sha256": digest}
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def render_report(summary: dict, cohorts: list[dict], strata: list[dict]) -> str:
    windows = _window_rows(cohorts)
    year_rows = [row for row in cohorts if row["scope"] == "year"]
    header = [
        "# Era / rule drift report (PR-07R)",
        "",
        f"- generated: {summary['generated_at_utc']}",
        f"- elapsed: {summary['elapsed_seconds']:.1f} s",
        f"- command: `{summary['command']}`",
        f"- working directory: `{summary['working_directory']}`",
        f"- seed: `{summary['seed']}`; per-year cap: `{summary['per_year']}`; "
        f"year filter: `{summary['years_filter'] or 'all inventory years'}`; "
        f"workers: `{summary['workers']}`",
        f"- sample: {summary['games_parsed']} parsed / {summary['games_selected']} selected "
        f"({summary['games_failed']} failed)",
        f"- replay: {'ENABLED' if summary['with_replay'] else 'DISABLED (--no-replay)'}; "
        f"replayed games: {summary['replay_games']}; replay errors: "
        f"{summary['replay_errors']}",
        f"- manifest: `{summary['manifest']}` (sha256 `{summary['manifest_sha256']}`)",
        f"- platform metadata: inventory sources "
        f"`{summary['sampled_source_ids']}`; per-log rule flags are **UNKNOWN / not "
        f"verified**, so this report does not claim a ruleset id",
        f"- python: `{summary['environment']['python']}`",
    ]
    if summary["replay_backend"].get("available"):
        header.append(
            f"- libriichi: `{summary['replay_backend']['module']}` "
            f"(sha256 `{summary['replay_backend']['sha256']}`)")
        if summary["environment"].get("PYTHONPATH"):
            header.append(f"- PYTHONPATH: `{summary['environment']['PYTHONPATH']}`")
    elif summary["replay_backend"].get("skipped"):
        header.append("- libriichi: not imported (`--no-replay`)")
    else:
        header.append(f"- libriichi: UNAVAILABLE ({summary['replay_backend'].get('error')})")
    lines = header

    lines += [
        "",
        "## Sample and failures",
        "",
        "Failures are counted by stage and reason; no selected game is silently dropped.",
        "",
        "| year | selected | parsed | failed |",
        "|---:|---:|---:|---:|",
    ]
    selected_by_year = summary["sampled_by_year"]
    parsed_by_year = summary["parsed_by_year"]
    for year in sorted(selected_by_year):
        selected = selected_by_year[year]
        parsed = parsed_by_year.get(str(year), 0)
        lines.append(f"| {year} | {selected} | {parsed} | {selected - parsed} |")
    lines.append("")
    if summary["errors"]:
        lines += ["| stage | reason | games |", "|---|---|---:|"]
        for key, count in sorted(summary["errors"].items()):
            stage, _, reason = key.partition(":")
            lines.append(f"| {stage or 'other'} | `{reason or key}` | {count} |")
    else:
        lines.append("Parse/read failures: none.")
    lines.append("")
    if summary["replay_error_reasons"]:
        lines += ["| replay error kind | games |", "|---|---:|"]
        for kind, count in sorted(summary["replay_error_reasons"].items()):
            lines.append(f"| `{kind}` | {count} |")
        lines.append("")
    elif summary["with_replay"]:
        lines += ["Replay failures: none.", ""]

    lines += [
        "## Pre-registered windows",
        "",
        "| window | games | years | months | kyoku | decisions | note |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for window, coverage in summary["window_coverage"].items():
        lines.append(f"| `{window}` | {coverage['games']} | {coverage['years']} | "
                     f"{coverage['months']} | {coverage['kyoku']} | "
                     f"{coverage['decisions']} | {coverage['note']} |")

    lines += [
        "",
        "## Window action/length metrics",
        "",
        "Rates are per kyoku. `dahai`, `chi`, `pon`, `kan` and `ron` are event counts; "
        "`kan` includes daiminkan/ankan/kakan. `decisions` requires replay.",
        "",
        "| window | games | kyoku | decisions | events p50/p90/p99 | dahai/kyoku | "
        "chi/kyoku | pon/kyoku | kan/kyoku | ron/kyoku |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in windows:
        lines.append(
            f"| `{row['window']}` | {row['games']} | {row['kyoku']} | {row['decisions']} | "
            f"{row['events_per_game_p50']}/{row['events_per_game_p90']}/"
            f"{row['events_per_game_p99']} | {_fmt(row['dahai_per_kyoku'])} | "
            f"{_fmt(row['chi_per_kyoku'])} | {_fmt(row['pon_per_kyoku'])} | "
            f"{_fmt(row['kan_per_kyoku'])} | {_fmt(row['ron_per_kyoku'])} |"
        )

    lines += [
        "",
        "## Window rates (95% CI)",
        "",
        "CIs are normal approximations for event counts per kyoku (rates may exceed 1) "
        "and normal-approximation means for points/deal-ins; they are **sampling** "
        "intervals for the selected games, not causal intervals.",
        "",
        "| window | reach/kyoku | calls/kyoku | hora/kyoku | ryukyoku/kyoku | "
        "deal-ins/game | mean hora pts |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in windows:
        lines.append(
            f"| `{row['window']}` | "
            f"{_fmt_ci(row['reach_per_kyoku'], row['reach_ci_low'], row['reach_ci_high'])} | "
            f"{_fmt_ci(row['calls_per_kyoku'], row['calls_ci_low'], row['calls_ci_high'])} | "
            f"{_fmt_ci(row['hora_per_kyoku'], row['hora_ci_low'], row['hora_ci_high'])} | "
            f"{_fmt_ci(row['ryukyoku_per_kyoku'], row['ryukyoku_ci_low'], row['ryukyoku_ci_high'])} | "
            f"{_fmt_ci(row['mean_deal_ins_per_game'], row['deal_in_ci_low'], row['deal_in_ci_high'])} | "
            f"{_fmt_ci(row['mean_hora_points'], row['mean_hora_points_ci_low'], row['mean_hora_points_ci_high'], 1)} |"
        )

    lines += [
        "",
        "## Game termination",
        "",
        "`reached S4/W` is the share of games whose last kyoku is South 4 or a West "
        "extension; the remaining games ended early (e.g. bust). Final-event shares "
        "describe the last hora/ryukyoku of a game.",
        "",
        "| window | reached S4/W | 95% CI | final hora | 95% CI | final ryukyoku | 95% CI |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for row in windows:
        lines.append(
            f"| `{row['window']}` | {_fmt(row['reached_south4_share'])} | "
            f"[{_fmt(row['reached_south4_ci_low'])}, {_fmt(row['reached_south4_ci_high'])}] | "
            f"{_fmt(row['final_hora_share'])} | "
            f"[{_fmt(row['final_hora_ci_low'])}, {_fmt(row['final_hora_ci_high'])}] | "
            f"{_fmt(row['final_ryukyoku_share'])} | "
            f"[{_fmt(row['final_ryukyoku_ci_low'])}, {_fmt(row['final_ryukyoku_ci_high'])}] |"
        )

    if year_rows:
        lines += [
            "",
            "## Per-year basic metrics",
            "",
            "| year | games | kyoku | decisions | events p50/p90/p99 | reach/kyoku | "
            "calls/kyoku | ron/kyoku |",
            "|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
        for row in year_rows:
            lines.append(
                f"| {row['cohort']} | {row['games']} | {row['kyoku']} | {row['decisions']} | "
                f"{row['events_per_game_p50']}/{row['events_per_game_p90']}/"
                f"{row['events_per_game_p99']} | {_fmt(row['reach_per_kyoku'])} | "
                f"{_fmt(row['calls_per_kyoku'])} | {_fmt(row['ron_per_kyoku'])} |"
            )

    lines += [
        "",
        "## Conditional strata (unadjusted, not causal)",
        "",
        "`drift_strata.csv` contains all three stratum types for every window, "
        "including `pre_2019`. Rates below are raw conditional frequencies.",
        "",
        "### window x bakaze",
        "",
        "| stratum | kyoku | reach/kyoku | 95% CI | calls/kyoku | 95% CI |",
        "|---|---:|---|---|---|---|",
    ]
    for row in _window_strata(strata, "window_bakaze"):
        lines.append(
            f"| `{row['stratum']}` | {row['kyoku']} | {_fmt(row['reach_per_kyoku'])} | "
            f"[{_fmt(row['reach_ci_low'])}, {_fmt(row['reach_ci_high'])}] | "
            f"{_fmt(row['calls_per_kyoku'])} | "
            f"[{_fmt(row['calls_ci_low'])}, {_fmt(row['calls_ci_high'])}] |"
        )
    lines += [
        "",
        "### window x bakaze x all_last (bakaze=S, kyoku=4)",
        "",
        "| stratum | kyoku | reach/kyoku | 95% CI | calls/kyoku | 95% CI |",
        "|---|---:|---|---|---|---|",
    ]
    for row in _window_strata(strata, "window_bakaze_last"):
        lines.append(
            f"| `{row['stratum']}` | {row['kyoku']} | {_fmt(row['reach_per_kyoku'])} | "
            f"[{_fmt(row['reach_ci_low'])}, {_fmt(row['reach_ci_high'])}] | "
            f"{_fmt(row['calls_per_kyoku'])} | "
            f"[{_fmt(row['calls_ci_low'])}, {_fmt(row['calls_ci_high'])}] |"
        )
    lines += [
        "",
        "### window x bakaze x reach turn share",
        "",
        "Denominator is all reach declarations in the window/bakaze; the turn is the "
        "declaring player's own discard index (junme). This is **not** a per-kyoku rate.",
        "",
        "| stratum | reaches | share | 95% CI |",
        "|---|---:|---|---|",
    ]
    for row in _window_strata(strata, "window_bakaze_turn"):
        lines.append(
            f"| `{row['stratum']}` | {row['reach']} | {_fmt(row['reach_share'])} | "
            f"[{_fmt(row['reach_ci_low'])}, {_fmt(row['reach_ci_high'])}] |"
        )

    lines += [
        "",
        "## Confounders (not adjusted)",
        "",
    ]
    lines += [f"- {item}" for item in summary["confounders"]]

    lines += [
        "",
        "## Interpretation rules and explicit warning",
        "",
        "- **`recent = stronger` is NOT a conclusion of this report.** The observed "
        "frequency changes are consistent with rules, player-pool, archive-selection "
        "and coverage confounders, all of which are unadjusted here.",
        "- These are **frequency differences**, not demonstrated skill differences. "
        "Any quality claim needs conditioning, a pre-registered hypothesis and "
        "out-of-sample confirmation.",
        "- 2026 is January-only locally; no annual claim is made for 2026.",
    ]
    if summary["years_filter"]:
        lines.append(
            f"- This run is explicitly filtered to years `{summary['years_filter']}` "
            f"(the span of the pre-registered windows); inventory rows outside that "
            f"range were not sampled and no failure is implied by their absence.")
    lines += [
        "- The sample is deterministic but partial; CIs reflect sampled games, not "
        "the whole corpus or the data-generating process.",
        "- Multi-ron counts each winner; tsumo is not a deal-in.",
        "- 2025/2026 zip members were plaintext and extracted copies are gzip, so "
        "parsing used compression magic sniffing rather than file extensions.",
        "- No player identifiers or account lists are produced; optional HMAC "
        "aggregation is in-memory only.",
        "",
        "## Teacher / AI provenance",
        "",
        "- **Teacher audit: NOT_RUN.** No authorised teacher-evaluation "
        "protocol/config exists; local Mortal checkpoints have unverified "
        "distillation rights and were not used here.",
        "- **AI provenance: all players `unknown`.** No local confirmed-AI evidence "
        "and no account list is published. Absence of evidence is not "
        "`confirmed_human`.",
        "",
        "## Artefacts and reproduction",
        "",
        "| file | rows | sha256 |",
        "|---|---:|---|",
    ]
    for name, meta in summary["artifacts"].items():
        lines.append(f"| `{name}` | {meta.get('rows', 'n/a')} | `{meta['sha256']}` |")
    lines += [
        "",
        "To reproduce, run the exact command above from the working directory after "
        "ensuring the recorded `libriichi` module path is importable. The per-year "
        "hash-ranked sample is fixed by the seed and manifest hash.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Era/rule drift measurement (PR-07R)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--per-year", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--years", default=None,
                    help="optional year filter, e.g. 2019-2026 or 2019,2025")
    ap.add_argument("--no-replay", action="store_true",
                    help="skip native replay (no decision counts; for fast audits/tests)")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    args = ap.parse_args(argv)

    started = time.perf_counter()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    command = " ".join(shlex.quote(part) for part in
                       [sys.executable, "scripts/rf_drift.py", *raw_argv])
    repo = Path(args.repo_root).resolve()
    manifest = _resolve(repo, args.manifest)
    out = _resolve(repo, args.out)
    report = _resolve(repo, args.report)
    years_filter = parse_years(args.years) if args.years else None
    out.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    try:
        rows = select_sample(manifest, args.per_year, args.seed, years_filter)
    except Exception as exc:  # noqa: BLE001 - startup failure
        print(f"rf_drift: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print("rf_drift: no rows selected", file=sys.stderr)
        return 2

    hmac_key = None
    try:
        from riichiformer.data.player_quality import hmac_key_from_env
        hmac_key = hmac_key_from_env()
    except Exception:  # noqa: BLE001 - optional; unique player counts stay null
        pass

    replay_backend = {"available": False, "skipped": True}
    if not args.no_replay:
        replay_backend = _replay_backend()
        if not replay_backend.get("available"):
            print(f"rf_drift: WARNING libriichi unavailable: "
                  f"{replay_backend.get('error')}", file=sys.stderr)

    errors = Counter()
    games = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for record, error in pool.map(
            lambda row: parse_one(row, hmac_key, not args.no_replay), rows
        ):
            if error:
                errors[error] += 1
            else:
                games.append(record)

    cohorts = drift.aggregate_cohorts(games)
    strata = drift.aggregate_strata(games)
    coverage = drift.cohort_windows_with_coverage(games)
    cohorts_path = out / "drift_cohorts.csv"
    strata_path = out / "drift_strata.csv"
    _write_csv(cohorts_path, cohorts)
    _write_csv(strata_path, strata)
    artifacts = {
        "drift_cohorts.csv": {"path": str(cohorts_path), "rows": len(cohorts),
                              "sha256": _sha256(cohorts_path)},
        "drift_strata.csv": {"path": str(strata_path), "rows": len(strata),
                             "sha256": _sha256(strata_path)},
    }
    sampled_by_year = Counter(row["year"] for row in rows)
    parsed_by_year = Counter(game.year for game in games if game.year is not None)
    replay_errors = Counter(game.replay_error for game in games if game.replay_error)
    summary = {
        "schema": "rf-drift-summary/2",
        "generated_at_utc": _utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": command,
        "working_directory": str(repo),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "per_year": args.per_year,
        "seed": args.seed,
        "years_filter": sorted(years_filter) if years_filter else None,
        "workers": args.workers,
        "games_selected": len(rows),
        "games_parsed": len(games),
        "games_failed": sum(errors.values()),
        "sampled_by_year": {str(year): count
                            for year, count in sorted(sampled_by_year.items())},
        "parsed_by_year": {str(year): count
                           for year, count in sorted(parsed_by_year.items())},
        "sampled_source_ids": dict(sorted(Counter(
            row.get("source_id") or "unknown" for row in rows).items())),
        "errors": dict(sorted(errors.items())),
        "with_replay": not args.no_replay,
        "replay_games": sum(1 for game in games if game.decisions is not None),
        "replay_errors": sum(1 for game in games if game.replay_error),
        "replay_error_reasons": dict(sorted(replay_errors.items())),
        "replay_backend": replay_backend,
        "hmac_players_available": hmac_key is not None,
        "window_coverage": coverage,
        "cohorts": cohorts,
        "strata": strata,
        "artifacts": artifacts,
        "environment": {
            "python": platform.python_version(),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "sys_path_head": [str(part) for part in sys.path[:5]],
        },
        "ruleset": "per_log_ruleset_NOT_VERIFIED",
        "confounders": CONFOUNDERS,
        "teacher_audit": "NOT_RUN_no_authorized_teacher_protocol",
        "ai_provenance": "all_unknown",
        "notes": [
            "frequency differences are not skill differences",
            "recent = stronger is NOT a conclusion",
            "2026 is January-only locally",
            "sample is deterministic per year, not the full corpus",
            "tsumo hora has target == actor and is not a deal-in",
        ],
    }
    summary_path = out / "drift_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report.write_text(render_report(summary, cohorts, strata), encoding="utf-8")
    report_sha = _sha256(report)
    summary["report"] = {"path": str(report), "sha256": report_sha}
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"rf_drift: games={len(games)}/{len(rows)} failed={sum(errors.values())} "
          f"errors={dict(errors)} replay_errors={dict(replay_errors)}")
    print(f"rf_drift: windows="
          f"{ {w: c['games'] for w, c in coverage.items()} }")
    print(f"rf_drift: elapsed={summary['elapsed_seconds']:.1f}s "
          f"cohorts={cohorts_path} ({len(cohorts)} rows, "
          f"sha256={artifacts['drift_cohorts.csv']['sha256']})")
    print(f"rf_drift: strata={strata_path} ({len(strata)} rows, "
          f"sha256={artifacts['drift_strata.csv']['sha256']})")
    print(f"rf_drift: summary={summary_path} report={report} "
          f"(sha256={report_sha})")
    return 0 if games else 1


if __name__ == "__main__":
    sys.exit(main())
