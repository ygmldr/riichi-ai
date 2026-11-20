#!/usr/bin/env python3
"""RiichiFormer V2 PR-06: player HMAC + empirical-Bayes quality audit.

  # player-linked (requires a local secret key; the key is never written)
  RF_PLAYER_HMAC_KEY="$(python3 -c 'import secrets;print(secrets.token_hex(32))')" \
    .venv/bin/python scripts/rf_quality.py --manifest artifacts/riichiformer/split/split_manifest_L.parquet \
      --sample 5000 --out artifacts/riichiformer/quality

  # basic audit without the key: no player identifiers, no names
  .venv/bin/python scripts/rf_quality.py --manifest ... --sample 500

Outputs: quality_scores.parquet (HMAC-only), quality_audit.md, quality_summary.json.
Only the manifest's ``train`` split is read; history is restricted to those games.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from riichiformer.data import canonicalize, dedup, player_quality as pq, replay  # noqa: E402

DEFAULT_MANIFEST = "artifacts/riichiformer/split/split_manifest_L.parquet"
DEFAULT_OUT = "artifacts/riichiformer/quality"
DEFAULT_REPORT = "docs/riichiformer/quality_audit.md"


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path)


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_train_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq_arrow

    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path} (run PR-05 first)")
    rows = pq_arrow.read_table(path, columns=[
        "game_id", "source_id", "year", "split", "relpath", "root",
    ]).to_pylist()
    return [row for row in rows if row["split"] == "train"]


def parse_game(row: dict, key: bytes | None):
    path = Path(row["root"]) / row["relpath"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return row, [], f"read:{type(exc).__name__}"
    if key is None:
        # Basic mode: ranks are still useful for an audit, names are not touched.
        try:
            compression = canonicalize.detect_compression_bytes(raw)
            payload = canonicalize.decode_log_bytes(raw, compression)
            events = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
            scores = replay.final_scores_from_events(events)
        except Exception as exc:  # noqa: BLE001
            return row, [], f"decode:{type(exc).__name__}"
        if scores is None:
            return row, [], "missing_scores"
        return row, pq.ranks_from_scores(scores), None
    appearances, error = pq.extract_appearances(
        raw, row["relpath"], row["game_id"], row.get("year"), row.get("source_id"), key
    )
    return row, appearances, error


def quality_schema():
    import pyarrow as pa

    return pa.schema([
        ("game_id", pa.string()), ("seat", pa.int8()), ("player_hmac", pa.string()),
        ("match_time", pa.string()), ("year", pa.int16()), ("source_id", pa.string()),
        ("n_prior_games", pa.int32()), ("mean_rank_prior", pa.float64()),
        ("q", pa.float64()), ("confidence", pa.float64()),
        ("insufficient_history", pa.bool_()), ("temporal", pa.bool_()),
        ("target_rank_for_eval", pa.int8()), ("is_last_appearance", pa.bool_()),
        ("top25_seat", pa.bool_()),
    ])


def write_quality(rows: list[dict], path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq_arrow

    schema = quality_schema()
    table = pa.Table.from_pydict(
        {name: [row.get(name) for row in rows] for name in schema.names}, schema=schema
    )
    pq_arrow.write_table(table, path, compression="zstd")
    return table.num_rows


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_audit(summary: dict, reliability: list[dict], errors: dict) -> str:
    lines = [
        "# Player quality audit (PR-06)",
        "",
        f"- generated: {summary['generated_at_utc']}",
        f"- manifest: `{summary['manifest']}` (sha256 `{summary['manifest_sha256']}`)",
        f"- split filter: `train`; sample: {summary['sample_size']} games (seed {summary['seed']})",
        f"- player-linked: **{summary['player_linked']}**",
        f"- prior mean rank={summary['prior']}, k={summary['k']}, min_games={summary['min_games']}",
        "",
        "## Coverage",
        "",
        f"- games parsed: {summary['games_parsed']} / {summary['games_selected']}"
        f" (errors {sum(errors.values())}: {errors or '{}'})",
        f"- seat rows: {summary['rows']}; unique players (HMAC): {summary['unique_players']}",
        f"- insufficient history (<{summary['min_games']} prior games): {summary['insufficient_rows']}"
        f" ({_fmt(summary['insufficient_fraction'])} of rows)",
        f"- top25 seat rows: {summary['top25_rows']} (threshold q={_fmt(summary['top25_threshold'])})",
        "",
        "## q distribution (eligible rows only)",
        "",
    ]
    quantiles = summary.get("q_quantiles") or {}
    lines += [
        "| p10 | p25 | p50 | p75 | p90 |",
        "|---:|---:|---:|---:|---:|",
        f"| {_fmt(quantiles.get('p10'))} | {_fmt(quantiles.get('p25'))} "
        f"| {_fmt(quantiles.get('p50'))} | {_fmt(quantiles.get('p75'))} "
        f"| {_fmt(quantiles.get('p90'))} |",
        "",
        "## Reliability (players' last appearance: prior q vs that game's rank)",
        "",
    ]
    if reliability:
        lines += ["| bucket | players | mean rank | win rate |", "|---|---:|---:|---:|"]
        for row in reliability:
            lines.append(f"| {row['bucket']} | {row['players']} | {row['mean_target_rank']} "
                         f"| {row['target_win_rate']} |")
    else:
        lines.append("No player reached the minimum history in this sample; **no reliability claim**.")
    lines += [
        "",
        "## Privacy / metadata",
        "",
        "- Names are read only in memory; artefacts contain **HMAC-SHA256** identifiers only.",
        "- `RF_PLAYER_HMAC_KEY` is supplied via environment and is never written or logged.",
        "- No local annual player database / dan / R / rating was found; those fields are `null`/unknown.",
        "- No AI-account inference is made; absence of evidence is `unknown`, not `human`.",
        "",
        "## Limitations",
        "",
        f"- sample-based: {summary['sample_size']} train games, not the full 2.5 M corpus;",
        "  history is limited to games present in this sample;",
        "- q is a shrunk mean rank from prior **train** games only; it is a candidate feature,",
        "  not a certified player strength; `target_rank_for_eval` is never used to compute q;",
        "- labels are excluded from history by time (`temporal=true`) or marked `temporal=false`;",
        "- top25 is a candidate reweighting flag; the other three seats in a game are not marked.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Player HMAC + empirical-Bayes quality audit (PR-06)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prior", type=float, default=pq.DEFAULT_PRIOR_MEAN_RANK)
    ap.add_argument("--k", type=int, default=pq.DEFAULT_K)
    ap.add_argument("--min-games", type=int, default=pq.DEFAULT_MIN_GAMES)
    ap.add_argument("--top-quantile", type=float, default=pq.DEFAULT_TOP_QUANTILE)
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    manifest = _resolve(repo, args.manifest)
    out = _resolve(repo, args.out)
    report = _resolve(repo, args.report)
    out.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    try:
        rows = load_train_rows(manifest)
        rows = dedup.stratified_sample(rows, max_total=args.sample, seed=args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"rf_quality: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        key = pq.hmac_key_from_env()
        key_error = None
    except pq.HmacKeyMissing as exc:
        key = None
        key_error = str(exc)

    errors = Counter()
    parsed = 0
    rank_counts = Counter()
    appearances = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for row, payload, error in pool.map(lambda r: parse_game(r, key), rows):
            if error:
                errors[error] += 1
                continue
            parsed += 1
            if key is None:
                for rank in payload:
                    rank_counts[str(rank)] += 1
            else:
                appearances.extend(payload)

    rows_out = []
    reliability = []
    summary_extra = {}
    if key is not None:
        history_ids = {row["game_id"] for row in rows}
        rows_out = pq.compute_quality(
            appearances, prior=args.prior, k=args.k, min_games=args.min_games,
            top_quantile=args.top_quantile, history_game_ids=history_ids,
        )
        pq.assert_no_names(rows_out)
        summary_extra = pq.quality_summary(rows_out, k=args.k, prior=args.prior,
                                           min_games=args.min_games)
        reliability = pq.reliability_table(rows_out)
        write_quality(rows_out, out / "quality_scores.parquet")
    else:
        write_quality([], out / "quality_scores.parquet")

    summary = {
        "schema": pq.SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split_filter": "train",
        "sample_size": len(rows),
        "seed": args.seed,
        "games_selected": len(rows),
        "games_parsed": parsed,
        "errors": dict(errors),
        "player_linked": key is not None,
        "key_error": key_error,
        "prior": args.prior,
        "k": args.k,
        "min_games": args.min_games,
        "top_quantile": args.top_quantile,
        "rows": len(rows_out),
        "unique_players": len({row["player_hmac"] for row in rows_out}),
        "insufficient_rows": sum(1 for row in rows_out if row["insufficient_history"]),
        "insufficient_fraction": round(
            sum(1 for row in rows_out if row["insufficient_history"]) / len(rows_out), 4
        ) if rows_out else None,
        "top25_rows": sum(1 for row in rows_out if row["top25_seat"]),
        "top25_threshold": summary_extra.get("top25_threshold"),
        "q_quantiles": summary_extra.get("q_quantiles"),
        "n_prior_histogram": summary_extra.get("n_prior_histogram"),
        "temporal_rows": summary_extra.get("temporal_rows"),
        "non_temporal_rows": summary_extra.get("non_temporal_rows"),
        "years": summary_extra.get("years"),
        "sources": summary_extra.get("sources"),
        "rank_distribution_basic_mode": dict(sorted(rank_counts.items())) if key is None else None,
        "metadata": {
            "local_player_database": False,
            "dan_rating_available": False,
            "name_source": "start_game.names",
            "note": "no dan/R/rating/date metadata; nothing inferred from names",
        },
        "reliability": reliability,
        "limitations": [
            "sample-based; full corpus not processed",
            "history limited to sampled train games",
            "HMAC key is ephemeral and not stored; scores are not reproducible without it",
            "no AI-account inference",
        ],
    }
    (out / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report.write_text(render_audit(summary, reliability, dict(errors)), encoding="utf-8")
    print(f"rf_quality: player_linked={key is not None} games={parsed}/{len(rows)} "
          f"rows={len(rows_out)} players={summary['unique_players']} "
          f"insufficient={summary['insufficient_rows']} top25={summary['top25_rows']}")
    if key_error:
        print(f"  player-linked audit disabled: {key_error}")
    print(f"  scores={out / 'quality_scores.parquet'} report={report}")
    return 0 if parsed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
