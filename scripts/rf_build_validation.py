#!/usr/bin/env python3
"""PR-45 CLI: build an independent validation corpus and its diagnostic shard.

Steps (all CPU, no GPU, no network):

1. validate the fresh wall window and ``seed_key`` against B0 + every consumed
   evaluation window;
2. play the window with the pinned native arena (four-seat rotation, weak local
   heuristic bots);
3. prove game-id / seed / generator disjointness (JSON intersection report);
4. seal the corpus into an ``rf-shard/3`` with ``purpose=diagnostic``,
   ``source_kind=arena_selfplay``, ``ruleset_id=mortal_arena_pinned`` and
   R02/R25/R26/R29 provenance;
5. audit visibility, row hashes and score conservation, and write the shard
   report next to the shard.

Exit codes: 0 complete, 2 a contract/rights refusal, 1 an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.policy.config import (  # noqa: E402
    PolicyConfigError,
    load_policy_config,
)
from riichiformer.selfplay.bots import default_lineup  # noqa: E402
from riichiformer.selfplay.provenance import ProvenanceError, write_json  # noqa: E402
from riichiformer.validation.corpus import (  # noqa: E402
    MAX_VALIDATION_GAMES,
    MIN_VALIDATION_GAMES,
    TARGET_VALIDATION_GAMES,
    ValidationCorpusError,
    ValidationWindow,
    generate_seed_key,
    run_validation_selfplay,
    validation_game_ids,
)
from riichiformer.validation.disjoint import (  # noqa: E402
    build_intersection_report,
)
from riichiformer.validation.shard_build import (  # noqa: E402
    ValidationShardError,
    ValidationShardSpec,
    build_validation_shard,
)

TASK = "PR-45"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PR-45 independent native self-play validation corpus + shard")
    parser.add_argument("--out", required=True,
                        help="fresh corpus directory (must not exist or be empty)")
    parser.add_argument("--shard-out", default=None,
                        help="diagnostic shard path (required unless --audit-only)")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--games", type=int, required=True,
                        help=f"multiple of 4 in [{MIN_VALIDATION_GAMES},"
                             f"{MAX_VALIDATION_GAMES}]")
    parser.add_argument("--seed-key", type=int, default=None,
                        help="fresh 64-bit key; omitted => generated with "
                             "secrets.randbits(64)")
    parser.add_argument("--lineup", default=",".join(default_lineup()))
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--b0-artifact", default=None,
                        help="B0 sample parquet/shard/JSON (default: auto-discover)")
    parser.add_argument("--games-dir", default=None,
                        help="existing engine log directory (default <out>/games)")
    parser.add_argument("--skip-selfplay", action="store_true",
                        help="reuse --games-dir instead of playing new games")
    parser.add_argument("--audit-only", action="store_true",
                        help="validate arguments/window/B0 disjointness; no games, "
                             "no shard")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _parse_lineup(text: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(names) != 4:
        raise ValidationCorpusError(f"--lineup needs four bot names, got {names}")
    return names


def _corpus_game_ids(out: Path, games_dir: Path) -> list[str]:
    manifest = out / "validation_corpus.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return [str(game_id) for game_id in payload["game_ids"]]
    return validation_game_ids(games_dir)


def run(args) -> dict:
    if args.games < MIN_VALIDATION_GAMES or args.games > MAX_VALIDATION_GAMES \
            or args.games % 4 != 0:
        raise ValidationCorpusError(
            f"--games must be a multiple of 4 in "
            f"[{MIN_VALIDATION_GAMES},{MAX_VALIDATION_GAMES}], got {args.games}")
    policy = load_policy_config(args.policy_config)
    if policy.profile != "operator_private_research":
        raise ValidationCorpusError(
            "validation generation requires the operator_private_research profile")
    if policy.release_allowed or policy.network_upload_allowed:
        raise ValidationCorpusError(
            "the validation policy must keep release/upload disabled")
    if "selfplay_own" not in policy.allowed_local_sources:
        raise ValidationCorpusError(
            "policy allowlist does not cover the selfplay_own source")
    out = Path(args.out)
    games_dir = Path(args.games_dir) if args.games_dir else out / "games"
    manifest_path = out / "validation_corpus.json"
    if args.skip_selfplay and manifest_path.exists():
        corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        corpus_window = corpus_manifest["window"]
        if int(corpus_window["seed_start"]) != args.seed_start \
                or int(corpus_window["games"]) != args.games:
            raise ValidationCorpusError(
                "existing corpus window "
                f"[{corpus_window['seed_start']},{corpus_window['seed_start'] + corpus_window['walls']})"
                f"/{corpus_window['games']} games does not match --seed-start "
                f"{args.seed_start} --games {args.games}")
        if args.seed_key is not None and int(corpus_window["seed_key"]) != args.seed_key:
            raise ValidationCorpusError(
                "existing corpus seed_key does not match --seed-key")
        seed_key = int(corpus_window["seed_key"])
    else:
        seed_key = args.seed_key if args.seed_key is not None else generate_seed_key()
    window = ValidationWindow(seed_start=args.seed_start, games=args.games,
                              seed_key=int(seed_key))
    lineup = _parse_lineup(args.lineup)
    plan = {
        "task": TASK,
        "window": window.as_dict(),
        "lineup": list(lineup),
        "policy": {"profile": policy.profile, "hash": policy.policy_hash,
                   "version": policy.policy_version},
        "audit_only": bool(args.audit_only),
        "skip_selfplay": bool(args.skip_selfplay),
        "seed_key_generated": args.seed_key is None,
    }
    if args.audit_only:
        if not args.skip_selfplay:
            raise ValidationCorpusError(
                "--audit-only does not play games; pass --skip-selfplay with "
                "--games-dir to audit an existing corpus")
        if not games_dir.exists():
            raise ValidationCorpusError(f"--games-dir not found: {games_dir}")
        game_ids = _corpus_game_ids(out, games_dir)
        report = build_intersection_report(
            seed_start=window.seed_start, games=window.games,
            seed_key=window.seed_key, validation_game_ids=game_ids,
            generator_sha256=_generator_sha(out, games_dir),
            b0_artifact=args.b0_artifact)
        report["plan"] = plan
        _print_report(report, args)
        return {"plan": plan, "intersection": report, "shard": None}

    if args.skip_selfplay:
        if not games_dir.exists():
            raise ValidationCorpusError(f"--games-dir not found: {games_dir}")
        corpus = None
    else:
        corpus = run_validation_selfplay(
            out, window, lineup=lineup,
            progress=None if args.quiet else (lambda message: print(
                f"PR45_PROGRESS {message}", flush=True)))
    game_ids = _corpus_game_ids(out, games_dir)
    generator_sha = _generator_sha(out, games_dir)
    report = build_intersection_report(
        seed_start=window.seed_start, games=window.games,
        seed_key=window.seed_key, validation_game_ids=game_ids,
        generator_sha256=generator_sha, b0_artifact=args.b0_artifact)
    report["plan"] = plan
    intersection_path = Path(args.json_out) if args.json_out else \
        (out / "intersection.json")
    write_json(intersection_path, report)
    if report["ok"] is not True:
        raise ValidationCorpusError(
            "disjointness is not proven (report: "
            f"{intersection_path}); refusing to build a validation shard")
    if not args.shard_out:
        if corpus is not None:
            print(f"PR45_CORPUS games={corpus['games']} decisions="
                  f"{corpus['decisions']} seed_key=0x{window.seed_key:016x} "
                  f"window={window.window} generator={generator_sha}")
        print(f"PR45_INTERSECTION ok=True report={intersection_path}")
        print("PR45_NO_SHARD --shard-out not requested")
        return {"plan": plan, "intersection": report, "shard": None}
    spec = ValidationShardSpec(
        seed_start=window.seed_start, games=window.games,
        seed_key=window.seed_key, ruleset_id=window.ruleset_id,
        lineup=lineup, generator_sha256=generator_sha)
    shard_report = build_validation_shard(
        games_dir=games_dir, out_shard=args.shard_out, spec=spec, policy=policy)
    shard_report["intersection"] = {
        "path": str(intersection_path), "ok": report["ok"],
        "b0_intersection": report["checks"]["b0_game_ids"]["game_id_intersection"],
        "seed_window_ok": report["checks"]["seed_window"]["ok"],
    }
    shard_report["target_games"] = TARGET_VALIDATION_GAMES
    shard_report["shortfall"] = (
        None if TARGET_VALIDATION_GAMES[0] <= window.games else
        f"games {window.games} < target {TARGET_VALIDATION_GAMES[0]}")
    report_path = Path(str(args.shard_out) + ".validation.json")
    write_json(report_path, shard_report)
    if corpus is not None:
        print(f"PR45_CORPUS games={corpus['games']} decisions={corpus['decisions']} "
              f"seed_key=0x{window.seed_key:016x} window={window.window} "
              f"generator={generator_sha}")
    print(f"PR45_INTERSECTION ok=True b0_intersection="
          f"{report['checks']['b0_game_ids']['game_id_intersection']} "
          f"window={window.window} report={intersection_path}")
    print(f"PR45_SHARD path={args.shard_out} sha256={shard_report['shard_sha256']} "
          f"games={shard_report['game_count']} "
          f"decisions={shard_report['decision_count']} purpose=diagnostic "
          f"source_kind=arena_selfplay ruleset=mortal_arena_pinned "
          f"deviations={','.join(shard_report['metadata']['ruleset_deviations'])} "
          f"policy_hash={policy.policy_hash}")
    if shard_report["shortfall"]:
        print(f"PR45_SHORTFALL {shard_report['shortfall']}")
    print(f"PR45_DONE report={report_path}")
    return {"plan": plan, "intersection": report, "shard": shard_report}


def _generator_sha(out: Path, games_dir: Path) -> str:
    manifest = out / "validation_corpus.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return str(payload["generator_sha256"])
    provenance = out / "provenance.json"
    if provenance.exists():
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        return str(payload["generator"]["code_sha256"])
    from riichiformer.selfplay.provenance import generator_fingerprint

    return generator_fingerprint()["code_sha256"]


def _print_report(report: dict, args) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValidationCorpusError, ValidationShardError, ProvenanceError,
            PolicyConfigError) as exc:
        print(f"PR45_REFUSED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"PR45_FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
