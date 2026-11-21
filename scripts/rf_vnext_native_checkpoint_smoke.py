#!/usr/bin/env python3
"""Native 4-game smoke using a trained VNext checkpoint (no strength claim)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.eval.m5d_arena import run_arena  # noqa: E402
from riichiformer.eval.mortal_anchor import (  # noqa: E402
    build_baseline_registry, write_baseline_registry,
)
from riichiformer.vnext.runtime import load_vnext_checkpoint  # noqa: E402
from scripts.rf_vnext_native_smoke import VNextArenaBackend  # noqa: E402


class TrainedBackend(VNextArenaBackend):
    name = "vnext-bc-trained"

    def as_dict(self) -> dict:
        return {
            "mode": "vnext_trained_bc",
            "name": self.name,
            "inference_mode": "deterministic_argmax",
            "device": "cpu",
            "synthetic_fixture": False,
            "strength_claim": "NONE",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="runs/vnext_native_trained_smoke")
    parser.add_argument("--seed-start", type=int, default=991000)
    parser.add_argument("--seed-key", default="0x5EEDF00D5EEDF00E")
    parser.add_argument("--sets", type=int, default=1)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty smoke dir {out}")
    out.mkdir(parents=True, exist_ok=True)

    model = load_vnext_checkpoint(args.checkpoint)
    checkpoint_sha = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    registry_path = out / "smoke_registry.yaml"
    registry = build_baseline_registry(repo=REPO_ROOT)
    write_baseline_registry(registry_path, registry)

    backend = TrainedBackend(model)
    run = run_arena(
        seed_key=args.seed_key,
        seed_start=args.seed_start,
        seed_count=args.sets,
        candidate="rf",
        candidate_backend=backend,
        registry_path=registry_path,
        out_dir=out / "arena",
        log_dir=out / "arena" / "games",
        progress=False,
    )
    assert run.games == 4 * args.sets
    assert run.candidate.get("rf_candidate") == "RUN"
    assert run.candidate.get("mode") == "vnext_trained_bc"
    assert int(run.candidate.get("illegal", -1)) == 0
    assert int(run.candidate.get("fallback", -1)) == 0
    result = {
        "status": "PASS",
        "kind": "native_vnext_trained_checkpoint_smoke",
        "synthetic_fixture": False,
        "strength_claim": "NONE",
        "games": run.games,
        "seed_window": [run.seed_start, run.seed_start + run.seed_count],
        "seed_key": hex(run.seed_key),
        "rank_counts": list(run.rank_counts),
        "backend_decisions": backend.decisions,
        "candidate": run.candidate,
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": checkpoint_sha,
        },
        "native": {
            "libriichi_build_sha256": registry["libriichi_build_sha256"],
            "libriichi_git_sha": registry["libriichi_git_sha"],
            "patch_manifest_hash": registry["patch_manifest_hash"],
            "registry_hash": run.registry_hash,
        },
        "not_run": ["multi_kan_native_end_to_end"],
        "note": "four games, loss-trained checkpoint; integration smoke only, no strength result",
    }
    (out / "smoke_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
