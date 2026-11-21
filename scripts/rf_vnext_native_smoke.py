#!/usr/bin/env python3
"""PR-05 native VNext smoke: one synthetic checkpoint, four native arena games.

This is an integration smoke, not a strength experiment and not a training run.
It builds a *randomly initialised* VNext model, saves it through the real
``save_vnext_checkpoint`` ABI, loads it through ``load_vnext_checkpoint``, and
lets the native ``libriichi.arena.OneVsThree`` driver play one complete set
(four seat rotations) against the frozen local M5D-R0 anchor.

The generated registry is a smoke-only registry: it records this worktree's
native build/patch identity so the frozen M5D registry is never rewritten.  The
run must be labelled ``synthetic_fixture`` and must never be cited as strength
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.eval.m5d_arena import (  # noqa: E402
    RFMjaiLogEngine, run_arena,
)
from riichiformer.eval.mortal_anchor import (  # noqa: E402
    build_baseline_registry, write_baseline_registry,
)
from riichiformer.schemas.action_v3 import ActionPhase  # noqa: E402
from riichiformer.vnext.contracts import VNextConfig  # noqa: E402
from riichiformer.vnext.model import VNextPolicy  # noqa: E402
from riichiformer.vnext.runtime import (  # noqa: E402
    VNextPolicyAdapter, load_vnext_checkpoint,
)
from riichiformer.vnext.train import (  # noqa: E402
    VNextTrainConfig, make_optimizer, save_vnext_checkpoint,
)
from riichiformer.train.checkpoint import config_hash  # noqa: E402


class VNextArenaBackend:
    """Cacheless VNext adapter with one per-game stream, keyed by POV."""

    name = "vnext-synthetic"

    def __init__(self, model: VNextPolicy):
        self.model = model.eval()
        self.adapter: VNextPolicyAdapter | None = None
        self.decisions = 0
        self.kan_selections = 0

    def start_game(self, index: int, pov: int) -> None:
        self.adapter = VNextPolicyAdapter(
            self.model, match_id=f"native-smoke:{int(index)}", pov=int(pov))

    def act(self, decision, meta) -> int:
        if self.adapter is None or self.adapter.pov != int(meta.pov):
            self.adapter = VNextPolicyAdapter(
                self.model, match_id=f"native-smoke:{int(meta.pov)}",
                pov=int(meta.pov))
        action = self.adapter.act(decision)
        self.decisions += 1
        if decision.phase is ActionPhase.KAN_SELECTION:
            self.kan_selections += 1
        return action

    def as_dict(self) -> dict:
        return {
            "mode": "vnext_synthetic",
            "name": self.name,
            "inference_mode": "deterministic_argmax",
            "device": "cpu",
            "synthetic_fixture": True,
            "strength_claim": "NONE",
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_checkpoint(out: Path, *, seed: int) -> Path:
    torch.manual_seed(seed)
    config = VNextConfig(
        d_model=64, n_layers=1, n_heads=4, ffn_hidden=128,
        cross_attn_heads=4, dropout=0.0, max_positions=1024)
    model = VNextPolicy(config)
    train_config = VNextTrainConfig(learning_rate=1e-4, seed=seed)
    optimizer = make_optimizer(model, train_config)
    provenance = {
        "source_id": "synthetic_fixture",
        "split_manifest_sha256": hashlib.sha256(b"synthetic-split").hexdigest(),
        "canonical_manifest_sha256": hashlib.sha256(b"synthetic-manifest").hexdigest(),
        "canonical_index_sha256": hashlib.sha256(b"synthetic-index").hexdigest(),
        "research_policy": "synthetic_no_rights",
        "research_policy_hash": hashlib.sha256(b"synthetic-policy").hexdigest(),
    }
    path = out / "synthetic_vnext.pt"
    sha = save_vnext_checkpoint(
        path, model=model, optimizer=optimizer, config=train_config,
        counters={"step": 0, "tokens": 0, "decisions": 0},
        provenance=provenance)
    assert sha == _sha256(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/vnext_native_smoke")
    parser.add_argument("--seed-start", type=int, default=990000)
    parser.add_argument("--seed-key", default="0x5EEDF00D5EEDF00D")
    parser.add_argument("--sets", type=int, default=1)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty smoke dir {out}")
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = _make_checkpoint(out, seed=20261122)
    model = load_vnext_checkpoint(checkpoint)
    model_config_hash = config_hash(model.config.as_dict())

    registry_path = out / "smoke_registry.yaml"
    registry = build_baseline_registry(repo=REPO_ROOT)
    write_baseline_registry(registry_path, registry)

    backend = VNextArenaBackend(model)
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

    expected_games = 4 * int(args.sets)
    assert run.games == expected_games, (run.games, expected_games)
    assert run.candidate.get("rf_candidate") == "RUN"
    assert run.candidate.get("mode") == "vnext_synthetic"
    assert run.candidate.get("self_control_only") is False
    assert int(run.candidate.get("illegal", -1)) == 0
    assert int(run.candidate.get("fallback", -1)) == 0
    assert int(run.candidate.get("kan_selections", -1)) == backend.kan_selections

    result = {
        "status": "PASS",
        "kind": "native_vnext_smoke",
        "not_run": (["multi_kan_native_end_to_end"]
                    if backend.kan_selections == 0 else []),
        "synthetic_fixture": True,
        "strength_claim": "NONE",
        "games": run.games,
        "seed_window": [run.seed_start, run.seed_start + run.seed_count],
        "seed_key": hex(run.seed_key),
        "rank_counts": list(run.rank_counts),
        "wall_seconds": run.wall_seconds,
        "backend_decisions": backend.decisions,
        "backend_kan_selections": backend.kan_selections,
        "candidate": run.candidate,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
            "model_config_hash": model_config_hash,
        },
        "native": {
            "libriichi_build_sha256": registry["libriichi_build_sha256"],
            "libriichi_git_sha": registry["libriichi_git_sha"],
            "patch_manifest_hash": registry["patch_manifest_hash"],
            "registry_hash": run.registry_hash,
        },
        "note": (
            "Random VNext weights, one native set. This proves ABI/runtime/native "
            "integration only; it is not a win-rate, dan, or strength measurement."
        ),
    }
    (out / "smoke_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
