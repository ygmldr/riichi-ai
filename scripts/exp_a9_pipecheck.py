#!/usr/bin/env python3
"""A9 real-data pipeline check (no model, no GPU).

Loads real gz logs through libriichi's GameplayLoader and the vendor
`FileDatasetsIter`, then:

  1. A/B check on the SAME 64 real entries: the patched collate must differ from
     the original one, preserve every river block's slot multiset/per-element
     counts, preserve the event-token row multiset + padding, keep
     lengths/mask/causal, keep the first token a start_kyoku, and touch only
     window channels;
  2. DataLoader integration with num_workers=0 (stats accumulate in-process) and
     num_workers=2 (fork must inherit the patched collate; any in-worker assert
     kills the loader).

Writes `runs/exp/A9/pipecheck.json` (the evidence file `PVALID.txt` summarises).
Also (re)builds `/tmp/opencode/a9smoke/file_index.pth` with enough real files for
the training smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VENDOR_MORTAL = REPO / "vendor" / "mortal" / "mortal"
SCRIPTS = REPO / "scripts"
LIBRIICHI_DIR = REPO / "vendor" / "mortal" / "target" / "release"
SMOKE_ROOT = Path("/tmp/opencode/a9smoke")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/tmp/opencode/a9smoke/smoke.toml")
    ap.add_argument("--check-files", type=int, default=8)
    ap.add_argument("--index-files", type=int, default=40)
    ap.add_argument("--out", default=str(REPO / "runs" / "exp" / "A9" / "pipecheck.json"))
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    assert cfg_path.is_file(), cfg_path
    os.environ["MORTAL_CFG"] = str(cfg_path)
    for p in (str(SCRIPTS), str(VENDOR_MORTAL), str(LIBRIICHI_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from torch.utils.data import DataLoader

    import exp_a9_shuffle as a9
    import dataloader
    from dataloader import FileDatasetsIter, collate_batch, worker_init_fn
    from config import config

    patched, original = a9.patch_dataloader()
    assert getattr(dataloader, "_a9_patched", False)
    assert dataloader.collate_batch is patched

    year_dir = REPO / "datasets" / "extracted" / "2009"
    all_files = sorted(year_dir.rglob("*.mjson"))
    assert len(all_files) >= args.index_files, len(all_files)
    index_files = all_files[: args.index_files]
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = SMOKE_ROOT / "file_index.pth"
    torch.save({"file_list": [str(f) for f in index_files]}, index_path)
    check_files = [str(f) for f in index_files[: args.check_files]]
    print(f"[A9 pipecheck] index {len(index_files)} files -> {index_path}", flush=True)

    def make_iter(files, file_batch_size=2):
        return FileDatasetsIter(
            version=4,
            file_list=list(files),
            pts=[6.0, 4.0, 2.0, 0.0],
            file_batch_size=file_batch_size,
            reserve_ratio=0.0,
            num_epochs=1,
            enable_augmentation=False,
            augmented_first=False,
        )

    timings = {}

    def run_loader(workers: int, batches: int, batch_size: int):
        a9.STATS = a9.new_stats()
        data = make_iter([str(f) for f in index_files])
        loader = iter(
            DataLoader(
                dataset=data,
                batch_size=batch_size,
                drop_last=False,
                num_workers=workers,
                pin_memory=False,
                worker_init_fn=worker_init_fn,
                collate_fn=dataloader.collate_batch,
            )
        )
        t0 = time.time()
        n = 0
        shapes = None
        per_batch = []
        for batch in loader:
            tb = time.time()
            n += 1
            shapes = (tuple(batch[0].shape), str(batch[0].dtype), tuple(batch[6]["tokens"].shape))
            assert batch[0].shape[1] == a9.OBS_CHANNELS
            assert batch[6]["tokens"].shape[2] == a9.EVENT_TOKEN_DIM
            assert batch[6]["tokens"].sum().item() != 0
            per_batch.append(round(time.time() - tb, 4))
            if n >= batches:
                break
        wall = time.time() - t0
        stats_snapshot = dict(a9.STATS) if workers == 0 else None
        timings[f"workers_{workers}"] = {
            "batches": n,
            "batch_size": batch_size,
            "wall_seconds": round(wall, 2),
            "seconds_per_batch": per_batch,
            "shapes": shapes,
            "stats": stats_snapshot if stats_snapshot is not None else "worker-process counters (see stdout)",
        }
        return n, stats_snapshot

    n2, _ = run_loader(2, 2, 256)
    n0, stats0 = run_loader(0, 4, 64)
    assert n0 == 4 and n2 == 2
    assert stats0["blocks_changed"] > 0, stats0
    assert stats0["events_changed"] > 0, stats0

    entries_iter = iter(make_iter(check_files))
    entries = [next(entries_iter) for _ in range(64)]
    for entry in entries:
        assert np.asarray(entry[0]).shape == (a9.OBS_CHANNELS, a9.TILE_KINDS), np.asarray(entry[0]).shape
        assert np.asarray(entry[-1]).ndim == 2 and np.asarray(entry[-1]).shape[1] == a9.EVENT_TOKEN_DIM

    orig_batch = original(entries)
    orig_snapshot = {
        "obs": orig_batch[0].clone(),
        "tokens": orig_batch[6]["tokens"].clone(),
        "lengths": orig_batch[6]["lengths"].clone(),
        "mask": orig_batch[6]["mask"].clone(),
        "causal": orig_batch[6]["causal"].clone(),
    }
    perm_batch = dataloader.collate_batch(entries)
    assert torch.equal(orig_batch[0], orig_snapshot["obs"]) and torch.equal(
        orig_batch[6]["tokens"], orig_snapshot["tokens"]
    )

    obs_before = orig_batch[0].numpy()
    obs_after = perm_batch[0].numpy()
    assert not np.array_equal(obs_before, obs_after), "A/B: obs unchanged"
    diff_ch = np.flatnonzero(np.any(obs_after != obs_before, axis=(0, 2)))
    assert np.all(a9.WINDOW_CHANNELS[diff_ch]), "A/B: blast radius outside window region"
    for off, n, c, name, _p in a9.RIVER_PERMS:
        a9._check_block(
            obs_before[:, off : off + n * c, :], obs_after[:, off : off + n * c, :], n, c, f"A/B {name}"
        )
    for key in ("lengths", "mask", "causal"):
        assert torch.equal(perm_batch[6][key], orig_snapshot[key]), f"A/B: events.{key} changed"
    toks_before = orig_batch[6]["tokens"].numpy()
    toks_after = perm_batch[6]["tokens"].numpy()
    assert not np.array_equal(toks_before, toks_after), "A/B: event tokens unchanged"
    lens = orig_batch[6]["lengths"].numpy()
    for L in np.unique(lens):
        L = int(L)
        if L <= 1:
            continue
        idx = np.flatnonzero(lens == L)
        assert np.array_equal(
            a9._rows_sorted(toks_before[idx, :L]), a9._rows_sorted(toks_after[idx, :L])
        ), f"A/B: event row multiset changed at L={L}"
    assert (toks_before[:, 0, 0] == 2).all(), "A/B: first real token is not start_kyoku(2)"
    assert not np.array_equal(obs_before[0], obs_after[0]) or not np.array_equal(
        toks_before[0], toks_after[0]
    )
    print(
        f"[A9 pipecheck] A/B OK: 64 entries, {len(diff_ch)} channels differ, "
        f"all inside the window region; token lengths {sorted(set(lens.tolist()))}",
        flush=True,
    )

    payload = {
        "config": str(cfg_path),
        "index_path": str(index_path),
        "index_files": len(index_files),
        "check_files": len(check_files),
        "ab_check": {
            "entries": len(entries),
            "obs_changed": True,
            "obs_diff_channels": int(len(diff_ch)),
            "blast_radius_inside_windows": True,
            "blocks_checked": 8,
            "slot_multiset_preserved": True,
            "per_element_counts_preserved": True,
            "events_tokens_changed": True,
            "events_lengths_mask_causal_unchanged": True,
            "events_row_multiset_preserved": True,
            "first_token_start_kyoku": True,
        },
        "perm_fingerprint_sha256": a9.perms_fingerprint(),
        "perm_seed": a9.PERM_SEED,
        "dataloader": timings,
        "done_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[A9 pipecheck] OK -> {out}", flush=True)
    print(json.dumps({k: v for k, v in payload.items() if k != "dataloader"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
