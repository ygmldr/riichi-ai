#!/usr/bin/env python3
"""A9 (plan todo 20): training-time shuffled-history arm.

Question: A1 (todo 12) permuted river item slots at INFERENCE time on the pinned
challenger and made it worse (Δavg_rank +0.0458, CI excludes 0) -> the trained
model uses discard ORDER at test time. A9 asks the training-time analogue: can a
model learn just as well when the history-dependent inputs are permanently
reordered by ONE FIXED permutation?

Intervention (applied to every training/holdout sample, inside the data pipeline,
i.e. in `collate_batch` running in the DataLoader workers):

  * RIVER WINDOW BLOCKS -- the same eight item-encoded blocks A1 permuted
    (offsets/strides/multiset logic from `runs/analysis/obs_layout.json` and
    `scripts/exp_a1_shuffle.py`). The fixed permutation is drawn from
    `numpy.random.default_rng([0xA15EED, 0, 0])`, i.e. exactly the RNG stream
    A1's shuffle arm used for its (call 0, obs 0) observation, one
    `rng.permutation(items)` per block in layout order. Unlike A1 (which re-drew
    per call/observation) A9 must apply the SAME arrangement to every sample, so
    the stream is drawn once at startup and reused.
  * EVENT-TOKEN ORDER -- one fixed position permutation `pi` of [0, 384)
    (`default_rng([0xA15EED, 1])`). A stream of length L has its L real tokens
    reordered by the rank of `pi[:L]` (the unique length-consistent extension of
    a global position permutation to variable-length streams); padding stays
    zero and `lengths`/`mask`/`causal` are untouched. This is the event arm's
    (E3a-family) history pathway.

In-pipeline self-checks (asserted on EVERY collated batch, in the worker):
  * per river block: the item-SLOT multiset is preserved exactly (bit-packed
    lexicographic compare; all-zero None slots are their own class) AND the
    per-element value counts (sum over slots) are preserved;
  * a non-uniform block whose arrangement changed is evidence the permutation
    applied; the batch is asserted to DIFFER from the unpermuted batch;
  * blast radius: channels that differ from the original are contained in the
    window region;
  * event tokens: per-sample row multiset preserved, padding unchanged, and at
    least one non-uniform stream in the batch must differ.

No vendor file is modified: `scripts/exp_a9_train.py` monkey-patches
`dataloader.collate_batch` before `train.train()` imports it. `--selftest` runs
the synthetic checks with no torch/libriichi dependency; `scripts/exp_a9_pipecheck.py`
runs the same checks on real gz logs from the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LAYOUT_JSON = REPO / "runs" / "analysis" / "obs_layout.json"

# Registered in runs/exp/A1/PREREG.md; A9 reuses the exact same seed.
PERM_SEED = 0xA15EED  # 10575597
OBS_CHANNELS = 1012
TILE_KINDS = 34
EVENT_TOKEN_DIM = 5
MAX_EVENT_TOKENS = 384

EXPECTED_WINDOWS = (
    (35, 6, 4, "self_kawa_recent6"),
    (59, 18, 4, "self_kawa_old18"),
    (132, 6, 8, "opp1_shimocha_kawa_recent6"),
    (180, 18, 8, "opp1_shimocha_kawa_old18"),
    (327, 6, 8, "opp2_toimen_kawa_recent6"),
    (375, 18, 8, "opp2_toimen_kawa_old18"),
    (522, 6, 8, "opp3_kamicha_kawa_recent6"),
    (570, 18, 8, "opp3_kamicha_kawa_old18"),
)


# --------------------------------------------------------------------------- #
# layout + fixed permutations                                                #
# --------------------------------------------------------------------------- #
def load_windows(path: Path = LAYOUT_JSON) -> list[tuple[int, int, int, str]]:
    """Read the verified todo-11 layout; offsets are never re-derived here."""
    layout = json.loads(path.read_text(encoding="utf-8"))
    assert layout["schema"] == "riichi-obs-layout/v1", layout.get("schema")
    assert layout["shape"] == {"channels": OBS_CHANNELS, "tile_kinds": TILE_KINDS}, layout["shape"]
    assert layout["totals"]["covered_channels"] == OBS_CHANNELS
    assert layout["totals"]["gaps"] == []

    top = {b["name"]: b for b in layout["blocks"]}
    assert len(top) == 64, len(top)
    windows: list[tuple[int, int, int, str]] = []
    for b in layout["river_blocks"]["window_blocks"]:
        ref = top[b["name"]]
        assert ref["group"] == "river" and ref["is_river"] is True, b["name"]
        assert b["item_stride"] == b["item_channels"], b
        assert b["count"] == b["items"] * b["item_channels"], b
        assert ref["offset"] == b["offset"] and ref["count"] == b["count"], b
        windows.append((b["offset"], b["items"], b["item_channels"], b["name"]))
    assert len(windows) == 8, windows
    assert tuple(windows) == EXPECTED_WINDOWS, windows
    return windows


WINDOWS = load_windows()


def build_permutations():
    """One fixed draw: river blocks from A1's (call0, obs0) stream + event pi."""
    rng = np.random.default_rng([PERM_SEED, 0, 0])
    river: list[tuple[int, int, int, str, np.ndarray]] = []
    for off, n, c, name in WINDOWS:
        p = rng.permutation(n).astype(np.int64)
        assert np.array_equal(np.sort(p), np.arange(n)), name
        river.append((off, n, c, name, p))
    evpi = np.random.default_rng([PERM_SEED, 1]).permutation(MAX_EVENT_TOKENS).astype(np.int64)
    # Normalise pi[0] > pi[1] so that EVERY prefix restriction pi[:L] (L >= 2) is
    # non-increasing, hence `argsort(pi[:L])` is never the identity: no length
    # class can silently become a no-op (a global permutation alone can have an
    # increasing prefix by chance, e.g. pi[:3] sorted).
    if evpi[0] < evpi[1]:
        evpi[0], evpi[1] = evpi[1], evpi[0]
    assert np.array_equal(np.sort(evpi), np.arange(MAX_EVENT_TOKENS))
    assert evpi[0] > evpi[1]
    return river, evpi


RIVER_PERMS, EVENT_PI = build_permutations()
WINDOW_CHANNELS = np.zeros(OBS_CHANNELS, dtype=bool)
for _off, _n, _c, _name, _p in RIVER_PERMS:
    WINDOW_CHANNELS[_off : _off + _n * _c] = True


def perms_fingerprint() -> str:
    h = hashlib.sha256()
    for _off, _n, _c, name, p in RIVER_PERMS:
        h.update(name.encode())
        h.update(p.astype("<i8").tobytes())
    h.update(EVENT_PI.astype("<i8").tobytes())
    return h.hexdigest()


def perms_summary() -> dict:
    return {
        "perm_seed": PERM_SEED,
        "perm_seed_hex": hex(PERM_SEED),
        "river_blocks": [
            {"name": name, "offset": off, "items": n, "item_channels": c,
             "perm": [int(x) for x in p]}
            for off, n, c, name, p in RIVER_PERMS
        ],
        "event_pi_sha256": hashlib.sha256(EVENT_PI.astype("<i8").tobytes()).hexdigest(),
        "event_pi_head": [int(x) for x in EVENT_PI[:16]],
        "fingerprint_sha256": perms_fingerprint(),
    }


def event_order(L: int) -> np.ndarray:
    """Length-consistent reorder: new[k] = old[argsort(pi[:L])[k]]."""
    assert 0 <= L <= MAX_EVENT_TOKENS, L
    return np.argsort(EVENT_PI[:L], kind="stable")


# --------------------------------------------------------------------------- #
# stats                                                                       #
# --------------------------------------------------------------------------- #
def new_stats() -> dict:
    return {
        "batches": 0,
        "obs_samples": 0,
        "blocks_checked": 0,
        "blocks_changed": 0,
        "blocks_uniform": 0,
        "obs_changed": 0,
        "events_samples": 0,
        "events_checked": 0,
        "events_changed": 0,
        "events_uniform": 0,
    }


STATS = new_stats()
_REPORT_EVERY = 50


def _maybe_report(stats: dict, tag: str) -> None:
    stats["batches"] += 1
    if stats["batches"] % _REPORT_EVERY == 0:
        print(
            f"[A9 pipe {tag} pid={__import__('os').getpid()} {time.strftime('%H:%M:%S')}] "
            f"batches={stats['batches']} obs={stats['obs_samples']} "
            f"blocks={stats['blocks_checked']} changed={stats['blocks_changed']} "
            f"uniform={stats['blocks_uniform']} ev_samples={stats['events_samples']} "
            f"ev_checked={stats['events_checked']} ev_changed={stats['events_changed']}",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# core permutations + exact checks                                            #
# --------------------------------------------------------------------------- #
def _as_batch_obs(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None]
    assert a.ndim == 3 and a.shape[1] == OBS_CHANNELS and a.shape[2] == TILE_KINDS, a.shape
    return a


def _slot_multiset(block: np.ndarray, n: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    """(sorted bit-packed slots, per-element counts) for a (B, n*c, 34) block."""
    b = block.reshape(block.shape[0], n, c * TILE_KINDS)
    bits = b if b.dtype == np.bool_ else (b != 0)
    packed = np.packbits(bits, axis=2)
    packed = np.ascontiguousarray(packed)
    k = packed.shape[2]
    v = packed.view(f"V{k}").reshape(packed.shape[0], n)
    return np.sort(v, axis=1), b.sum(axis=1)


def _check_block(before: np.ndarray, after: np.ndarray, n: int, c: int, name: str) -> None:
    """Exact slot-multiset + per-element-count equality for one river block."""
    sb, cb = _slot_multiset(before, n, c)
    sa, ca = _slot_multiset(after, n, c)
    assert np.array_equal(sb, sa), f"{name}: slot multiset changed"
    assert np.array_equal(cb, ca), f"{name}: per-element counts changed"


def _rows_sorted(sub: np.ndarray) -> np.ndarray:
    """Sorted per-sample row multiset of a (m, L, 5) int16 token block."""
    v = np.ascontiguousarray(sub)
    assert v.shape[-1] == EVENT_TOKEN_DIM, v.shape
    v = v.view(np.dtype(("V", v.dtype.itemsize * EVENT_TOKEN_DIM)))
    return np.sort(v.reshape(v.shape[0], v.shape[1]), axis=1)


def permute_obs(arr, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Apply the fixed river-window permutation to a (B,1012,34) batch.

    Returns (permuted copy, per-call stats). Raises AssertionError if any block's
    slot multiset / per-element count changes, if the batch is identical to the
    original, or if a channel outside the window region changed.
    """
    a = _as_batch_obs(arr)
    st = {
        "blocks_checked": 0,
        "blocks_changed": 0,
        "blocks_uniform": 0,
    }
    out = a.copy()
    for off, n, c, name, perm in RIVER_PERMS:
        seg = out[:, off : off + n * c, :]
        before = seg.copy()
        after_view = seg.reshape(a.shape[0], n, c, TILE_KINDS)[:, perm, :, :]
        seg[:] = after_view.reshape(a.shape[0], n * c, TILE_KINDS)
        _check_block(before, seg, n, c, name)
        st["blocks_checked"] += 1
        if np.array_equal(before, seg):
            st["blocks_uniform"] += 1
        else:
            st["blocks_changed"] += 1
    diff_mask = np.any(out != a, axis=(0, 2))
    diff_ch = np.flatnonzero(diff_mask)
    assert np.all(WINDOW_CHANNELS[diff_ch]), (
        f"permutation touched non-window channels: {diff_ch[~WINDOW_CHANNELS[diff_ch]][:8]}"
    )
    if st["blocks_changed"] == 0:
        raise AssertionError("permuted batch is identical to the unpermuted batch (no-op arm)")
    if stats is not None:
        stats["obs_samples"] += a.shape[0]
        stats["obs_changed"] += int(np.any(out != a, axis=(1, 2)).sum())
        for k in st:
            stats[k] += st[k]
    return out, st


def permute_events(tokens, lengths, mask=None, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Reorder event tokens within each stream by the fixed length-consistent pi."""
    toks = np.asarray(tokens)
    assert toks.ndim == 3 and toks.shape[2] == EVENT_TOKEN_DIM, toks.shape
    assert toks.shape[1] <= MAX_EVENT_TOKENS, toks.shape
    lens = np.asarray(lengths).astype(np.int64)
    assert lens.shape == (toks.shape[0],), (lens.shape, toks.shape)
    st = {"events_checked": 0, "events_changed": 0, "events_uniform": 0}
    out = toks.copy()
    for L in np.unique(lens):
        L = int(L)
        if L <= 1:
            continue
        idx = np.flatnonzero(lens == L)
        order = event_order(L)
        sub = out[idx, :L].copy()
        sb = _rows_sorted(sub)
        new = sub[:, order]
        sa = _rows_sorted(new)
        assert np.array_equal(sb, sa), f"events L={L}: row multiset changed"
        uniform = np.all(sub == sub[:, :1, :], axis=(1, 2))
        changed = np.any(new != sub, axis=(1, 2))
        assert np.all(changed[~uniform]), (
            f"events L={L}: non-uniform stream did not change under the permutation"
        )
        out[idx[:, None], np.arange(L)] = new
        st["events_checked"] += int(len(idx))
        st["events_changed"] += int(changed.sum())
        st["events_uniform"] += int(uniform.sum())
    if mask is not None:
        m = np.asarray(mask)
        assert m.shape == out.shape[:2], (m.shape, out.shape)
        assert np.array_equal(out[~m], toks[~m]), "event padding region changed"
    if stats is not None:
        stats["events_samples"] += int(toks.shape[0])
        for k in st:
            stats[k] += st[k]
    return out, st


# --------------------------------------------------------------------------- #
# collate wrapper / dataloader patch                                          #
# --------------------------------------------------------------------------- #
def make_shuffled_collate(original_collate):
    """Wrap `dataloader.collate_batch`: permute obs river blocks + event tokens."""

    def shuffled_collate(batch):
        import torch

        out = original_collate(batch)
        obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, events = out
        obs_np = obs.numpy() if hasattr(obs, "numpy") else np.asarray(obs)
        obs_new, _ = permute_obs(obs_np, STATS)
        toks = events["tokens"].numpy() if hasattr(events["tokens"], "numpy") else np.asarray(events["tokens"])
        lens = events["lengths"].numpy() if hasattr(events["lengths"], "numpy") else np.asarray(events["lengths"])
        mask = events["mask"].numpy() if hasattr(events["mask"], "numpy") else np.asarray(events["mask"])
        toks_new, ev_st = permute_events(toks, lens, mask, STATS)
        if ev_st["events_checked"] > 0:
            assert ev_st["events_changed"] > 0, "no event stream in the batch changed"
        events = dict(events)
        events["tokens"] = torch.from_numpy(toks_new)
        _maybe_report(STATS, "train")
        return (
            torch.from_numpy(obs_new),
            actions,
            masks,
            steps_to_done,
            kyoku_rewards,
            player_ranks,
            events,
        )

    return shuffled_collate


def patch_dataloader():
    """Monkey-patch `dataloader.collate_batch` in this process (no vendor edit)."""
    import dataloader

    assert hasattr(dataloader, "collate_batch"), "vendor dataloader layout changed"
    if getattr(dataloader, "_a9_patched", False):
        return dataloader.collate_batch, None
    original = dataloader.collate_batch
    dataloader.collate_batch = make_shuffled_collate(original)
    dataloader._a9_patched = True
    dataloader._a9_original_collate = original
    return dataloader.collate_batch, original


# --------------------------------------------------------------------------- #
# selftest                                                                    #
# --------------------------------------------------------------------------- #
class _DummyOrigCollate:
    """Stand-in for `dataloader.collate_batch`: returns torch tensors."""

    def __init__(self, obs, tokens, lengths):
        self.obs = obs
        self.tokens = tokens
        self.lengths = lengths

    def __call__(self, batch):
        import torch

        B, T = self.tokens.shape[0], self.tokens.shape[1]
        events = {
            "tokens": torch.as_tensor(self.tokens, dtype=torch.int16),
            "lengths": torch.as_tensor(self.lengths, dtype=torch.int64),
            "mask": torch.as_tensor(
                np.arange(T)[None, :] < self.lengths[:, None], dtype=torch.bool
            ),
            "causal": torch.tril(torch.ones((T, T), dtype=torch.bool)),
        }
        return (
            torch.as_tensor(self.obs),
            torch.zeros(B, dtype=torch.int64),
            torch.zeros((B, 46), dtype=torch.bool),
            torch.zeros(B, dtype=torch.int64),
            torch.zeros(B, dtype=torch.float64),
            torch.zeros(B, dtype=torch.int64),
            events,
        )


def _synthetic_obs(rng: np.random.Generator, B: int, uniform_block: int | None = None,
                   binary: bool = True) -> np.ndarray:
    obs = np.zeros((B, OBS_CHANNELS, TILE_KINDS), dtype=np.float32)
    hi = 2 if binary else 5
    for b_idx, (off, n, c, _name, _p) in enumerate(RIVER_PERMS):
        if b_idx == uniform_block:
            continue
        obs[:, off : off + n * c] = rng.integers(0, hi, size=(B, n * c, TILE_KINDS)).astype(
            np.float32
        )
    return obs


def selftest() -> dict:
    rng = np.random.default_rng(12345)
    checks: list[str] = []

    assert tuple(w[:4] for w in WINDOWS) == EXPECTED_WINDOWS
    checks.append("layout: 8 window blocks match A1 offsets/items/item_channels")

    # permutations are real, fixed, and (for every block) not the identity
    for off, n, c, name, p in RIVER_PERMS:
        assert np.array_equal(np.sort(p), np.arange(n)), name
        assert not np.array_equal(p, np.arange(n)), f"{name}: fixed permutation is identity"
    assert not np.array_equal(EVENT_PI, np.arange(MAX_EVENT_TOKENS))
    checks.append("perms: 8 river perms + event pi are fixed non-identity permutations")

    # length-consistent extension: pi[:L] reorder changes L>=3 streams
    for L in (3, 6, 18, 100, 384):
        assert not np.array_equal(event_order(L), np.arange(L)), L
    checks.append("event pi: order extension non-identity for L in {3,6,18,100,384}")

    # river permutation: binary batch, one all-empty block
    obs = _synthetic_obs(rng, B=8, uniform_block=1)
    st = new_stats()
    out, ost = permute_obs(obs, st)
    assert ost["blocks_checked"] == 8 and ost["blocks_uniform"] == 1, ost
    assert ost["blocks_changed"] == 7, ost
    assert np.any(out != obs)
    # determinism: same input -> same output
    out2, _ = permute_obs(obs)
    assert np.array_equal(out, out2)
    checks.append("river: multiset/count asserts pass; 7/8 changed, 1 uniform; deterministic")

    # non-binary fallback path
    obs_f = _synthetic_obs(rng, B=3, binary=False)
    _, stf = permute_obs(obs_f)
    assert stf["blocks_changed"] == 8, stf
    checks.append("river: non-binary values pass the same slot-multiset checks")

    before = obs[0:1, 35 : 35 + 24].copy()
    bad_bit = before.copy()
    bad_bit[0, 0, 0] = 1.0 - bad_bit[0, 0, 0]
    bad_val = before.copy()
    bad_val[0, 0, 0] += 2.0
    for label, bad in (("bit-flip", bad_bit), ("value-bump", bad_val)):
        caught = False
        try:
            _check_block(before, bad, 6, 4, f"negative-control-{label}")
        except AssertionError:
            caught = True
        assert caught, f"slot check did not detect a corrupted block ({label})"
    checks.append("river: negative controls (bit flip + value bump) are caught")

    # events: lengths 1/5/12 non-uniform + one all-same stream
    lengths = np.array([1, 5, 12, 12], dtype=np.int64)
    T = 12
    toks = np.zeros((4, T, 5), dtype=np.int16)
    toks[1, :5] = rng.integers(1, 17, size=(5, 5)).astype(np.int16)
    toks[2, :12] = rng.integers(1, 17, size=(12, 5)).astype(np.int16)
    toks[3, :12] = np.tile(np.array([[2, 0, 0, 0, 0]], dtype=np.int16), (12, 1))
    mask = np.arange(T)[None, :] < lengths[:, None]
    new, est = permute_events(toks, lengths, mask)
    assert est["events_checked"] == 3 and est["events_uniform"] == 1, est  # L=1 skipped
    assert est["events_changed"] == 2, est
    assert np.array_equal(new[0], toks[0])  # L=1 untouched
    assert np.array_equal(new[3], toks[3])  # uniform stream unchanged
    assert np.array_equal(new[~mask], toks[~mask])
    # determinism
    new2, _ = permute_events(toks, lengths, mask)
    assert np.array_equal(new, new2)
    checks.append("events: row multiset preserved, padding kept, non-uniform streams changed")

    # collate wrapper via a dummy original collate
    sys.path.insert(0, str(REPO / "scripts"))
    import torch  # noqa: F401

    B = 4
    obs_t = _synthetic_obs(rng, B)
    dummy = _DummyOrigCollate(
        obs_t,
        toks,
        lengths,
    )
    wrapped = make_shuffled_collate(dummy)
    res = wrapped([None] * B)
    assert res[0].shape == (B, OBS_CHANNELS, TILE_KINDS), res[0].shape
    assert str(res[6]["tokens"].dtype) == "torch.int16", res[6]["tokens"].dtype
    assert not np.array_equal(res[0].numpy(), obs_t)
    assert not np.array_equal(res[6]["tokens"].numpy(), toks)
    checks.append("collate: wrapper permutes obs + events and keeps dtypes/shapes")

    summary = {
        "ok": True,
        "checks": checks,
        "n_checks": len(checks),
        "perms": perms_summary(),
        "perm_apply_stats": st,
        "done_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A9 shuffled-history permutation core")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-perms", action="store_true")
    args = ap.parse_args(argv)
    if args.dump_perms:
        print(json.dumps(perms_summary(), indent=2))
        return 0
    if args.selftest:
        summary = selftest()
        print(json.dumps(summary, indent=2))
        print(
            f"SELFTEST OK: {summary['n_checks']} checks, fingerprint "
            f"{summary['perms']['fingerprint_sha256'][:16]}"
        )
        return 0
    ap.error("nothing to do; pass --selftest or --dump-perms")


if __name__ == "__main__":
    sys.exit(main())
