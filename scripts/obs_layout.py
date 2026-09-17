#!/usr/bin/env python3
"""v4 observation plane layout -> runs/analysis/obs_layout.json.

Machine-readable map of libriichi's v4 observation tensor layout ((1012, 34)):
1012 channel planes x 34 tile kinds.  Every channel block is enumerated from
`vendor/mortal/libriichi/src/state/obs_repr.rs` top to bottom with a running
channel offset, and the total is asserted to equal `obs_shape(4)` parsed from
`vendor/mortal/libriichi/src/consts.rs`.

Consumers:
  * todo 12 (A1 experiment) permutes the river window blocks (and zeroes the
    recency planes); it needs byte-exact offsets/strides -- see the
    `river_blocks` section and the per-block `is_river`/`item_channels` fields.

Failure mode (by design): if the block table does not sum to the expected
channel count, the gap intervals are printed and the script exits non-zero.
Never fudge the table to make it fit.

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OBS_REPR_RS = REPO_ROOT / "vendor/mortal/libriichi/src/state/obs_repr.rs"
CONSTS_RS = REPO_ROOT / "vendor/mortal/libriichi/src/consts.rs"
DEFAULT_OUTPUT = REPO_ROOT / "runs/analysis/obs_layout.json"

# Paths as recorded inside the JSON (repo-relative, for reviewer auditing).
OBS_REPR_REF = "vendor/mortal/libriichi/src/state/obs_repr.rs"
CONSTS_REF = "vendor/mortal/libriichi/src/consts.rs"

EXPECTED_TOTAL = 1012  # todo 11 acceptance: assert total channels == 1012
TILE_KINDS = 34

# Relative player order used everywhere in the obs (player_state.rs:65 rotates
# every per-player array so index 0 is self).
PLAYER_NAMES = ["self", "shimocha", "toimen", "kamicha"]


def parse_rust_usize(text: str, name: str, source: Path) -> int:
    """Parse `const NAME: usize = <int>;` from a Rust file."""
    m = re.search(rf"const\s+{name}\s*:\s*usize\s*=\s*(\d+)\s*;", text)
    if not m:
        raise SystemExit(f"FATAL: could not find `const {name}: usize` in {source}")
    return int(m.group(1))


def parse_obs_shape(text: str, version: int, source: Path) -> tuple[int, int]:
    """Parse the `obs_shape` match arm for `version` (not oracle_obs_shape)."""
    start = text.find("pub const fn obs_shape")
    end = text.find("pub const fn oracle_obs_shape")
    if start < 0 or end < 0 or end <= start:
        raise SystemExit(f"FATAL: could not locate obs_shape() in {source}")
    m = re.search(rf"\b{version}\s*=>\s*\((\d+)\s*,\s*(\d+)\)", text[start:end])
    if not m:
        raise SystemExit(f"FATAL: obs_shape has no arm for version {version} in {source}")
    return int(m.group(1)), int(m.group(2))


class BlockBuilder:
    """Sequential channel-block builder: offsets are implicit and contiguous."""

    def __init__(self) -> None:
        self.blocks: list[dict] = []
        self._pos = 0

    def add(
        self,
        name: str,
        count: int,
        semantics: str,
        refs: list[str],
        *,
        group: str,
        is_river: bool = False,
        is_recency_plane: bool = False,
        river_player: int | None = None,
        window: str | None = None,
        items: int | None = None,
        item_channels: int | None = None,
        item_order: str | None = None,
        zero_padded: bool = False,
    ) -> None:
        assert count > 0, f"block {name}: count must be positive"
        block: dict = {
            "name": name,
            "offset": self._pos,
            "count": count,
            "end": self._pos + count,
            "semantics": semantics,
            "refs": refs,
            "group": group,
            "is_river": is_river,
        }
        if is_river or is_recency_plane:
            assert river_player is not None, f"block {name}: river block needs river_player"
            block.update(
                {
                    "is_recency_plane": is_recency_plane,
                    "river_player": river_player,
                    "river_player_name": PLAYER_NAMES[river_player],
                    "window": window,
                    "items": items,
                    "item_channels": item_channels,
                    "item_stride": item_channels,
                    "item_order": item_order,
                    "zero_padded": zero_padded,
                }
            )
        self.blocks.append(block)
        self._pos += count

    @property
    def total(self) -> int:
        return self._pos


def build_blocks(c: dict[str, int]) -> list[dict]:
    """Enumerate every channel block of `encode_obs` for version 4.

    `c` carries the Rust constants parsed from obs_repr.rs:
    SELF_KAWA_ITEM_CHANNELS, KAWA_ITEM_CHANNELS, MAX_NUM_TURNS.
    """
    self_kawa = c["SELF_KAWA_ITEM_CHANNELS"]  # 4
    kawa = c["KAWA_ITEM_CHANNELS"]  # 8
    num_turns = c["MAX_NUM_TURNS"]  # 17
    b = BlockBuilder()

    # --- global / game state -------------------------------------------------
    b.add(
        "tehai",
        4,
        "Own hand as 4 copy-planes per tile kind: plane offset+c is 1 for tile t "
        "iff tehai[t] > c (hand counts, not one-hot; 4 channels per 34-wide plane).",
        [f"{OBS_REPR_REF}:130-139"],
        group="hand",
    )
    b.add(
        "aka_in_hand",
        3,
        "Own hand red fives: [+0]=5mr, [+1]=5pr, [+2]=5sr (akas_in_hand order); "
        "1 iff that red five is in hand.",
        [f"{OBS_REPR_REF}:141-147"],
        group="hand",
    )
    b.add(
        "scores",
        8,
        "Relative player scores, 2 planes per player in relative order "
        "(self, shimocha, toimen, kamicha): score.clamp(0,100000)/100000, then "
        "score.clamp(0,30000)/30000 (v4-only second plane).",
        [f"{OBS_REPR_REF}:149-165", f"{OBS_REPR_REF}:158-162"],
        group="state",
    )
    b.add(
        "rank",
        4,
        "Own rank one-hot: plane offset+n set, n = state.rank in 0..3.",
        [f"{OBS_REPR_REF}:167-169"],
        group="state",
    )
    b.add(
        "kyoku",
        4,
        "Kyoku (round within bakaze) one-hot: plane offset+n set, n = state.kyoku.",
        [f"{OBS_REPR_REF}:171-178"],
        group="state",
    )
    b.add(
        "honba",
        1,
        "Honba count / 10, clamped (v4 IntegerEncoder honors only rescale here; "
        "`rbf_intervals` is ignored by the v4 branch of IntegerEncoder::encode).",
        [f"{OBS_REPR_REF}:180-189", f"{OBS_REPR_REF}:92-104"],
        group="state",
    )
    b.add(
        "kyotaku",
        1,
        "Kyotaku (riichi sticks on table) / 10, clamped (same v4 rescale-only rule).",
        [f"{OBS_REPR_REF}:190-194", f"{OBS_REPR_REF}:92-104"],
        group="state",
    )
    b.add(
        "bakaze_jikaze",
        2,
        "+0 = bakaze one-hot (E/S/W/N encode to tile kinds 27..30), "
        "+1 = jikaze one-hot.",
        [f"{OBS_REPR_REF}:196-198"],
        group="state",
    )
    b.add(
        "round_index",
        1,
        "Game round index ((bakaze-E).min(1)*4 + kyoku) / 7, clamped.",
        [f"{OBS_REPR_REF}:200-205"],
        group="state",
    )
    b.add(
        "dora_indicators",
        7,
        "Dora indicator tiles via encode_tile_set: 4 copy-planes per tile kind "
        "(up to 5 indicators) + 3 aka planes at +4+i for aka five i "
        "(5mr/5pr/5sr).",
        [f"{OBS_REPR_REF}:207", f"{OBS_REPR_REF}:694-712"],
        group="dora",
    )

    # --- rivers: self --------------------------------------------------------
    b.add(
        "self_kawa_recent6",
        6 * self_kawa,
        "Own river window A: earliest 6 kawa slots in discard order "
        "(kawa[0].iter().take(6)). 4 planes per item: +0 kan tile (deaka'd), "
        "+1 sutehai tile (deaka'd), +2 sutehai is aka, +3 sutehai is dora. A "
        "None slot (turn skipped by a call, or the pre-dealer pads at kyoku "
        "start) is all-zero. Always consumes exactly 6 slots.",
        [f"{OBS_REPR_REF}:209-213", f"{OBS_REPR_REF}:714-734"],
        group="river",
        is_river=True,
        river_player=0,
        window="recent6",
        items=6,
        item_channels=self_kawa,
        item_order="oldest_first",
        zero_padded=True,
    )
    b.add(
        "self_kawa_old18",
        18 * self_kawa,
        "Own river window B: last 18 kawa slots in REVERSE order (newest first; "
        "kawa[0].iter().rev().take(18)), 4 planes per item, same layout as "
        "window A; None slots anywhere are all-zero and the window always "
        "consumes exactly 18 slots (tail zero-padded when the river is shorter).",
        [f"{OBS_REPR_REF}:215-220", f"{OBS_REPR_REF}:714-734"],
        group="river",
        is_river=True,
        river_player=0,
        window="old18",
        items=18,
        item_channels=self_kawa,
        item_order="newest_first",
        zero_padded=True,
    )
    b.add(
        "self_kawa_recency",
        1,
        "Own river recency plane: one-hot over the deaka'd discard tile per kawa "
        "slot, weighted exp(-0.2*(max_kawa_len-1-turn)) (newer discards heavier); "
        "single 34-wide channel.",
        [f"{OBS_REPR_REF}:222-233"],
        group="river",
        is_recency_plane=True,
        river_player=0,
        window="recency",
        items=1,
        item_channels=1,
        item_order="single",
    )

    # --- rivers: other players (kawa[1..], relative order) -------------------
    for rel in range(1, 4):
        seat = PLAYER_NAMES[rel]
        b.add(
            f"opp{rel}_{seat}_kawa_recent6",
            6 * kawa,
            f"Opponent kawa[{rel}] ({seat}, relative +{rel}) river window A: "
            "earliest 6 kawa slots in discard order. 8 planes per item: +0/+1 "
            "chi/pon consumed tiles (deaka'd min/max), +2 kan tile, +3 sutehai "
            "tile (deaka'd), +4 sutehai aka, +5 sutehai dora, +6 sutehai "
            "tedashi, +7 sutehai riichi. A None slot (turn skipped by a call, "
            "or the pre-dealer pads at kyoku start) is all-zero. Always "
            "consumes exactly 6 slots.",
            [f"{OBS_REPR_REF}:235-240", f"{OBS_REPR_REF}:736-773"],
            group="river",
            is_river=True,
            river_player=rel,
            window="recent6",
            items=6,
            item_channels=kawa,
            item_order="oldest_first",
            zero_padded=True,
        )
        b.add(
            f"opp{rel}_{seat}_kawa_old18",
            18 * kawa,
            f"Opponent kawa[{rel}] ({seat}) river window B: last 18 kawa slots "
            "in REVERSE order (newest first), 8 planes per item, same layout as "
            "window A; None slots anywhere are all-zero and the window always "
            "consumes exactly 18 slots (tail zero-padded when the river is "
            "shorter).",
            [f"{OBS_REPR_REF}:241-247", f"{OBS_REPR_REF}:736-773"],
            group="river",
            is_river=True,
            river_player=rel,
            window="old18",
            items=18,
            item_channels=kawa,
            item_order="newest_first",
            zero_padded=True,
        )
        b.add(
            f"opp{rel}_{seat}_kawa_recency",
            3,
            f"Opponent kawa[{rel}] ({seat}) recency planes, one per kawa slot "
            "weighted exp(-0.2*(max_kawa_len-1-turn)): +0 sutehai tile one-hot, "
            "+1 tedashi flag, +2 riichi flag.",
            [f"{OBS_REPR_REF}:261-277"],
            group="river",
            is_recency_plane=True,
            river_player=rel,
            window="recency",
            items=1,
            item_channels=3,
            item_order="per_slot",
        )

    # --- remaining global state ---------------------------------------------
    b.add(
        "tiles_left",
        1,
        "state.tiles_left / 69 (live-wall tiles remaining).",
        [f"{OBS_REPR_REF}:282-284"],
        group="state",
    )
    b.add(
        "doras_owned",
        4,
        "Dora count owned (hand + melds) per relative player / 12, clamped "
        "(v4 rescale only).",
        [f"{OBS_REPR_REF}:286-291", f"{OBS_REPR_REF}:92-104"],
        group="dora",
    )
    b.add(
        "doras_unseen",
        1,
        "Unseen dora count (dora_indicators.len()*4 + 3 - doras_seen) / 23, "
        "clamped (v4 rescale only).",
        [f"{OBS_REPR_REF}:293-297", f"{OBS_REPR_REF}:92-104"],
        group="dora",
    )
    for rel in range(4):
        seat = PLAYER_NAMES[rel]
        b.add(
            f"kawa_overview_{seat}",
            7,
            f"River tiles of relative player {rel} ({seat}) as a tile set "
            "(kawa_overview) via encode_tile_set: 4 copy-planes per tile kind + "
            "3 aka planes (+4+i for aka five i). Every river slot contributes "
            "one tile to its tile kind's next copy plane.",
            [f"{OBS_REPR_REF}:299-301", f"{OBS_REPR_REF}:694-712"],
            group="river_overview",
        )
    for rel in range(4):
        seat = PLAYER_NAMES[rel]
        b.add(
            f"fuuro_overview_{seat}",
            4 * 5,
            f"Called melds of relative player {rel} ({seat}): 4 meld slots x 5 "
            "planes = 4 copy-planes for the meld tiles (deaka'd) + 1 plane = "
            "meld contains an aka five. Missing meld slots are all-zero.",
            [f"{OBS_REPR_REF}:303-321"],
            group="melds",
        )
    for rel in range(4):
        seat = PLAYER_NAMES[rel]
        b.add(
            f"ankan_overview_{seat}",
            1,
            f"Closed kans (ankan_overview) of relative player {rel} ({seat}): "
            "one-hot over the deaka'd ankan tile kinds.",
            [f"{OBS_REPR_REF}:323-329"],
            group="melds",
        )
    b.add(
        "tiles_seen",
        1,
        "Count of each tile kind seen by everyone / 4 (tiles_seen).",
        [f"{OBS_REPR_REF}:331-335"],
        group="state",
    )
    for rel in range(1, 4):
        seat = PLAYER_NAMES[rel]
        b.add(
            f"last_tedashi_{seat}",
            3,
            f"Relative opponent {rel} ({seat})'s last tedashi discard: +0 tile "
            "one-hot (deaka'd), +1 aka flag, +2 is_dora flag.",
            [f"{OBS_REPR_REF}:337-351"],
            group="opponent_state",
        )
    for rel in range(1, 4):
        seat = PLAYER_NAMES[rel]
        b.add(
            f"riichi_sutehai_{seat}",
            3,
            f"Relative opponent {rel} ({seat})'s riichi declaration discard: "
            "+0 tile one-hot (deaka'd), +1 aka flag, +2 is_dora flag.",
            [f"{OBS_REPR_REF}:352-366"],
            group="opponent_state",
        )
    b.add(
        "riichi_declared_opponents",
        3,
        "One-hot over the 3 relative opponents that declared riichi "
        "(riichi_declared[1..], declaration not yet accepted).",
        [f"{OBS_REPR_REF}:369-374"],
        group="opponent_state",
    )
    b.add(
        "riichi_accepted_opponents",
        3,
        "One-hot over the 3 relative opponents whose riichi is accepted "
        "(riichi_accepted[1..]).",
        [f"{OBS_REPR_REF}:375-380"],
        group="opponent_state",
    )

    # --- own decision state ---------------------------------------------------
    b.add(
        "waits",
        1,
        "Own winning tiles (waits) one-hot over tile kinds.",
        [f"{OBS_REPR_REF}:382-388"],
        group="self_state",
    )
    b.add(
        "at_furiten",
        1,
        "1 if self is in (temporary or riichi) furiten.",
        [f"{OBS_REPR_REF}:390-393"],
        group="self_state",
    )
    b.add(
        "shanten",
        7,
        "Own shanten one-hot over 0..6 (IntegerEncoder one_hot, cap 6 -> cap+1 = "
        "7 planes; shanten values >= 6 clamp onto the last plane).",
        [f"{OBS_REPR_REF}:395-396", f"{OBS_REPR_REF}:92-104"],
        group="self_state",
    )
    b.add(
        "own_riichi_accepted",
        1,
        "1 if own riichi is accepted (riichi_accepted[0]).",
        [f"{OBS_REPR_REF}:398-401"],
        group="self_state",
    )
    b.add(
        "at_kan_select",
        1,
        "1 while the agent is at a kan-choice (chankan) decision point.",
        [f"{OBS_REPR_REF}:403-406"],
        group="action",
    )
    b.add(
        "last_kawa_tile_call",
        3,
        "Tile that can be called (state.last_kawa_tile, used for "
        "chi/pon/daiminkan/ron features): +0 tile one-hot (deaka'd), +1 aka "
        "flag, +2 dora_factor > 0 flag.",
        [f"{OBS_REPR_REF}:408-429"],
        group="action",
    )
    b.add(
        "discard_candidates",
        5,
        "Discard candidates when cans.can_discard: +0 discards (deaka'd, "
        "candidates_aka collapsed to base five) one-hot, +1 keep_shanten "
        "discards, +2 next_shanten discards, +3 unconditional-tenpai discards "
        "(only when shanten <= 1), +4 own riichi already declared flag.",
        [f"{OBS_REPR_REF}:431-476"],
        group="action",
    )
    b.add(
        "can_riichi",
        1,
        "1 if riichi is available (cans.can_riichi).",
        [f"{OBS_REPR_REF}:478-484"],
        group="action",
    )
    b.add(
        "can_chi",
        3,
        "Chi availability from the last discard: +0 low (1-2-3 shape), +1 mid, "
        "+2 high.",
        [f"{OBS_REPR_REF}:486-504"],
        group="action",
    )
    b.add(
        "can_pon",
        1,
        "1 if pon is available (cans.can_pon).",
        [f"{OBS_REPR_REF}:506-512"],
        group="action",
    )
    b.add(
        "can_daiminkan",
        1,
        "1 if daiminkan is available (cans.can_daiminkan).",
        [f"{OBS_REPR_REF}:514-520"],
        group="action",
    )
    b.add(
        "can_ankan",
        1,
        "Closed-kan choice: one-hot over ankan_candidates tile kinds "
        "(deaka'd; multiple candidates can be lit).",
        [f"{OBS_REPR_REF}:522-533"],
        group="action",
    )
    b.add(
        "can_kakan",
        1,
        "Added-kan choice: one-hot over kakan_candidates tile kinds "
        "(deaka'd).",
        [f"{OBS_REPR_REF}:535-546"],
        group="action",
    )
    b.add(
        "can_agari",
        1,
        "1 if agari (win) is available (cans.can_agari()).",
        [f"{OBS_REPR_REF}:548-554"],
        group="action",
    )
    b.add(
        "can_ryukyoku",
        1,
        "1 if nine-nine/exhaustive draw declaration is available "
        "(cans.can_ryukyoku).",
        [f"{OBS_REPR_REF}:556-562"],
        group="action",
    )

    # --- v4-only single-player (SP) tables ------------------------------------
    b.add(
        "sp_ev",
        2,
        "v4 single-player max EV: +0 value.clamp(0,100000)/100000, +1 "
        "value.clamp(0,30000)/30000 (same value at two scales; "
        "agari-point fallback used when SP tables are unavailable).",
        [f"{OBS_REPR_REF}:564-574", f"{OBS_REPR_REF}:632-638"],
        group="sp",
    )
    b.add(
        "sp_required_tiles",
        2 * TILE_KINDS + 2,
        "v4 single-player required-tile planes (can_discard layout): +0..+33 "
        "plane per discard tile kind = one-hot required tiles for non-shanten-"
        "down candidates; +34..+67 same for shanten-down candidates; +68 "
        "one-hot of the discard tile with max required tiles; +69 reserved / "
        "always zero. When !can_discard the same 70 slots hold the single "
        "candidate's required tiles at +68 (and +69 stays reserved).",
        [f"{OBS_REPR_REF}:576-608"],
        group="sp",
    )
    b.add(
        "sp_tables",
        3 * num_turns,
        "v4 single-player table: 3 x MAX_NUM_TURNS(17) planes. +0..+16 = "
        "tenpai probability per future turn (each plane 34-wide over discard "
        "tiles), +17..+33 = win probability, +34..+50 = exp value normalized "
        "by max EV. All-zero when shanten >= 4 or all tenpai probs are zero; "
        "slots are always consumed.",
        [f"{OBS_REPR_REF}:640-692"],
        group="sp",
    )
    return b.blocks


def find_gaps(blocks: list[dict], expected_total: int) -> list[tuple[int, int]]:
    """Return uncovered [start, end) channel intervals over [0, expected_total)."""
    gaps: list[tuple[int, int]] = []
    pos = 0
    for blk in blocks:
        if blk["offset"] > pos:
            gaps.append((pos, blk["offset"]))
        if blk["offset"] < pos:
            raise SystemExit(
                f"FATAL: block {blk['name']} overlaps previous coverage at "
                f"{blk['offset']} (already at {pos})"
            )
        pos = blk["end"]
    if pos < expected_total:
        gaps.append((pos, expected_total))
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    if not OBS_REPR_RS.is_file():
        raise SystemExit(f"FATAL: missing {OBS_REPR_RS} (vendor submodule not checked out?)")
    if not CONSTS_RS.is_file():
        raise SystemExit(f"FATAL: missing {CONSTS_RS}")

    repr_text = OBS_REPR_RS.read_text(encoding="utf-8")
    consts_text = CONSTS_RS.read_text(encoding="utf-8")

    constants = {
        "SELF_KAWA_ITEM_CHANNELS": parse_rust_usize(
            repr_text, "SELF_KAWA_ITEM_CHANNELS", OBS_REPR_RS
        ),
        "KAWA_ITEM_CHANNELS": parse_rust_usize(repr_text, "KAWA_ITEM_CHANNELS", OBS_REPR_RS),
        "MAX_NUM_TURNS": parse_rust_usize(repr_text, "MAX_NUM_TURNS", OBS_REPR_RS),
    }
    expected_total, tile_kinds = parse_obs_shape(consts_text, 4, CONSTS_RS)

    if expected_total != EXPECTED_TOTAL or tile_kinds != TILE_KINDS:
        print(
            f"FATAL: consts.rs obs_shape(4) = ({expected_total}, {tile_kinds}) but todo 11 "
            f"requires ({EXPECTED_TOTAL}, {TILE_KINDS})",
            file=sys.stderr,
        )
        return 1

    blocks = build_blocks(constants)
    gaps = find_gaps(blocks, expected_total)
    total = sum(blk["count"] for blk in blocks)

    print(f"source: {OBS_REPR_REF}")
    print(f"constants: {constants}")
    print(f"expected total channels (consts.rs obs_shape(4)): {expected_total}")
    print(f"enumerated total channels: {total}")
    print()
    print(f"{'offset':>7} {'count':>6} {'end':>7}  {'kind':<12} name")
    for blk in blocks:
        kind = "river" if blk["is_river"] else "recency" if blk.get("is_recency_plane") else blk["group"]
        print(f"{blk['offset']:>7} {blk['count']:>6} {blk['end']:>7}  {kind:<12} {blk['name']}")
    print()
    print("cross-check refs (obs_repr.rs):")
    for blk in blocks:
        print(f"  [{blk['offset']:>4},{blk['end']:>4}) {blk['name']:<32} {' '.join(blk['refs'])}")

    if gaps or total != expected_total:
        print()
        print("MISMATCH: channel enumeration does not cover the v4 obs tensor.")
        for start, end in gaps:
            print(f"  gap interval [{start}, {end}) -> {end - start} unaccounted channel(s)")
        print(f"  enumerated={total} expected={expected_total} delta={expected_total - total}")
        print("Refusing to write a map with unaccounted channels; re-read obs_repr.rs.")
        return 1

    windows = [blk for blk in blocks if blk["is_river"]]
    recencies = [blk for blk in blocks if blk.get("is_recency_plane")]

    payload = {
        "schema": "riichi-obs-layout/v1",
        "description": (
            "Channel block map of libriichi's v4 observation tensor "
            f"({expected_total}, {tile_kinds}): {expected_total} channel planes x "
            f"{tile_kinds} tile kinds. Blocks are contiguous and cover [0, "
            f"{expected_total}) exactly."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": {
            "obs_repr": OBS_REPR_REF,
            "consts": CONSTS_REF,
            "vendor_commit": _vendor_commit(),
        },
        "version": 4,
        "shape": {"channels": expected_total, "tile_kinds": tile_kinds},
        "constants": {
            name: {"value": value, "ref": ref}
            for name, value, ref in (
                (
                    "SELF_KAWA_ITEM_CHANNELS",
                    constants["SELF_KAWA_ITEM_CHANNELS"],
                    f"{OBS_REPR_REF}:14",
                ),
                ("KAWA_ITEM_CHANNELS", constants["KAWA_ITEM_CHANNELS"], f"{OBS_REPR_REF}:15"),
                ("MAX_NUM_TURNS", constants["MAX_NUM_TURNS"], f"{OBS_REPR_REF}:16"),
            )
        },
        "derivation_notes": [
            "Channel offsets follow encode_obs() in obs_repr.rs top to bottom; the "
            "Rust assert_eq!(self.idx, self.arr.rows()) at :626 guarantees the same "
            "sum at runtime.",
            "IMPORTANT v4 subtlety: IntegerEncoder::encode's v4 branch (:92-104) "
            "honors only one_hot and rescale, NOT rbf_intervals. Blocks built with "
            ".rbf_intervals(...) but no .one_hot therefore consume 1 plane in v4 "
            "(honba, kyotaku, doras_owned, doras_unseen), not 2-4.",
            "Player-indexed planes use the engine's rotated/relative player order "
            "(player_state.rs:65): index 0 = self, 1 = shimocha (next), 2 = toimen "
            "(across), 3 = kamicha (previous).",
            "River windows are zero-padded: self window A consumes exactly "
            "6*SELF_KAWA_ITEM_CHANNELS and window B exactly 18*SELF_KAWA_ITEM_CHANNELS "
            "regardless of river length; same for opponents with 8 channels/item.",
            "Tile-kind column order (34) is the libriichi Tile ordering; see "
            "tile.rs:12-19 (1m..9m, 1p..9p, 1s..9s, E,S,W,N, white,green,red).",
        ],
        "totals": {
            "blocks": len(blocks),
            "sum_block_channels": total,
            "expected_channels": expected_total,
            "covered_channels": expected_total - sum(e - s for s, e in gaps),
            "gaps": [{"start": s, "end": e, "missing": e - s} for s, e in gaps],
            "matches_consts_obs_shape": total == expected_total and not gaps,
        },
        "river_blocks": {
            "note": (
                "todo 12 (A1) permutes the river window blocks while preserving "
                "per-block channel counts; the recency planes are the target of the "
                "recency-zero arm. Window blocks are item-encoded: channel "
                "offset + item_index * item_channels + plane; item plane layout is "
                "listed in each block's semantics. `kawa_overview_*` blocks (group "
                "river_overview) are order-insensitive tile-set summaries and are "
                "NOT valid permutation targets."
            ),
            "window_blocks": [
                {
                    "name": blk["name"],
                    "offset": blk["offset"],
                    "count": blk["count"],
                    "player": blk["river_player_name"],
                    "items": blk["items"],
                    "item_channels": blk["item_channels"],
                    "item_stride": blk["item_stride"],
                    "item_order": blk["item_order"],
                }
                for blk in windows
                if blk.get("window") in ("recent6", "old18")
            ],
            "recency_planes": [
                {
                    "name": blk["name"],
                    "offset": blk["offset"],
                    "count": blk["count"],
                    "player": blk["river_player_name"],
                }
                for blk in recencies
            ],
        },
        "blocks": blocks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print()
    print(
        f"OK: {total} channels across {len(blocks)} blocks; "
        f"wrote {args.output.relative_to(REPO_ROOT)}"
    )
    return 0


def _vendor_commit() -> str:
    """Best-effort vendor/mortal HEAD for provenance (never fatal)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT / "vendor/mortal"), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
