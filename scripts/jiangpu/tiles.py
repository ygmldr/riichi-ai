"""牌 id ↔ mjai 字符串。顺序逐字对齐 libriichi tile.rs:11-19 的 MJAI_PAI_STRINGS。

Python 侧此前不存在此映射,故本模块是唯一来源。
"""
TILE_STRINGS = (
    [f"{n}m" for n in range(1, 10)]
    + [f"{n}p" for n in range(1, 10)]
    + [f"{n}s" for n in range(1, 10)]
    + ["E", "S", "W", "N", "P", "F", "C"]
    + ["5mr", "5pr", "5sr"]
    + ["?"]
)
_IDS = {s: i for i, s in enumerate(TILE_STRINGS[:37])}
_AKA = {34, 35, 36}


def tile_str(tid: int) -> str:
    return TILE_STRINGS[tid]


def tile_id(s: str) -> int:
    return _IDS[s]


def deaka(tid: int) -> int:
    return {34: 4, 35: 13, 36: 22}.get(tid, tid)


def is_aka(tid: int) -> bool:
    return tid in _AKA
