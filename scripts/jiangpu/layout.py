"""读 runs/analysis/obs_layout.json 得到 block 名 → (offset, count)。

注意: scripts/obs_layout.py **不导出**任何偏移常量,它只是生成器。
本仓既有的正确访问模式是 exp_a1_shuffle.py:83-84 的 name→block 字典。
"""
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT = REPO_ROOT / "runs/analysis/obs_layout.json"
_CACHE: dict = {}


def load_layout(path=None) -> dict:
    key = str(path or DEFAULT_LAYOUT)
    if key not in _CACHE:
        raw = json.loads(pathlib.Path(key).read_text(encoding="utf-8"))
        if raw.get("schema") != "riichi-obs-layout/v1":
            raise ValueError(f"unexpected layout schema: {raw.get('schema')!r}")
        _CACHE[key] = {b["name"]: b for b in raw["blocks"]}
    return _CACHE[key]


def span(top: dict, name: str) -> tuple:
    b = top[name]
    return int(b["offset"]), int(b["count"])
