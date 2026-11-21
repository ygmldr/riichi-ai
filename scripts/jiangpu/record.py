"""单元 1: 牌谱读取与规范化。

输入 = gzip 包着的 mjai NDJSON(每行一个事件)。本模块负责:
  - 拿到完整原文(GRP 需要 raw log 字符串,见 libriichi dataset/grp.rs load_log)
  - 解析成事件列表
  - 校验完整性(恰一个 start_game / end_game)与身份(文件名 <seed>_<key>_<split>)
"""
import dataclasses
import gzip
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from eval_report import FILE_RE  # noqa: E402  复用既有正则,不重写

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FROZEN_FIXTURE = str(REPO_ROOT / "runs/1v3/10219_579411421376059130_d.json.gz")


class RecordError(Exception):
    """牌谱不完整、格式非法或身份不符。"""


@dataclasses.dataclass
class Record:
    path: str
    raw_text: str
    events: list
    names: list
    seed: int
    key: int
    split: str


def _parse_filename(path: str):
    m = FILE_RE.match(pathlib.Path(path).name)
    if m is None:
        raise RecordError(f"文件名不符合 <seed>_<key>_<split>.json.gz: {pathlib.Path(path).name}")
    return int(m.group("seed")), int(m.group("key")), m.group("split")


def read_record(path: str) -> Record:
    seed, key, split = _parse_filename(path)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            raw_text = fh.read()
    except OSError as exc:
        raise RecordError(f"无法读取 gzip: {path}: {exc}") from exc

    events = []
    for lineno, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecordError(f"{path}:{lineno} 非法 JSON: {exc}") from exc
        if not isinstance(ev, dict) or "type" not in ev:
            raise RecordError(f"{path}:{lineno} 事件缺少 type 字段")
        events.append(ev)

    starts = [e for e in events if e["type"] == "start_game"]
    ends = [e for e in events if e["type"] == "end_game"]
    if len(starts) != 1:
        raise RecordError(f"start_game 数量应为 1,实为 {len(starts)}")
    if len(ends) != 1:
        raise RecordError(f"end_game 数量应为 1,实为 {len(ends)}")
    if events[-1]["type"] != "end_game":
        raise RecordError(f"最后一个事件应为 end_game,实为 {events[-1]['type']}")

    names = starts[0].get("names")
    if not isinstance(names, list) or len(names) != 4:
        raise RecordError(f"start_game.names 应为长度 4 的列表,实为 {names!r}")

    return Record(path=path, raw_text=raw_text, events=events,
                  names=[str(n) for n in names], seed=seed, key=key, split=split)
