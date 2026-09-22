"""单元 2: 确定性重放 —— 由 libriichi.dataset.Gameplay 提供。

为什么用 Gameplay 而不是 mjai.Bot:
  - Bot 内部私有持有 PlayerState,且需要构造 engine;Gameplay 直接给出每决策点的
    obs / mask / 日志中记录的动作,重放期**没有任何模型参与选择动作**。
已知限制: PlayerState 不暴露 scores/dora/场风/牌河 ⇒ 公共状态从事件流解析。
"""
import dataclasses
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from libriichi.dataset import GameplayLoader  # noqa: E402

from jiangpu import layout  # noqa: E402

_VERSION = 4


@dataclasses.dataclass
class Decision:
    index: int          # **全局唯一**:跨 POV 连续计数(见 iter_decisions 的 idx)
    player_id: int
    obs: np.ndarray
    mask: np.ndarray
    legal_actions: list  # 升序动作索引(由 mask 派生)
    logged_action: int
    at_kyoku: int
    at_turn: int
    shanten: int
    done: bool
    public: dict


def public_by_kyoku(record) -> dict:
    """从事件流解析每局的公共状态(PlayerState 不暴露这些)。"""
    out = {}
    for ev in record.events:
        if ev["type"] == "start_kyoku":
            k = len(out)
            out[k] = {
                "bakaze": ev["bakaze"],
                "kyoku": int(ev["kyoku"]),
                "honba": int(ev["honba"]),
                "kyotaku": int(ev["kyotaku"]),
                "oya": int(ev["oya"]),
                "scores": [int(s) for s in ev["scores"]],
                "dora_markers": [ev["dora_marker"]],
            }
        elif ev["type"] == "dora" and out:
            out[max(out)]["dora_markers"].append(ev["dora_marker"])
    return out


def tiles_left_from_obs(obs: np.ndarray) -> int:
    off, count = layout.span(layout.load_layout(), "tiles_left")
    row = obs[off:off + count]
    return int(round(float(row.max()) * 69))   # obs_repr.rs 编码为 tiles_left/69


def iter_decisions(record, pov="log"):
    loader = GameplayLoader(version=_VERSION, oracle=False)
    per_file = loader.load_gz_log_files([record.path])
    if not per_file or len(per_file) != 1 or len(per_file[0]) != 4:
        raise RuntimeError(f"GameplayLoader 未返回 [1 文件][4 视角]: {record.path}")

    pub = public_by_kyoku(record)
    wanted = [0, 1, 2, 3] if pov == "log" else [int(pov)]

    out = []
    for pid in wanted:
        game = per_file[0][pid]
        obs = game.take_obs()
        masks = game.take_masks()
        actions = game.take_actions()
        at_kyoku = game.take_at_kyoku()
        at_turns = game.take_at_turns()
        shantens = game.take_shantens()
        dones = game.take_dones()
        lens = {len(obs), len(masks), len(actions), len(at_kyoku),
                len(at_turns), len(shantens), len(dones)}
        if len(lens) != 1:
            raise RuntimeError(f"Gameplay 各序列长度不一致: {record.path} pov={pid}: {lens}")
        for i in range(len(obs)):
            k = int(at_kyoku[i])
            if k not in pub:
                raise RuntimeError(f"at_kyoku={k} 在事件流中找不到对应 start_kyoku")
            m = np.asarray(masks[i], dtype=np.bool_)
            out.append(Decision(
                index=len(out),          # 全局唯一,跨 POV 不重复(provider 以它为字典键)
                player_id=pid,
                obs=np.asarray(obs[i], dtype=np.float32),
                mask=m,
                legal_actions=[int(j) for j, ok in enumerate(m.tolist()) if ok],
                logged_action=int(actions[i]),
                at_kyoku=k,
                at_turn=int(at_turns[i]),
                shanten=int(shantens[i]),
                done=bool(dones[i]),
                public=pub[k],
            ))
    return out
