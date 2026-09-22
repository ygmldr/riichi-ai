"""检查点加载。复用 vendor/mortal/mortal 的既有构造路径,不自造网络。

参考调用序列: vendor/mortal/mortal/mortal.py:29-59、player.py:16-41。
"""
import dataclasses
import hashlib
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MORTAL_DIR = REPO_ROOT / "vendor/mortal/mortal"
if str(_MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(_MORTAL_DIR))

from model import DQN, AuxNet, GRP, brain_from_config, load_brain_state_dict  # noqa: E402

CHECKPOINT_FIXTURE = str(REPO_ROOT / "runs/exp/PINNED/mortal.pth.at-396000")
GRP_CHECKPOINT = str(REPO_ROOT / "checkpoints/grp.pth")
GRP_NETWORK = {"hidden_size": 64, "num_layers": 2}


def checkpoint_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclasses.dataclass
class Bundle:
    brain: object
    dqn: object
    aux_net: object
    grp: object
    version: int
    steps: int
    sha256: str
    cfg: dict
    device: str


def _load_grp(device: str, grp_path: str = GRP_CHECKPOINT):
    grp = GRP(**GRP_NETWORK).to(device).eval()
    state = torch.load(grp_path, weights_only=True, map_location="cpu")
    grp.load_state_dict(state["model"])
    return grp


def load_bundle(checkpoint_path: str, grp_dir=None, device: str = "cuda:0") -> Bundle:
    torch.backends.cudnn.benchmark = False
    state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    cfg = state["config"]
    version = int(cfg["control"].get("version", 1))

    brain = brain_from_config(cfg).to(device).eval()
    dqn = DQN(version=version).to(device).eval()
    load_brain_state_dict(brain, state["mortal"], source=checkpoint_path)
    dqn.load_state_dict(state["current_dqn"])

    aux_net = None
    if "aux_net" in state:
        aux_net = AuxNet((4,)).to(device).eval()
        aux_net.load_state_dict(state["aux_net"])

    grp = _load_grp(device, grp_dir or GRP_CHECKPOINT)
    return Bundle(brain=brain, dqn=dqn, aux_net=aux_net, grp=grp, version=version,
                  steps=int(state.get("steps", -1)),
                  sha256=checkpoint_sha256(checkpoint_path), cfg=cfg, device=device)
