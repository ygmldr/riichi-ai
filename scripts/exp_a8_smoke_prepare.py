#!/usr/bin/env python3
"""A8 eval-path smoke 准备:合成一个 A8 检查点 + 4 局 /tmp 评估配置。

Purpose: exercise the exact real launch path (`scripts/exp_a8_launch.py eval`
-> hook -> `one_vs_three.main()` -> `brain_from_config` -> arena -> event feed)
before the 20k-step training and the 12k pool-A eval. The synthetic checkpoint
is an untrained branch-enabled E3a-shaped Brain (no training happened yet);
the smoke seed window is 40000, so pools A (10000) and B (50000) stay untouched.
"""

import os
import pathlib
import sys
import time
import tomllib

import toml
import torch

REPO = pathlib.Path('/home/ygsldr/riichi-ai')
SMOKE = pathlib.Path('/tmp/opencode/a8_smoke')
MORTAL_DIR = REPO / 'vendor/mortal/mortal'
LIB = REPO / 'vendor/mortal/target/release'
SCRIPTS = REPO / 'scripts'
for p in (str(SCRIPTS), str(MORTAL_DIR), str(LIB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import Brain, DQN, AuxNet  # noqa: E402


def main():
    SMOKE.mkdir(parents=True, exist_ok=True)
    with open(REPO / 'runs/exp/A8/eval.toml', 'rb') as f:
        cfg = tomllib.load(f)
    cfg['control']['state_file'] = str(SMOKE / 'state.pth')
    cfg['control']['best_state_file'] = str(SMOKE / 'best.pth')
    cfg['control']['tensorboard_dir'] = str(SMOKE / 'tb')
    cfg['test_play']['log_dir'] = str(SMOKE / 'test_play')
    cfg['train_play']['default']['log_dir'] = str(SMOKE / 'train_play')
    cfg['1v3']['log_dir'] = str(SMOKE / 'games')
    cfg['1v3']['seed_start'] = 40000
    cfg['1v3']['games_per_iter'] = 4
    cfg['1v3']['iters'] = 1
    cfg['1v3']['challenger']['state_file'] = str(SMOKE / 'state.pth')
    with open(SMOKE / 'eval.toml', 'w', encoding='utf-8') as f:
        toml.dump(cfg, f)

    torch.manual_seed(4242)
    brain = Brain(version=4, conv_channels=192, num_blocks=40, use_event_branch=True)
    dqn = DQN(version=4)
    aux_net = AuxNet((4,))
    state = {
        'mortal': brain.state_dict(),
        'current_dqn': dqn.state_dict(),
        'aux_net': aux_net.state_dict(),
        'steps': 0,
        'timestamp': time.time(),
        'best_perf': {'avg_rank': 4.0, 'avg_pt': -135.0},
        'config': cfg,
    }
    torch.save(state, SMOKE / 'state.pth')
    loaded = torch.load(SMOKE / 'state.pth', weights_only=True, map_location='cpu')
    assert loaded['config']['model']['use_event_branch'] is True
    assert loaded['steps'] == 0
    assert any(k.startswith('event_branch.') for k in loaded['mortal'])
    assert sum(p.numel() for p in brain.parameters()) == 11_818_752
    print(f'A8 SMOKE PREPARE OK: {SMOKE}/{{state.pth,eval.toml}} '
          f'(params={sum(p.numel() for p in brain.parameters()):,}, '
          f'seed_start={cfg["1v3"]["seed_start"]}, games={cfg["1v3"]["games_per_iter"]})')


if __name__ == '__main__':
    main()
