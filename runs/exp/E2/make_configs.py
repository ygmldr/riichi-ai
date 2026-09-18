#!/usr/bin/env python3
"""E2 (todo 17) config generator — never touches a live training path.

Mirrors `scripts/exp_e3a_make_configs.py`: reads the pre-patch-0007 repo
config (`configs/config.toml.bak-19`), applies exactly the E2 changes and
writes

  configs/exp_e2.toml      canonical E2 arm config (training); `[model]
                           use_event_branch = false` is written explicitly so
                           E2 can never accidentally instantiate the branch
  runs/exp/E2/train.toml   byte-identical copy handed to `MORTAL_CFG`
  runs/exp/E2/eval.toml    pool-A eval config (seed_key 0x5dfaa4cef265cd7,
                           seed_start 10000, 12,000 games, challenger =
                           runs/exp/E2/state.pth)

Changes vs the base config (everything else mirrors `runs/exp/E3a/train.toml`
so the arm comparison is not confounded by training conditions):
  `[resnet].conv_channels` 192 -> 202 (parameter match, `measure_params.py`)
  `[model].use_event_branch` absent -> false
  state/best/tb/test_play/train_play/1v3 paths -> runs/exp/E2/
  `[control].enable_compile` -> false (E3a used false; E2 must match)

Safety: every path training writes is asserted under `runs/exp/E2/` and
different from the live path; `use_event_branch` is asserted false.
"""

import argparse
import os
import tomllib

import toml

REPO = '/home/ygsldr/riichi-ai'
BASE_CONFIG = os.path.join(REPO, 'configs/config.toml.bak-19')
E2_DIR = f'{REPO}/runs/exp/E2'
CONV_CHANNELS = 202  # measured in runs/exp/E2/params.json (rel diff 0.3618 %)

LIVE = {
    'state_file': f'{REPO}/checkpoints/mortal.pth',
    'best_state_file': f'{REPO}/checkpoints/best.pth',
    'tensorboard_dir': f'{REPO}/runs',
    'test_play_log_dir': f'{REPO}/runs/test_play',
    'train_play_log_dir': f'{REPO}/runs/train_play',
}

SEED_KEY = 0x5dfaa4cef265cd7
CHAMPION = f'{REPO}/checkpoints/mortal.pth.at-148800'


def edit(cfg, path, value):
    keys = path.split('.')
    node = cfg
    for key in keys[:-1]:
        assert key in node, f'unknown section {path}'
        node = node[key]
    assert keys[-1] in node, f'unknown key {path}'
    node[keys[-1]] = value


def base_cfg():
    with open(BASE_CONFIG, 'rb') as f:
        cfg = tomllib.load(f)
    # `config.toml.bak-19` predates patch 0007 and has no `[model]` section;
    # E2's toggle is written explicitly, never inherited.
    cfg['model'] = {'use_event_branch': False}
    cfg['resnet']['conv_channels'] = CONV_CHANNELS
    return cfg


def apply_train_paths(cfg):
    edit(cfg, 'control.state_file', f'{E2_DIR}/state.pth')
    edit(cfg, 'control.best_state_file', f'{E2_DIR}/best.pth')
    edit(cfg, 'control.tensorboard_dir', f'{E2_DIR}/tb')
    edit(cfg, 'control.enable_compile', False)
    edit(cfg, 'test_play.log_dir', f'{E2_DIR}/test_play')
    edit(cfg, 'train_play.default.log_dir', f'{E2_DIR}/train_play')
    edit(cfg, '1v3.log_dir', f'{E2_DIR}/games')
    edit(cfg, '1v3.seed_start', 10000)


def apply_eval(cfg):
    apply_train_paths(cfg)
    edit(cfg, '1v3.seed_key', SEED_KEY)
    edit(cfg, '1v3.games_per_iter', 12000)
    edit(cfg, '1v3.iters', 1)
    edit(cfg, '1v3.challenger.state_file', f'{E2_DIR}/state.pth')
    edit(cfg, '1v3.challenger.name', 'e2')


def in_e2(path):
    return path.startswith(f'{E2_DIR}/')


def check_safe(cfg, kind):
    assert cfg['model'].get('use_event_branch') is False, cfg['model']
    assert cfg['resnet']['conv_channels'] == CONV_CHANNELS
    assert cfg['control']['state_file'] != LIVE['state_file'], 'state_file still live'
    assert cfg['control']['best_state_file'] != LIVE['best_state_file'], 'best_state_file still live'
    assert cfg['control']['tensorboard_dir'] != LIVE['tensorboard_dir'], 'tensorboard_dir still live'
    assert cfg['test_play']['log_dir'] != LIVE['test_play_log_dir'], 'test_play log_dir still live'
    assert cfg['train_play']['default']['log_dir'] != LIVE['train_play_log_dir'], 'train_play log_dir still live'
    for key in ('state_file', 'best_state_file', 'tensorboard_dir'):
        assert in_e2(cfg['control'][key]), cfg['control'][key]
    assert in_e2(cfg['test_play']['log_dir'])
    assert in_e2(cfg['train_play']['default']['log_dir'])
    assert in_e2(cfg['1v3']['log_dir']), cfg['1v3']['log_dir']
    assert cfg['control']['enable_compile'] is False
    assert cfg['dataset']['num_workers'] == 3
    assert cfg['dataset']['file_index'] == f'{REPO}/checkpoints/file_index.pth'
    assert cfg['1v3']['akochan']['enabled'] is False
    if kind == 'eval':
        assert cfg['1v3']['seed_key'] == SEED_KEY
        assert cfg['1v3']['seed_start'] == 10000
        assert cfg['1v3']['games_per_iter'] == 12000 and cfg['1v3']['iters'] == 1
        assert cfg['1v3']['challenger']['state_file'] == f'{E2_DIR}/state.pth'
        assert cfg['1v3']['champion']['state_file'] == CHAMPION
    if kind == 'train':
        assert cfg['control']['save_every'] == 2000
        assert cfg['control']['test_every'] == 100000
        assert cfg['control']['batch_size'] == 1024
        assert cfg['control']['opt_step_every'] == 1
        assert cfg['control']['online'] is False


def dump(cfg, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        toml.dump(cfg, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', nargs='?', choices=['train', 'eval', 'all'], default='all')
    args = ap.parse_args()

    if args.kind in ('train', 'all'):
        cfg = base_cfg()
        apply_train_paths(cfg)
        check_safe(cfg, 'train')
        dump(cfg, f'{REPO}/configs/exp_e2.toml')
        dump(cfg, f'{E2_DIR}/train.toml')
        print(f'OK train -> configs/exp_e2.toml + runs/exp/E2/train.toml '
              f'(conv_channels={CONV_CHANNELS}, use_event_branch=False, '
              f'state_file={cfg["control"]["state_file"]})')

    if args.kind in ('eval', 'all'):
        cfg = base_cfg()
        apply_eval(cfg)
        check_safe(cfg, 'eval')
        dump(cfg, f'{E2_DIR}/eval.toml')
        print(f'OK eval -> runs/exp/E2/eval.toml '
              f'(seed_key={hex(SEED_KEY)}, seed_start=10000, games=12000, '
              f'challenger={cfg["1v3"]["challenger"]["state_file"]}, '
              f'champion={cfg["1v3"]["champion"]["state_file"]}, '
              f'log_dir={cfg["1v3"]["log_dir"]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
