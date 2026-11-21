#!/usr/bin/env python3
"""E3a (todo 19) 临时 config 生成器 —— 绝不触碰 live 训练路径。

从改动前的 repo config(`configs/config.toml.bak-19`)读取,修改后重新写出三类
config,并断言所有危险路径都落在 runs/exp/E3a/ 或 /tmp/opencode/ 下:

  loadtest  : 8 文件小数据集 + batch 1024 + save_every 2,产出 E3a smoke 检查点
  train     : 20k 步正式训练(config 落在 runs/exp/E3a/train.toml)
  verify:*  : 五个加载点的验证场景(见 scripts/exp_e3a_verify_load.py)

安全断言(全部 kind):`[control].state_file` / `best_state_file` /
`tensorboard_dir` 与 `[test_play]`/`[train_play.*]` 的 log_dir **不得**等于
live 值(`checkpoints/mortal.pth` 等);`[1v3].log_dir` 必须是
`runs/exp/E3a/` 或 /tmp 下的独立目录。
"""

import argparse
import os
import tomllib

import toml

REPO = '/home/ygsldr/riichi-ai'
BASE_CONFIG = os.path.join(REPO, 'configs/config.toml.bak-19')
LIVE = {
    'state_file': '/home/ygsldr/riichi-ai/checkpoints/mortal.pth',
    'best_state_file': '/home/ygsldr/riichi-ai/checkpoints/best.pth',
    'tensorboard_dir': '/home/ygsldr/riichi-ai/runs',
    'test_play_log_dir': '/home/ygsldr/riichi-ai/runs/test_play',
    'train_play_log_dir': '/home/ygsldr/riichi-ai/runs/train_play',
}
PINNED = '/home/ygsldr/riichi-ai/runs/exp/PINNED/mortal.pth.at-175200'
OLD = '/home/ygsldr/riichi-ai/checkpoints/mortal.pth.at-148800'
SMOKE_STATE = '/tmp/opencode/t19/loadtest/state.pth'
VERIFY_DIR = '/tmp/opencode/t19/verify'
LOADTEST = '/tmp/opencode/t19/loadtest'
DATASET_DIR = '/tmp/opencode/t19/data'
GLOB = [f'{DATASET_DIR}/**/*.mjson']
MODEL_SECTION = {
    'use_event_branch': True,
    'event_dim': 128,
    'event_heads': 4,
    'event_layers': 4,
    'event_ffn': 512,
}


def edit(cfg, path, value):
    keys = path.split('.')
    node = cfg
    for key in keys[:-1]:
        assert key in node, f'unknown section {path}'
        node = node[key]
    assert keys[-1] in node, f'unknown key {path}'
    node[keys[-1]] = value


def generate(kind, out, scenario=None):
    with open(BASE_CONFIG, 'rb') as f:
        cfg = tomllib.load(f)
    # `config.toml.bak-19` is the pre-0007 tree; the E3a `[model]` section is
    # exactly the one patch 0007 adds to the live config.
    cfg.setdefault('model', dict(MODEL_SECTION))

    state = f'{VERIFY_DIR}/{scenario}/state.pth' if scenario else None
    engine_state = {'e3a': state, 'old': PINNED, 'guard': state, 'guard2': state}.get(scenario)

    if kind == 'loadtest':
        edit(cfg, 'control.state_file', f'{LOADTEST}/state.pth')
        edit(cfg, 'control.best_state_file', f'{LOADTEST}/best.pth')
        edit(cfg, 'control.tensorboard_dir', f'{LOADTEST}/tb')
        edit(cfg, 'control.enable_compile', False)
        edit(cfg, 'control.save_every', 2)
        edit(cfg, 'test_play.log_dir', f'{LOADTEST}/test_play')
        edit(cfg, 'dataset.file_index', f'{LOADTEST}/file_index.pth')
        edit(cfg, 'dataset.globs', GLOB)
        edit(cfg, 'dataset.val_globs', [])
        edit(cfg, 'dataset.num_workers', 0)
        edit(cfg, 'dataset.file_batch_size', 2)
        edit(cfg, 'train_play.default.log_dir', f'{LOADTEST}/train_play')
        edit(cfg, '1v3.log_dir', f'{LOADTEST}/1v3')
        assert cfg['model']['use_event_branch'] is True
    elif kind == 'train':
        edit(cfg, 'control.state_file', f'{REPO}/runs/exp/E3a/state.pth')
        edit(cfg, 'control.best_state_file', f'{REPO}/runs/exp/E3a/best.pth')
        edit(cfg, 'control.tensorboard_dir', f'{REPO}/runs/exp/E3a/tb')
        edit(cfg, 'control.enable_compile', False)
        edit(cfg, 'test_play.log_dir', f'{REPO}/runs/exp/E3a/test_play')
        edit(cfg, 'train_play.default.log_dir', f'{REPO}/runs/exp/E3a/train_play')
        edit(cfg, '1v3.log_dir', f'{REPO}/runs/exp/E3a/games')
        edit(cfg, '1v3.seed_start', 10000)
        assert cfg['model']['use_event_branch'] is True
    elif kind == 'eval':
        edit(cfg, 'control.state_file', f'{REPO}/runs/exp/E3a/state.pth')
        edit(cfg, 'control.best_state_file', f'{REPO}/runs/exp/E3a/best.pth')
        edit(cfg, 'control.tensorboard_dir', f'{REPO}/runs/exp/E3a/tb')
        edit(cfg, 'control.enable_compile', False)
        edit(cfg, 'test_play.log_dir', f'{REPO}/runs/exp/E3a/test_play')
        edit(cfg, 'train_play.default.log_dir', f'{REPO}/runs/exp/E3a/train_play')
        edit(cfg, '1v3.log_dir', f'{REPO}/runs/exp/E3a/games')
        edit(cfg, '1v3.seed_start', 10000)
        edit(cfg, '1v3.games_per_iter', 12000)
        edit(cfg, '1v3.iters', 1)
        edit(cfg, '1v3.seed_key', 0x5dfaa4cef265cd7)
        edit(cfg, '1v3.challenger.state_file', f'{REPO}/runs/exp/E3a/state.pth')
        edit(cfg, '1v3.challenger.enable_compile', False)
        edit(cfg, '1v3.champion.enable_compile', False)
        assert cfg['1v3']['akochan']['enabled'] is False
        assert cfg['model']['use_event_branch'] is True
    elif kind == 'verify':
        assert scenario in ('e3a', 'old', 'guard', 'guard2')
        edit(cfg, 'control.state_file', state)
        edit(cfg, 'control.best_state_file', f'/tmp/opencode/t19/verify/{scenario}/best.pth')
        edit(cfg, 'control.tensorboard_dir', f'/tmp/opencode/t19/verify/{scenario}/tb')
        edit(cfg, 'control.enable_compile', False)
        edit(cfg, 'control.save_every', 8)
        edit(cfg, 'control.batch_size', 256)
        edit(cfg, 'test_play.log_dir', f'/tmp/opencode/t19/verify/{scenario}/test_play')
        edit(cfg, 'dataset.file_index', f'{LOADTEST}/file_index.pth')
        edit(cfg, 'dataset.globs', GLOB)
        edit(cfg, 'dataset.val_globs', [])
        edit(cfg, 'dataset.num_workers', 0)
        edit(cfg, 'dataset.file_batch_size', 2)
        edit(cfg, 'train_play.default.log_dir', f'/tmp/opencode/t19/verify/{scenario}/train_play')
        edit(cfg, '1v3.log_dir', f'/tmp/opencode/t19/verify/{scenario}/1v3')
        edit(cfg, '1v3.seed_start', 50000)
        edit(cfg, '1v3.games_per_iter', 4)
        baseline_state = PINNED if scenario == 'guard2' else engine_state
        cfg['baseline']['test']['state_file'] = baseline_state
        cfg['baseline']['train']['state_file'] = baseline_state
        cfg['model']['use_event_branch'] = scenario == 'e3a'
        if scenario == 'guard2':
            cfg['1v3']['champion']['state_file'] = state
            cfg['1v3']['challenger']['state_file'] = PINNED
        else:
            cfg['1v3']['champion']['state_file'] = PINNED if scenario == 'guard' else engine_state
            cfg['1v3']['challenger']['state_file'] = engine_state
    else:
        raise SystemExit(f'unknown kind {kind}')

    with open(out, 'w', encoding='utf-8') as f:
        toml.dump(cfg, f)
    check_safe(kind, cfg, scenario)
    print(f'OK {kind}{":" + scenario if scenario else ""} -> {out} '
          f'(use_event_branch={cfg["model"]["use_event_branch"]}, seed_start={cfg["1v3"]["seed_start"]}, '
          f'state_file={cfg["control"]["state_file"]})')


def in_safe_root(path):
    return path.startswith(f'{REPO}/runs/exp/E3a/') or path.startswith('/tmp/opencode/')


def check_safe(kind, cfg, scenario):
    assert cfg['control']['state_file'] != LIVE['state_file'], 'state_file still live'
    assert cfg['control']['best_state_file'] != LIVE['best_state_file'], 'best_state_file still live'
    assert cfg['control']['tensorboard_dir'] != LIVE['tensorboard_dir'], 'tensorboard_dir still live'
    assert cfg['test_play']['log_dir'] != LIVE['test_play_log_dir'], 'test_play log_dir still live'
    assert cfg['train_play']['default']['log_dir'] != LIVE['train_play_log_dir'], 'train_play log_dir still live'
    assert in_safe_root(cfg['1v3']['log_dir']), cfg['1v3']['log_dir']
    if kind == 'loadtest':
        for key in ('state_file', 'best_state_file', 'tensorboard_dir'):
            assert in_safe_root(cfg['control'][key]), cfg['control'][key]
        assert cfg['model']['use_event_branch'] is True
        assert cfg['1v3']['seed_start'] == 30000
    if kind == 'eval':
        assert cfg['control']['state_file'] == f'{REPO}/runs/exp/E3a/state.pth'
        assert cfg['1v3']['log_dir'] == f'{REPO}/runs/exp/E3a/games'
        assert cfg['1v3']['seed_start'] == 10000
        assert cfg['1v3']['games_per_iter'] == 12000 and cfg['1v3']['iters'] == 1
        assert cfg['1v3']['seed_key'] == 0x5dfaa4cef265cd7
        assert cfg['1v3']['challenger']['state_file'] == f'{REPO}/runs/exp/E3a/state.pth'
        assert cfg['1v3']['champion']['state_file'] == OLD
        assert cfg['model']['use_event_branch'] is True
    if kind == 'train':
        assert cfg['control']['state_file'] == f'{REPO}/runs/exp/E3a/state.pth'
        assert cfg['control']['best_state_file'] == f'{REPO}/runs/exp/E3a/best.pth'
        assert cfg['control']['tensorboard_dir'] == f'{REPO}/runs/exp/E3a/tb'
        assert cfg['test_play']['log_dir'] == f'{REPO}/runs/exp/E3a/test_play'
        assert cfg['model']['use_event_branch'] is True
        assert cfg['1v3']['seed_start'] == 10000
        assert cfg['1v3']['games_per_iter'] == 12000 and cfg['1v3']['iters'] == 1
        assert cfg['1v3']['seed_key'] == 0x5dfaa4cef265cd7
    if kind == 'verify':
        state = f'{VERIFY_DIR}/{scenario}/state.pth'
        engine_state = {'e3a': state, 'old': PINNED, 'guard': state, 'guard2': state}[scenario]
        assert cfg['control']['state_file'] == state
        assert cfg['model']['use_event_branch'] is (scenario == 'e3a')
        baseline_state = PINNED if scenario == 'guard2' else engine_state
        assert cfg['baseline']['test']['state_file'] == baseline_state
        assert cfg['baseline']['train']['state_file'] == baseline_state
        if scenario == 'guard2':
            assert cfg['1v3']['champion']['state_file'] == state
            assert cfg['1v3']['challenger']['state_file'] == PINNED
        else:
            assert cfg['1v3']['champion']['state_file'] == (PINNED if scenario == 'guard' else engine_state)
            assert cfg['1v3']['challenger']['state_file'] == engine_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['loadtest', 'train', 'eval', 'verify'])
    ap.add_argument('--scenario', choices=['e3a', 'old', 'guard', 'guard2'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    generate(args.kind, args.out, args.scenario)


if __name__ == '__main__':
    main()
