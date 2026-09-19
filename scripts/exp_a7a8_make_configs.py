#!/usr/bin/env python3
"""A7/A8 (todo 21) 临时 config 生成器 —— 绝不触碰 live 训练路径。

Provenance: `runs/exp/E3a/{train,eval}.toml` are the authoritative "matched
conditions" source for todo 21 (`[Runs/exp/E3a/PREREG.md] §5`): dataset globs
2009-2022 + 2023 holdout, `batch_size=1024`, `opt_step_every=1`,
`save_every=2000`, `test_every=100000`, `num_workers=3`,
`enable_compile=false`, `enable_amp=true`, unchanged optimizer/scheduler.
This generator deep-copies those files and changes ONLY:

  - every path under `runs/exp/E3a/` -> `runs/exp/A7/` or `runs/exp/A8/`;
  - A7: `[model] use_event_branch=false` (no event_* keys, mirroring E2's
    config convention -- the branch is never instantiated) and
    `[resnet] conv_channels=192` (E3a's exact CNN trunk);
  - A8: nothing but the paths -- the E3a `[model]` section (toggle=true,
    d=128/4 heads/4 layers/ffn=512) is kept verbatim. A8 zeroes `phi` at
    runtime in a launcher-level hook (`scripts/exp_a8_phi_zero.py`), so the
    config must stay E3a-identical.

Safety: asserts that none of the five live paths (`state_file`,
`best_state_file`, `tensorboard_dir`, `test_play.log_dir`,
`train_play.default.log_dir`) is used, that every writable path lives under
the arm's own directory, that `[dataset].file_index` stays the read-only
shared index, that the champion is the pinned `at-148800` snapshot, that
`[1v3].seed_start == 10000` (pool A; pool B 50000 must NOT be consumed) and
that `akochan.enabled` is false. Prints a per-key diff against E3a so the
"only the expected conditions differ" claim is auditable.

用法: python3 scripts/exp_a7a8_make_configs.py [--arm A7|A8|both] [--kind train|eval|both]
"""

import argparse
import os
import tomllib

import toml

REPO = '/home/ygsldr/riichi-ai'
E3A = f'{REPO}/runs/exp/E3a'
ARMS = ('A7', 'A8')
KINDS = ('train', 'eval')
LIVE = {
    'state_file': f'{REPO}/checkpoints/mortal.pth',
    'best_state_file': f'{REPO}/checkpoints/best.pth',
    'tensorboard_dir': f'{REPO}/runs',
    'test_play_log_dir': f'{REPO}/runs/test_play',
    'train_play_log_dir': f'{REPO}/runs/train_play',
}
PINNED = f'{REPO}/runs/exp/PINNED/mortal.pth.at-175200'
OLD = f'{REPO}/checkpoints/mortal.pth.at-148800'
SEED_KEY = 0x5dfaa4cef265cd7


def rewrite(obj, arm_root):
    """把 E3a 目录下的所有字符串路径改写为本臂目录。"""
    if isinstance(obj, dict):
        return {k: rewrite(v, arm_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v, arm_root) for v in obj]
    if isinstance(obj, str) and obj.startswith(f'{E3A}/'):
        return arm_root + obj[len(E3A):]
    return obj


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        p = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flat(v, p))
        else:
            out[p] = v
    return out


def diff_vs_e3a(cfg, e3a_cfg):
    """返回 (key, arm_value, e3a_value) 列表,路径改写后再比对。"""
    a, b = flat(cfg), flat(e3a_cfg)
    keys = sorted(set(a) | set(b))
    return [(k, a.get(k, '<absent>'), b.get(k, '<absent>')) for k in keys if a.get(k) != b.get(k)]


def build(arm, kind):
    src = f'{E3A}/{kind}.toml'
    with open(src, 'rb') as f:
        cfg = tomllib.load(f)
    arm_root = f'{REPO}/runs/exp/{arm}'
    cfg = rewrite(cfg, arm_root)

    if arm == 'A7':
        cfg['model'] = {'use_event_branch': False}
        assert cfg['resnet']['conv_channels'] == 192, cfg['resnet']
        assert cfg['resnet']['num_blocks'] == 40, cfg['resnet']
    elif arm == 'A8':
        assert cfg['model']['use_event_branch'] is True
        assert cfg['model'] == {
            'use_event_branch': True, 'event_dim': 128, 'event_heads': 4,
            'event_layers': 4, 'event_ffn': 512,
        }, cfg['model']
    else:
        raise SystemExit(f'unknown arm {arm}')

    if kind == 'eval':
        cfg['1v3']['challenger']['state_file'] = f'{arm_root}/state.pth'

    return cfg


def check_safe(arm, kind, cfg):
    arm_root = f'{REPO}/runs/exp/{arm}'
    c = cfg['control']
    assert c['state_file'] != LIVE['state_file'], 'state_file still live'
    assert c['best_state_file'] != LIVE['best_state_file'], 'best_state_file still live'
    assert c['tensorboard_dir'] != LIVE['tensorboard_dir'], 'tensorboard_dir still live'
    assert cfg['test_play']['log_dir'] != LIVE['test_play_log_dir'], 'test_play log_dir still live'
    assert cfg['train_play']['default']['log_dir'] != LIVE['train_play_log_dir'], 'train_play log_dir still live'
    for key in ('state_file', 'best_state_file', 'tensorboard_dir'):
        assert c[key].startswith(f'{arm_root}/'), (key, c[key])
    for key in ('test_play', 'train_play'):
        log_dir = cfg[key]['default']['log_dir'] if key == 'train_play' else cfg[key]['log_dir']
        assert log_dir.startswith(f'{arm_root}/'), (key, log_dir)
    assert cfg['1v3']['log_dir'] == f'{arm_root}/games', cfg['1v3']['log_dir']
    assert cfg['1v3']['seed_start'] == 10000, cfg['1v3']['seed_start']
    assert cfg['1v3']['seed_key'] == SEED_KEY, cfg['1v3']['seed_key']
    assert cfg['1v3']['games_per_iter'] == 12000 and cfg['1v3']['iters'] == 1
    assert cfg['1v3']['akochan']['enabled'] is False
    assert cfg['1v3']['champion']['state_file'] == OLD, cfg['1v3']['champion']['state_file']
    assert cfg['dataset']['file_index'] == f'{REPO}/checkpoints/file_index.pth'
    assert cfg['dataset']['globs'] == [f'{REPO}/datasets/extracted/{y}/**/*.mjson' for y in range(2009, 2023)]
    assert cfg['dataset']['val_globs'] == [f'{REPO}/datasets/extracted/2023/**/*.mjson']
    assert cfg['control']['batch_size'] == 1024
    assert cfg['control']['opt_step_every'] == 1
    assert cfg['control']['save_every'] == 2000
    assert cfg['control']['test_every'] == 100000
    assert cfg['control']['enable_compile'] is False
    assert cfg['control']['enable_amp'] is True
    assert cfg['dataset']['num_workers'] == 3
    assert cfg['freeze_bn']['mortal'] is False
    assert cfg['optim']['scheduler'] == {'peak': 0.0001, 'final': 0.0001, 'warm_up_steps': 0, 'max_steps': 0}
    if kind == 'eval':
        assert cfg['1v3']['challenger']['state_file'] == f'{arm_root}/state.pth'
    if kind == 'train':
        assert cfg['1v3']['challenger']['state_file'] == PINNED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=(*ARMS, 'both'), default='both')
    ap.add_argument('--kind', choices=(*KINDS, 'both'), default='both')
    args = ap.parse_args()

    arms = ARMS if args.arm == 'both' else (args.arm,)
    kinds = KINDS if args.kind == 'both' else (args.kind,)

    with open(f'{E3A}/train.toml', 'rb') as f:
        e3a_train = tomllib.load(f)
    with open(f'{E3A}/eval.toml', 'rb') as f:
        e3a_eval = tomllib.load(f)
    e3a_cfgs = {'train': e3a_train, 'eval': e3a_eval}

    for arm in arms:
        os.makedirs(f'{REPO}/runs/exp/{arm}', exist_ok=True)
        for kind in kinds:
            cfg = build(arm, kind)
            check_safe(arm, kind, cfg)
            out = f'{REPO}/runs/exp/{arm}/{kind}.toml'
            with open(out, 'w', encoding='utf-8') as f:
                toml.dump(cfg, f)
            # 读回断言:落盘内容与内存一致、且路径断言仍成立
            with open(out, 'rb') as f:
                back = tomllib.load(f)
            check_safe(arm, kind, back)
            diffs = diff_vs_e3a(back, e3a_cfgs[kind])
            print(f'OK {arm}/{kind} -> {out}')
            print(f'   use_event_branch={back["model"]["use_event_branch"]} '
                  f'conv_channels={back["resnet"]["conv_channels"]} '
                  f'seed_start={back["1v3"]["seed_start"]} '
                  f'state_file={back["control"]["state_file"]}')
            if diffs:
                for k, arm_v, e3a_v in diffs:
                    print(f'   DIFF {k}: {arm_v!r}  (E3a: {e3a_v!r})')
            else:
                print('   DIFF vs E3a: none (bit-identical after path rewrite)')
        print()


if __name__ == '__main__':
    main()
