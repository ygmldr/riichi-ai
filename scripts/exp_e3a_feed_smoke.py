#!/usr/bin/env python3
"""E3a (todo 19b) 评估路径投喂 + 分支生效 smoke(短程 arena,用 E3a 检查点)。

在真实 `one_vs_three` 路径上:
  1. 统计每个进入 Python 引擎的决策点的 token 张量(行数 / 非零数 / 首末 type);
  2. 断言全部非空非零(与 EVAL_FEED 门同口径,只统计引擎评估的决策);
  3. 取第一个 batch,比较 `brain(obs, events=ev)` 与纯 CNN 路径
     `actv(encoder(obs))` —— 训练后的分支必须让 1024-d 嵌入发生改变(diff > 0),
     证明"分支真的被喂到并参与决策",而不是静默退化成 E2。

用法:
  MORTAL_CFG=<eval.toml> python3 scripts/exp_e3a_feed_smoke.py \
      --sets 2 --out runs/exp/E3a/EVAL_FEED_E3A_smoke.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = '/home/ygsldr/riichi-ai'
sys.path.insert(0, os.path.join(REPO, 'vendor/mortal/mortal'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', type=int, default=2)
    ap.add_argument('--out', default=os.path.join(REPO, 'runs/exp/E3a/EVAL_FEED_E3A_smoke.json'))
    ap.add_argument('--log-dir', default='/tmp/opencode/t19/feed_smoke_games')
    args = ap.parse_args()

    from event_data import collate_events
    from engine import MortalEngine
    from model import DQN, brain_from_config, load_brain_state_dict
    from config import config
    from libriichi.arena import OneVsThree
    from one_vs_three import seed_interval

    def build_engine(which):
        cfg = config['1v3'][which]
        state = torch.load(cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
        brain = brain_from_config(state['config']).eval()
        dqn = DQN(version=state['config']['control'].get('version', 1)).eval()
        load_brain_state_dict(brain, state['mortal'], source=cfg['state_file'])
        dqn.load_state_dict(state['current_dqn'])
        engine = MortalEngine(
            brain, dqn,
            is_oracle=False,
            version=state['config']['control'].get('version', 1),
            device=torch.device(cfg['device']),
            enable_amp=cfg['enable_amp'],
            enable_rule_based_agari_guard=cfg['enable_rule_based_agari_guard'],
            name=cfg['name'],
        )
        return engine, brain, dqn

    engine_chal, brain, _ = build_engine('challenger')
    engine_champ, _, _ = build_engine('champion')
    assert brain.event_branch is not None, 'challenger checkpoint resolved to a branch-less Brain'

    samples = []
    first = {}

    def hook(tag, engine):
        orig = engine.react_batch

        def hooked(obs, masks, invisible_obs, events=None):
            if events is None:
                raise SystemExit(f'{tag}: react_batch called without events')
            for i, e in enumerate(events):
                a = np.asarray(e, dtype=np.int16)
                samples.append({
                    'tag': tag, 'rows': int(a.shape[0]),
                    'nz': int(np.count_nonzero(a)),
                    'first_type': int(a[0, 0]) if a.shape[0] else -1,
                    'last_type': int(a[-1, 0]) if a.shape[0] else -1,
                })
            if tag == 'chal' and 'obs' not in first:
                first.update(obs=obs, events=events)
            return orig(obs, masks, invisible_obs, events)

        engine.react_batch = hooked

    hook('chal', engine_chal)
    hook('champ', engine_champ)

    seed_start, _ = seed_interval(config['1v3'])
    env = OneVsThree(disable_progress_bar=False, log_dir=args.log_dir)
    rankings = env.py_vs_py(
        challenger=engine_chal,
        champion=engine_champ,
        seed_start=(seed_start, config['1v3']['seed_key']),
        seed_count=args.sets,
    )

    bad = [s for s in samples if s['rows'] < 1 or s['nz'] == 0]
    with torch.no_grad(), torch.autocast(engine_chal.device.type, enabled=engine_chal.enable_amp):
        obs = torch.as_tensor(np.stack(first['obs'], axis=0), device=engine_chal.device)
        ev = {k: v.to(engine_chal.device) for k, v in collate_events(first['events']).items()}
        phi_full = brain(obs, events=ev)
        phi_base = brain.actv(brain.encoder(obs))
    branch_diff = float((phi_full - phi_base).abs().max())

    payload = {
        'sets': args.sets,
        'games': args.sets * 4,
        'seed_start': seed_start,
        'seed_key': config['1v3']['seed_key'],
        'rankings': list(rankings),
        'engine_samples': len(samples),
        'empty_or_zero_samples': len(bad),
        'all_nonempty_nonzero': not bad,
        'first_batch_branch_diff_max': branch_diff,
        'branch_changes_embedding': branch_diff > 0.0,
        'n_range': [min(s['rows'] for s in samples), max(s['rows'] for s in samples)],
        'first_types': sorted({s['first_type'] for s in samples}),
        'failures': bad[:5],
    }
    assert payload['all_nonempty_nonzero'], payload
    assert payload['branch_changes_embedding'], payload
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
