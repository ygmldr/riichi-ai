#!/usr/bin/env python3
"""E3a (todo 19a) 五个 Brain 加载点逐个验证: (a) E3a 检查点 / (b) 旧检查点 / (c) 守卫。

每个场景一个进程(MORTAL_CFG 在 import 前固定),进程内真实执行各入口:

  train         : train.train()                      —— 8 文件小数据集, 真跑若干步
  one_vs_three  : one_vs_three.main()                 —— 4 局 arena(champion + challenger 两个加载点)
  player        : player.TestPlayer()/TrainPlayer()   —— 两个构造函数
  mortal        : mortal.main()(空 stdin, 只走到加载点)
  client        : client.main()(桩掉 socket/recv/TrainPlayer)

monkeypatch `model.load_brain_state_dict` 记录**真实调用**: 返回的
missing_keys / unexpected_keys / 分支是否存在 / 分支参数 |·| 和;守卫触发的调用
记录 `raised` + 错误消息。场景结束后由本脚本内建断言校验(见 validate)。

用法:
  python3 scripts/exp_e3a_verify_load.py prepare
  for s in e3a old guard guard2; do
      MORTAL_CFG=/tmp/opencode/t19/verify_$s.toml \
      python3 scripts/exp_e3a_verify_load.py run --scenario $s --out /tmp/opencode/t19/load_$s.json
  done
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import types

import numpy as np
import torch

REPO = '/home/ygsldr/riichi-ai'
MORTAL_DIR = os.path.join(REPO, 'vendor/mortal/mortal')
VERIFY_DIR = '/tmp/opencode/t19/verify'
LOADTEST_STATE = '/tmp/opencode/t19/loadtest/state.pth'
PINNED = os.path.join(REPO, 'runs/exp/PINNED/mortal.pth.at-175200')

EXPECTED_LOADS = {'train': 2, 'one_vs_three': 2, 'player': 2, 'mortal': 1, 'client': 1}
SITE_NAMES = ('train', 'one_vs_three', 'player', 'mortal', 'client')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    base = torch.load(LOADTEST_STATE, weights_only=True, map_location='cpu')
    assert any(k.startswith('event_branch.') for k in base['mortal'])
    assert base['config']['model']['use_event_branch'] is True
    os.makedirs(f'{VERIFY_DIR}/e3a', exist_ok=True)
    os.makedirs(f'{VERIFY_DIR}/old', exist_ok=True)
    for scenario in ('guard', 'guard2'):
        os.makedirs(f'{VERIFY_DIR}/{scenario}', exist_ok=True)
        tampered = {k: v for k, v in base.items()}
        tampered['config'] = json.loads(json.dumps(base['config']))
        tampered['config']['model']['use_event_branch'] = False
        torch.save(tampered, f'{VERIFY_DIR}/{scenario}/state.pth')
    shutil.copyfile(LOADTEST_STATE, f'{VERIFY_DIR}/e3a/state.pth')
    shutil.copyfile(PINNED, f'{VERIFY_DIR}/old/state.pth')
    for scenario in ('e3a', 'old', 'guard', 'guard2'):
        path = f'{VERIFY_DIR}/{scenario}/state.pth'
        st = torch.load(path, weights_only=True, map_location='cpu')
        print(json.dumps({
            'scenario': scenario,
            'state': path,
            'size': os.path.getsize(path),
            'sha256': sha256(path),
            'steps': st['steps'],
            'has_event_branch_weights': any(k.startswith('event_branch.') for k in st['mortal']),
            'config_use_event_branch': st['config'].get('model', {}).get('use_event_branch', False),
        }))


def install_recorder():
    import model

    original = model.load_brain_state_dict
    records = []

    def wrapper(mortal, state_dict, *, source='checkpoint'):
        branch = getattr(mortal, 'event_branch', None)
        record = {
            'source': os.path.basename(str(source)),
            'state_has_event_branch': any(k.startswith('event_branch.') for k in state_dict),
            'branch_present': branch is not None,
            'branch_param_count': sum(p.numel() for p in branch.parameters()) if branch else 0,
            'status': 'loaded',
        }
        try:
            inc = original(mortal, state_dict, source=source)
        except RuntimeError as ex:
            record['status'] = 'raised'
            record['error'] = str(ex)
            records.append(record)
            raise
        record['missing_keys'] = list(inc.missing_keys)
        record['unexpected_keys'] = list(inc.unexpected_keys)
        record['branch_abs_sum'] = (
            float(sum(p.detach().abs().sum() for p in branch.parameters())) if branch else 0.0
        )
        records.append(record)
        return inc

    model.load_brain_state_dict = wrapper
    return wrapper, records


def run(scenario, out_path):
    sys.path.insert(0, MORTAL_DIR)
    wrapper, records = install_recorder()
    import client
    import mortal as mortal_mod
    import one_vs_three
    import player
    import train

    for mod in (client, mortal_mod, one_vs_three, player, train):
        if hasattr(mod, 'load_brain_state_dict'):
            mod.load_brain_state_dict = wrapper

    def site_mortal():
        old_argv, old_stdin = sys.argv, sys.stdin
        sys.argv = ['mortal.py', '0']
        sys.stdin = io.StringIO('')
        try:
            mortal_mod.main()
        finally:
            sys.argv, sys.stdin = old_argv, old_stdin

    def site_player():
        errors = {}
        for name, ctor in (('TestPlayer', player.TestPlayer), ('TrainPlayer', player.TrainPlayer)):
            try:
                ctor()
            except Exception as ex:  # noqa: BLE001
                errors[name] = f'{type(ex).__name__}: {ex}'
        return errors

    def site_client():
        class FakeTrainPlayer:
            def train_play(self, mortal, dqn, device):
                return np.array([1, 0, 0, 0]), []

        class DummyConn:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def connect(self, addr):
                pass

        calls = {'recv': 0}

        def fake_recv(conn, **kwargs):
            calls['recv'] += 1
            if calls['recv'] > 1:
                raise SystemExit(0)
            st = torch.load(f'{VERIFY_DIR}/{scenario}/state.pth', weights_only=True, map_location='cpu')
            return {'status': 'ok', 'param_version': 1, 'mortal': st['mortal'], 'dqn': st['current_dqn']}

        client.socket = types.SimpleNamespace(socket=lambda *a, **kw: DummyConn())
        client.send_msg = lambda conn, msg: None
        client.recv_msg = fake_recv
        client.TrainPlayer = FakeTrainPlayer
        client.main()

    sites = {
        'train': train.train,
        'one_vs_three': one_vs_three.main,
        'player': site_player,
        'mortal': site_mortal,
        'client': site_client,
    }
    site_results = {}
    for name in SITE_NAMES:
        before = len(records)
        extra = None
        try:
            extra = sites[name]()
            error = None
        except SystemExit:
            error = None
        except RuntimeError as ex:
            error = f'RuntimeError: {ex}'
        site_results[name] = {
            'error': error,
            'extra': extra,
            'loads': records[before:],
        }

    payload = {'scenario': scenario, 'mort_cfg': os.environ.get('MORTAL_CFG'), 'sites': site_results}
    validate(scenario, payload)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    return payload


def validate(scenario, payload):
    sites = payload['sites']
    total_loads = sum(len(sites[name]['loads']) for name in SITE_NAMES)
    if scenario in ('e3a', 'old'):
        for name in SITE_NAMES:
            site = sites[name]
            assert site['error'] is None, f'{scenario}/{name} errored: {site["error"]}'
            assert len(site['loads']) == EXPECTED_LOADS[name], (
                f'{scenario}/{name}: {len(site["loads"])} loads, want {EXPECTED_LOADS[name]}'
            )
            for rec in site['loads']:
                assert rec['status'] == 'loaded', rec
                assert rec['missing_keys'] == [], rec
                assert rec['unexpected_keys'] == [], rec
                if scenario == 'e3a':
                    assert rec['branch_present'] and rec['branch_abs_sum'] > 0, rec
                else:
                    assert not rec['branch_present'] and rec['branch_abs_sum'] == 0.0, rec
        if scenario == 'e3a':
            assert sites['player']['extra'] == {}, sites['player']['extra']
    else:
        for name in SITE_NAMES:
            site = sites[name]
            raised = [r for r in site['loads'] if r['status'] == 'raised']
            loaded = [r for r in site['loads'] if r['status'] == 'loaded']
            if scenario == 'guard2' and name == 'player':
                assert not raised and site['extra'] == {} and len(loaded) == 2, site
                continue
            assert raised, f'{scenario}/{name}: no guard raise recorded: {site}'
            for rec in raised:
                assert 'event-branch weights' in rec['error'], rec
            for rec in loaded:
                assert not rec['branch_present'], rec
                assert rec['missing_keys'] == [] and rec['unexpected_keys'] == [], rec
        for name in ('train', 'mortal', 'client'):
            assert sites[name]['error'] and sites[name]['error'].startswith('RuntimeError'), sites[name]
        assert len([r for r in sites['one_vs_three']['loads'] if r['status'] == 'raised']) == 1, sites['one_vs_three']
        if scenario == 'guard':
            player_site = sites['player']
            assert player_site['extra'] and all('RuntimeError' in v for v in player_site['extra'].values()), player_site
    print(f'VALIDATE OK {scenario}: {total_loads} loads across {len(SITE_NAMES)} sites')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'run'])
    ap.add_argument('--scenario')
    ap.add_argument('--out', default='/tmp/opencode/t19/load.json')
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        assert args.scenario, '--scenario required'
        run(args.scenario, args.out)


if __name__ == '__main__':
    main()
