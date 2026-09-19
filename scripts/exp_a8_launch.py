#!/usr/bin/env python3
"""A8 launcher: install the `phi`-zero hook, then run stock train/eval code.

    python3 scripts/exp_a8_launch.py train     # -> mortal/train.py  main()
    python3 scripts/exp_a8_launch.py eval      # -> mortal/one_vs_three.py main()

Safety re-asserted here (in addition to the generator): `MORTAL_CFG` must point
inside `runs/exp/A8/`, none of the five live paths may appear in it, and in
eval mode `MORTAL_EVT_DEBUG` must be set (the todo-18-C event-feed gate).
The proof JSON is written by `exp_a8_phi_zero.py` at exit; after `main()`
returns, `assert_armed()` fails loudly if no branch-enabled Brain was hooked.
"""

import os
import pathlib
import sys
import tomllib

REPO = pathlib.Path('/home/ygsldr/riichi-ai')
ARM = REPO / 'runs/exp/A8'
SMOKE_DIR = pathlib.Path('/tmp/opencode/a8_smoke')
MORTAL_DIR = REPO / 'vendor/mortal/mortal'
LIB = REPO / 'vendor/mortal/target/release'
SCRIPTS = REPO / 'scripts'
LIVE = {
    'state_file': REPO / 'checkpoints/mortal.pth',
    'best_state_file': REPO / 'checkpoints/best.pth',
    'tensorboard_dir': REPO / 'runs',
    'test_play_log_dir': REPO / 'runs/test_play',
    'train_play_log_dir': REPO / 'runs/train_play',
}


def check_config(path, smoke=False):
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    c = cfg['control']
    if smoke:
        assert pathlib.Path(path).resolve().is_relative_to(SMOKE_DIR), f'smoke MORTAL_CFG outside {SMOKE_DIR}'
        root = SMOKE_DIR.resolve()
    else:
        root = ARM.resolve()
    for key in ('state_file', 'best_state_file', 'tensorboard_dir'):
        value = pathlib.Path(c[key])
        assert not value.resolve() == LIVE[key].resolve(), f'{key} is a LIVE path: {value}'
        assert value.resolve().is_relative_to(root), f'{key} outside {root}: {value}'
    for log_dir in (cfg['test_play']['log_dir'], cfg['train_play']['default']['log_dir'],
                    cfg['1v3']['log_dir']):
        assert pathlib.Path(log_dir).resolve().is_relative_to(root), f'{log_dir} outside {root}'
    if not smoke:
        assert pathlib.Path(cfg['1v3']['log_dir']).resolve() == (ARM / 'games').resolve()
        assert cfg['1v3']['seed_start'] == 10000, 'pool A only; pool B 50000 is reserved for todo 22'
        assert cfg['1v3']['games_per_iter'] == 12000 and cfg['1v3']['iters'] == 1
    else:
        assert cfg['1v3']['seed_start'] == 40000, 'smoke window; pools A/B must stay untouched'
        assert cfg['1v3']['games_per_iter'] == 4 and cfg['1v3']['iters'] == 1
    assert c['batch_size'] == 1024 and c['opt_step_every'] == 1
    assert c['save_every'] == 2000 and c['test_every'] == 100000
    assert c['enable_compile'] is False and c['enable_amp'] is True
    assert cfg['model']['use_event_branch'] is True, 'A8 must reuse E3a architecture'
    assert cfg['dataset']['num_workers'] == 3
    assert cfg['1v3']['seed_key'] == 0x5dfaa4cef265cd7
    print(f'A8 CONFIG SAFETY OK{"" if not smoke else " (SMOKE)"}: MORTAL_CFG={path}')
    for key in ('state_file', 'best_state_file', 'tensorboard_dir'):
        print(f'  {key} = {c[key]}')
    print(f'  test_play.log_dir = {cfg["test_play"]["log_dir"]}')
    print(f'  train_play.log_dir = {cfg["train_play"]["default"]["log_dir"]}')
    print(f'  1v3.log_dir = {cfg["1v3"]["log_dir"]} (seed_start={cfg["1v3"]["seed_start"]})')
    print(f'  model.use_event_branch = {cfg["model"]["use_event_branch"]} (phi zeroed by launcher hook)')
    return cfg


def main():
    assert len(sys.argv) == 2 and sys.argv[1] in ('train', 'eval'), 'usage: exp_a8_launch.py train|eval'
    mode = sys.argv[1]
    smoke = os.environ.get('A8_SMOKE') == '1'

    cfg_path = os.environ.get('MORTAL_CFG')
    if not cfg_path:
        raise SystemExit('A8 launcher: MORTAL_CFG is required (never use the repo config)')
    cfg_file = pathlib.Path(cfg_path).resolve()
    if smoke:
        assert cfg_file.is_relative_to(SMOKE_DIR), f'smoke MORTAL_CFG outside {SMOKE_DIR}: {cfg_file}'
    else:
        assert cfg_file.is_relative_to(ARM.resolve()), f'MORTAL_CFG outside {ARM}: {cfg_file}'
    check_config(cfg_file, smoke=smoke)

    if mode == 'eval':
        debug = os.environ.get('MORTAL_EVT_DEBUG')
        assert debug, 'A8 eval must set MORTAL_EVT_DEBUG (todo-18-C feed gate)'
        print(f'A8 EVAL FEED GATE: MORTAL_EVT_DEBUG={debug}')

    proof = os.environ.get('A8_PHI_ZERO_PROOF', str(ARM / f'phi_zero_proof_{mode}.json'))
    os.environ['A8_PHI_ZERO_PROOF'] = proof

    for p in (str(SCRIPTS), str(MORTAL_DIR), str(LIB)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # install BEFORE importing the vendor modules that build Brains
    from exp_a8_phi_zero import install, assert_armed, dump_proof
    install(proof_path=proof, mode=mode)

    if mode == 'train':
        import train as mortal_train
        mortal_train.main()
    else:
        import one_vs_three
        one_vs_three.main()

    assert_armed()
    payload = dump_proof()
    print('A8 PHI-ZERO PROOF:')
    for key in ('installed', 'hooked_brains', 'calls', 'max_abs_phi_before_zero',
                'total_zeroed_elements', 'first_call_ts', 'last_call_ts', 'armed'):
        print(f'  {key} = {payload[key]}')
    print(f'  proof_json = {proof}')


if __name__ == '__main__':
    main()
