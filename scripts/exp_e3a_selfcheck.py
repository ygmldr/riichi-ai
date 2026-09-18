#!/usr/bin/env python3
"""E3a (todo 19a) 前向自检: 零初始化残差融合恒等 + 因果 mask + 参数表。

命令: python3 scripts/exp_e3a_selfcheck.py [--out runs/exp/E3a/selfcheck.json]

断言(全部 exit != 0 on failure):
  1. 零初始化恒等: 事件分支刚构造(W_zero == 0)时,`phi_out` 与同权重纯 CNN
     基线逐元素差 < 1e-5 —— 对全零 token 与真实 token 都成立。
  2. 残差真实接线: 把 `proj.weight` 扰动后输出必须改变(防"死分支"假恒等)。
  3. 因果性: 改变序列中**未来** token 不影响更早位置的表示;改变更早 token 则影响。
  4. padding: 长度之外的 token 内容不影响池化输出。
  5. 开关: `use_event_branch=false` 时分支模块不存在;`true` 且 `events=None` 报错。
  6. 加载守卫: checkpoint 带事件权重而 Brain 无分支 -> RuntimeError;反向 -> missing key。
  7. numel 表(实测)。
"""

import argparse
import json
import os
import sys
import time

import torch

REPO = '/home/ygsldr/riichi-ai'
sys.path.insert(0, os.path.join(REPO, 'vendor/mortal/mortal'))

from event_data import collate_events  # noqa: E402
from libriichi.consts import obs_shape  # noqa: E402
from model import Brain, has_event_branch_weights, load_brain_state_dict  # noqa: E402

CONV_CHANNELS = 192
NUM_BLOCKS = 40
OBS_CHANNELS = obs_shape(4)[0]
EVENT_KW = dict(event_dim=128, event_heads=4, event_layers=4, event_ffn=512)

RESULTS = {}


def check(name, ok, detail):
    RESULTS[name] = {'ok': bool(ok), 'detail': detail}
    print(('PASS' if ok else 'FAIL') + f' {name}: {detail}')
    if not ok:
        raise SystemExit(f'ASSERTION FAILED: {name}: {detail}')


def make_events(streams, pad=0):
    return collate_events(streams)


def build_cnn():
    return Brain(version=4, conv_channels=CONV_CHANNELS, num_blocks=NUM_BLOCKS)


def build_e3a():
    return Brain(
        version=4, conv_channels=CONV_CHANNELS, num_blocks=NUM_BLOCKS,
        use_event_branch=True, **EVENT_KW,
    )


def cnn_state_from(sd):
    return {k: v for k, v in sd.items() if not k.startswith('event_branch.')}


def test_identity(e3a, cnn, obs):
    zero = make_events([[0] * 1 for _ in range(obs.shape[0])])
    real = make_events([
        [[2, 5, 0, 0, 0], [3, 1, 0, 1, 0], [4, 1, 0, 1, 0], [10, 9, 4, 1, 0]],
        [[2, 12, 1, 0, 0], [3, 3, 1, 1, 1], [11, 34, 1, 1, 0]],
    ])
    with torch.no_grad():
        base = cnn(obs)
        out_zero = e3a(obs, events=zero)
        out_real = e3a(obs, events=real)
        direct = e3a.actv(e3a.encoder(obs))
    d_zero = (out_zero - base).abs().max().item()
    d_real = (out_real - base).abs().max().item()
    d_direct = (base - direct).abs().max().item()
    check(
        'zero_init_identity',
        max(d_zero, d_real, d_direct) < 1e-5,
        f'max|phi_out - phi_base| zero_tokens={d_zero:.3e} real_tokens={d_real:.3e} '
        f'cnn_vs_direct={d_direct:.3e} (threshold 1e-5)',
    )
    proj = e3a.event_branch.proj.weight
    proj_max = float(proj.detach().abs().max())
    check('proj_zero_init', proj_max == 0.0, f'max|W_zero|={proj_max}')
    return real


def test_residual_wired(e3a, cnn, obs):
    stream = [[2, 5, 0, 0, 0], [3, 1, 0, 1, 0], [4, 1, 0, 1, 0]]
    ev = make_events([stream] * obs.shape[0])
    with torch.no_grad():
        base = cnn(obs)
        e3a.event_branch.proj.weight.normal_(0, 0.02)
        out = e3a(obs, events=ev)
        diff = (out - base).abs().max().item()
    check('residual_wired', diff > 1e-6, f'max|phi_out - phi_base| after perturb = {diff:.3e}')


def test_causal(e3a):
    layer = e3a.event_branch.layers[0].eval()
    x = torch.randn(1, 6, e3a.event_branch.dim)
    causal = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    x_future = x.clone()
    x_future[0, 5] = torch.randn(e3a.event_branch.dim) * 5
    x_past = x.clone()
    x_past[0, 3] = torch.randn(e3a.event_branch.dim) * 5
    with torch.no_grad():
        y = layer(x, src_mask=causal)[0]
        y_future = layer(x_future, src_mask=causal)[0]
        y_past = layer(x_past, src_mask=causal)[0]
    d_future = (y[:4] - y_future[:4]).abs().max().item()
    d_past = (y[4:] - y_past[4:]).abs().max().item()
    check(
        'causal_mask',
        d_future == 0.0 and d_past > 1e-6,
        f'changing pos>=4 leaves pos<4 unchanged ({d_future:.3e}); '
        f'changing pos 3 moves pos>=4 ({d_past:.3e})',
    )


def test_padding(e3a):
    branch = e3a.event_branch
    stream = [[2, 5, 0, 0, 0], [3, 1, 0, 1, 0], [4, 1, 0, 1, 0]]
    longer = stream + [[3, 2, 1, 1, 0], [4, 3, 1, 1, 0], [11, 20, 1, 1, 0]]
    ev = make_events([stream, longer])
    ev_junk = make_events([stream, longer])
    ev_junk['tokens'][0, 3:] = torch.tensor(
        [[9, 33, 3, 200, 7], [2, 1, 2, 3, 4], [13, 5, 4, 100, 1]], dtype=torch.int16,
    )
    with torch.no_grad():
        h = branch(ev)
        h_junk = branch(ev_junk)
    d = (h - h_junk).abs().max().item()
    check('padding_ignored', d == 0.0, f'max|h - h(junk padding)| = {d:.3e}')


def test_toggle_and_guard():
    cnn = build_cnn()
    check('toggle_off_no_branch', cnn.event_branch is None, 'use_event_branch=false -> event_branch is None')
    e3a = build_e3a()
    check('toggle_on_branch', e3a.event_branch is not None, 'use_event_branch=true -> branch instantiated')
    obs = torch.randn(1, OBS_CHANNELS, 34)
    try:
        e3a(obs, events=None)
        ok, msg = False, 'no error raised'
    except ValueError as ex:
        ok, msg = 'events' in str(ex), str(ex)
    check('branch_needs_events', ok, msg)

    fresh = build_e3a()
    base_sd = cnn.state_dict()
    try:
        load_brain_state_dict(cnn, e3a.state_dict(), source='unit:e3a->cnn')
        guard_ok, guard_msg = False, 'no error raised'
    except RuntimeError as ex:
        guard_ok, guard_msg = 'event-branch weights' in str(ex), str(ex)
    check('guard_branch_weights_into_cnn', guard_ok, guard_msg)

    inc = load_brain_state_dict(fresh, e3a.state_dict(), source='unit:e3a->e3a')
    check(
        'e3a_load_clean',
        not inc.missing_keys and not inc.unexpected_keys,
        f'missing={list(inc.missing_keys)} unexpected={list(inc.unexpected_keys)}',
    )
    inc = load_brain_state_dict(cnn, base_sd, source='unit:cnn->cnn')
    check(
        'cnn_load_clean',
        not inc.missing_keys and not inc.unexpected_keys,
        f'missing={list(inc.missing_keys)} unexpected={list(inc.unexpected_keys)}',
    )
    inc = load_brain_state_dict(build_e3a(), base_sd, source='unit:cnn->e3a')
    check(
        'cnn_into_e3a_missing_branch',
        bool(inc.missing_keys) and not inc.unexpected_keys
        and all(k.startswith('event_branch.') for k in inc.missing_keys),
        f'missing={len(inc.missing_keys)} keys (all event_branch), unexpected={list(inc.unexpected_keys)}',
    )
    return e3a, cnn


def test_numel():
    cnn = build_cnn()
    e3a = build_e3a()
    cnn_total = sum(p.numel() for p in cnn.parameters())
    e3a_total = sum(p.numel() for p in e3a.parameters())
    added = e3a_total - cnn_total
    check(
        'numel_consistency',
        has_event_branch_weights(e3a.state_dict()) and not has_event_branch_weights(cnn.state_dict()),
        f'baseline={cnn_total:,} e3a={e3a_total:,} added={added:,} '
        f'(ratio {(added / cnn_total * 100):.3f}% of baseline)',
    )
    return {'baseline_cnn_total': cnn_total, 'e3a_added': added, 'e3a_total': e3a_total}


def test_timing(e3a, cnn):
    if not torch.cuda.is_available():
        return {'skipped': 'cuda unavailable'}
    device = torch.device('cuda:0')
    e3a, cnn = e3a.to(device).eval(), cnn.to(device).eval()
    obs = torch.randn(1024, OBS_CHANNELS, 34, device=device)
    stream = [[2, 5, 0, 0, 0]] + [[3, 1, 0, 1, 0], [4, 1, 0, 1, 0]] * 50
    ev = {k: v.to(device) for k, v in make_events([stream] * 1024).items()}
    out = {}
    with torch.no_grad(), torch.autocast(device.type, enabled=True):
        for tag, fn in (('baseline', lambda: cnn(obs)), ('e3a', lambda: e3a(obs, events=ev))):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            out[tag] = (time.perf_counter() - t0) / 10
    out['ratio_e3a_over_baseline'] = out['e3a'] / out['baseline']
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'runs/exp/E3a/selfcheck.json'))
    args = ap.parse_args()

    torch.manual_seed(0)
    e3a = build_e3a().eval()
    cnn = build_cnn().eval()
    cnn.load_state_dict(cnn_state_from(e3a.state_dict()))
    obs = torch.randn(2, OBS_CHANNELS, 34)

    real = test_identity(e3a, cnn, obs)
    test_residual_wired(e3a, cnn, obs)
    test_causal(e3a)
    test_padding(e3a)
    test_toggle_and_guard()
    numel = test_numel()
    timing = test_timing(build_e3a(), build_cnn())

    payload = {
        'numel': numel,
        'timing_cuda_b1024_t153': timing,
        'checks': RESULTS,
        'all_pass': all(r['ok'] for r in RESULTS.values()),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'-> {args.out}')
    if not payload['all_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
