#!/usr/bin/env python3
"""A8 模型定义自检(CPU,无 GPU 训练):证明 φ 置零真的生效且模型只听事件分支。

Checklist(全部写入 `runs/exp/A8/selfcheck.json`):
  1. hook 的零断言被触发(calls>0)且被丢弃的 `max|phi|>0`(零化不空洞);
  2. 挂钩后 `brain(obs, events)` == `actv(0 + W_zero(h_event))`(残差融合输入 φ 全零);
  3. 挂钩后输出**不依赖 obs**:两个不同 obs(同一 events)输出逐元素相同;
     未挂钩对照则随 obs 变化(证明差异确实来自 φ);
  4. 梯度:encoder 参数 `grad is None`,事件分支 `proj.weight.grad` 非零
     (CNN 不参与训练,模型只能从事件流学);
  5. 分支守卫不变:`events=None` 仍抛 ValueError;
  6. 只 hook 分支臂:安装后构造的 `use_event_branch=false` Brain 不被挂钩
     (A8 eval 里的冻结冠军不受影响)。
"""

import json
import os
import sys

import torch

REPO = '/home/ygsldr/riichi-ai'
MORTAL_DIR = os.path.join(REPO, 'vendor/mortal/mortal')
LIB = os.path.join(REPO, 'vendor/mortal/target/release')
for p in (os.path.join(REPO, 'scripts'), MORTAL_DIR, LIB):
    if p not in sys.path:
        sys.path.insert(0, p)

from libriichi.consts import obs_shape  # noqa: E402
import model  # noqa: E402
import exp_a8_phi_zero as pz  # noqa: E402

RESULT = {}
PROOF = '/tmp/opencode/a8_selfcheck_phi_zero.json'


def tmax(x):
    return float(x.detach().abs().max())


def main():
    torch.manual_seed(20260918)
    obs_shape_v4 = obs_shape(4)
    print(f'obs_shape(4) = {obs_shape_v4}')

    # control brain built BEFORE the patch: never hooked, must depend on obs
    ctrl = model.Brain(version=4, conv_channels=192, num_blocks=40, use_event_branch=True).eval()

    pz.install(proof_path=PROOF, mode='selfcheck')
    assert getattr(model.Brain, '_a8_phi_zero_installed', False)

    brain = model.Brain(version=4, conv_channels=192, num_blocks=40, use_event_branch=True).eval()
    encoder_params = list(brain.encoder.parameters())
    events = {
        'tokens': torch.stack([
            torch.randint(0, 20, (2, 16)),
            torch.randint(0, 35, (2, 16)),
            torch.randint(0, 5, (2, 16)),
            torch.randint(0, 385, (2, 16)),
            torch.randint(0, 8, (2, 16)),
        ], dim=-1),
        'lengths': torch.tensor([16, 11]),
    }
    events['tokens'][:, 0, 0] = 2  # start_kyoku

    obs_a = torch.randn(2, obs_shape_v4[0], 34)
    obs_b = obs_a.clone()
    obs_b[:, 0, 0] += 7.5  # same events, different CNN input

    assert pz.STATE['hooked_brains'] == 1, pz.STATE

    h = brain.event_branch(events)
    phi_real = brain.encoder.forward(obs_a)  # bypass the hook to inspect the discarded phi
    out_hooked = brain(obs_a, events=events)             # hook fires -> phi := 0
    out_hooked_b = brain(obs_b, events=events)           # different obs, same events

    RESULT['obs_shape_v4'] = list(obs_shape_v4)
    RESULT['phi_zero_calls'] = pz.STATE['calls']
    RESULT['max_abs_phi_before_zero'] = pz.STATE['max_abs_phi_before_zero']
    RESULT['max_abs_phi_real_bypass'] = tmax(phi_real)
    RESULT['numel_brain'] = sum(p.numel() for p in brain.parameters())
    RESULT['numel_encoder'] = sum(p.numel() for p in encoder_params)
    RESULT['numel_event_branch'] = sum(p.numel() for p in brain.event_branch.parameters())

    # 1) the hook really ran and the discarded phi was non-zero
    assert RESULT['phi_zero_calls'] > 0
    assert RESULT['max_abs_phi_before_zero'] > 0, RESULT
    RESULT['check_1_hook_ran_nonvacuous'] = True

    # 2a) zero-init W_zero: hooked output == actv(0), zeroed phi is exactly zero
    expected_zeroinit = brain.actv(torch.zeros_like(h))
    RESULT['check_2a_zeroinit_maxdiff'] = tmax(out_hooked - expected_zeroinit)
    assert RESULT['check_2a_zeroinit_maxdiff'] == 0.0
    RESULT['check_2a_zeroinit'] = True

    # 2b) random W_zero: hooked output == actv(W_zero(h_event)) exactly
    torch.nn.init.normal_(brain.event_branch.proj.weight, std=0.05)
    out_hooked = brain(obs_a, events=events)
    expected_random = brain.actv(brain.event_branch(events))
    RESULT['check_2b_branch_only_maxdiff'] = tmax(out_hooked - expected_random)
    identity = brain.actv(phi_real + brain.event_branch(events))
    RESULT['check_2b_vs_identity_maxdiff'] = tmax(out_hooked - identity)
    assert RESULT['check_2b_branch_only_maxdiff'] < 1e-5, RESULT
    assert RESULT['check_2b_vs_identity_maxdiff'] > 1e-3, RESULT
    RESULT['check_2b_branch_only'] = True

    # 3) hooked output ignores obs entirely (same events)
    out_hooked_b = brain(obs_b, events=events)
    RESULT['check_3_obs_invariance_maxdiff'] = tmax(out_hooked - out_hooked_b)
    assert RESULT['check_3_obs_invariance_maxdiff'] == 0.0, RESULT
    RESULT['check_3_obs_invariance'] = True

    # 3b) control: an un-hooked Brain *does* depend on obs
    ctrl.event_branch.load_state_dict(brain.event_branch.state_dict())
    ctrl_out_a = ctrl(obs_a, events=events)
    ctrl_out_b = ctrl(obs_b, events=events)
    RESULT['check_3b_control_obs_sensitivity'] = tmax(ctrl_out_a - ctrl_out_b)
    assert RESULT['check_3b_control_obs_sensitivity'] > 1e-3, RESULT
    RESULT['check_3b_control_obs_sensitivity_ok'] = True
    assert pz.STATE['hooked_brains'] == 1, pz.STATE

    # 4) gradients: encoder receives none, the event branch does
    brain.zero_grad(set_to_none=True)
    loss = brain(obs_a, events=events).sum()
    loss.backward()
    enc_grads = [p.grad is None for p in encoder_params]
    RESULT['check_4_encoder_params_without_grad'] = sum(enc_grads)
    RESULT['check_4_encoder_params_total'] = len(enc_grads)
    RESULT['check_4_proj_grad_norm'] = float(brain.event_branch.proj.weight.grad.norm())
    assert all(enc_grads), 'encoder params should have no gradient under A8'
    assert RESULT['check_4_proj_grad_norm'] > 0, RESULT
    RESULT['check_4_grad_route'] = True

    # 5) the branch guard is untouched
    try:
        brain(obs_a, events=None)
        raise AssertionError('expected ValueError for events=None')
    except ValueError as e:
        RESULT['check_5_guard_message'] = str(e)
    RESULT['check_5_guard'] = True

    # 6) branch-less Brains are not hooked
    before = pz.STATE['hooked_brains']
    model.Brain(version=4, conv_channels=192, num_blocks=40, use_event_branch=False)
    RESULT['check_6_branchless_not_hooked'] = pz.STATE['hooked_brains'] == before
    assert RESULT['check_6_branchless_not_hooked']

    payload = pz.dump_proof()
    RESULT['proof'] = payload
    RESULT['all_pass'] = True

    os.makedirs(os.path.join(REPO, 'runs/exp/A8'), exist_ok=True)
    with open(os.path.join(REPO, 'runs/exp/A8/selfcheck.json'), 'w') as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))
    print('A8 SELFCHECK: ALL PASS')


if __name__ == '__main__':
    main()
