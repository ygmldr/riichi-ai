#!/usr/bin/env python3
"""E2 (todo 17) parameter measurement: pure-CNN arm vs E3a fusion arm.

Run with the training env active (needs `libriichi` for `obs_shape`):

    source scripts/activate_env.sh
    python3 runs/exp/E2/measure_params.py

What it does (all measured with `sum(p.numel())`, never parsed from logs):
  1. baseline CNN (c=192, b=40) and E3a (same + event branch) numel, asserted
     against the numbers todo 19a wrote into `runs/exp/E3a/PREREG.md` §3;
  2. a closed-form analytic CNN numel, asserted equal to the measured one
     (this is the "pure-CNN analytic value" the branch-absent proof uses);
  3. a scan of `conv_channels` around the analytic estimate, so todo 17's
     choice is data, not a guess;
  4. branch-absent proof for the chosen channel count: `event_branch is None`
     and numel == analytic CNN value.

The chosen `conv_channels` must satisfy
`|total_E2 - total_E3a| / total_E3a <= 5%`.
"""

import argparse
import json
import os
import sys

import torch

REPO = '/home/ygsldr/riichi-ai'
sys.path.insert(0, os.path.join(REPO, 'vendor/mortal/mortal'))

from libriichi.consts import obs_shape  # noqa: E402
from model import Brain  # noqa: E402

VERSION = 4
BASELINE_C = 192
NUM_BLOCKS = 40

# todo 19a measured values (`runs/exp/E3a/PREREG.md` §3, `selfcheck.json`)
BASELINE_EXPECTED = 10_787_456
E3A_ADDED_EXPECTED = 1_031_296
E3A_TOTAL_EXPECTED = 11_818_752
TOLERANCE = 0.05

EVENT_KW = dict(event_dim=128, event_heads=4, event_layers=4, event_ffn=512)


def numel(model):
    return sum(p.numel() for p in model.parameters())


def build_cnn(conv_channels):
    return Brain(version=VERSION, conv_channels=conv_channels, num_blocks=NUM_BLOCKS)


def build_e3a(conv_channels=BASELINE_C):
    return Brain(
        version=VERSION,
        conv_channels=conv_channels,
        num_blocks=NUM_BLOCKS,
        use_event_branch=True,
        **EVENT_KW,
    )


def analytic_cnn(conv_channels, in_channels):
    """Closed form of the branch-less `Brain(version=4)` parameter count.

    ResNet(pre_actv=True): one bias-less stem conv, `num_blocks` residual
    blocks (2 x [BN, act, bias-less 3x3 conv] = 6C^2 + 4C, plus the channel-
    attention MLPs 2*C*(C//16) + C//16 + C), a trailing BN (2C), the 3x3
    conv to 32 channels with bias (96C + 32), and the 1088->1024 head
    (1088*1024 + 1024). version 4 has no latent heads.
    """
    per_block = 6 * conv_channels * conv_channels + 4 * conv_channels
    per_block += 2 * conv_channels * (conv_channels // 16) + conv_channels // 16 + conv_channels
    stem = 3 * conv_channels * in_channels
    tail = 2 * conv_channels
    tail += 32 * conv_channels * 3 + 32
    tail += 32 * 34 * 1024 + 1024
    return stem + NUM_BLOCKS * per_block + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', help='optional JSON output path')
    args = ap.parse_args()

    in_channels = obs_shape(VERSION)[0]

    # --- 1. both arms, re-derived from the model definition (patch 0007) ---
    cnn_base = build_cnn(BASELINE_C)
    e3a = build_e3a()
    n_base = numel(cnn_base)
    n_e3a = numel(e3a)
    n_added = n_e3a - n_base
    assert cnn_base.event_branch is None, 'baseline CNN unexpectedly has an event branch'
    assert n_base == BASELINE_EXPECTED, (n_base, BASELINE_EXPECTED)
    assert n_e3a == E3A_TOTAL_EXPECTED, (n_e3a, E3A_TOTAL_EXPECTED)
    assert n_added == E3A_ADDED_EXPECTED, (n_added, E3A_ADDED_EXPECTED)

    # --- 2. analytic CNN formula vs measurement ---
    an_base = analytic_cnn(BASELINE_C, in_channels)
    assert an_base == n_base, (an_base, n_base)

    print(f'obs_shape({VERSION}) = ({in_channels}, 34)')
    print('measured with sum(p.numel()):')
    print(f'  baseline CNN (c={BASELINE_C}, b={NUM_BLOCKS}) : {n_base:,}')
    print(f'  E3a added by the event branch          : {n_added:,}')
    print(f'  E3a total                              : {n_e3a:,}')
    print(f'  analytic CNN(c={BASELINE_C})                    : {an_base:,}  (== measured)')
    print()

    # --- 3. scan conv_channels ---
    target = E3A_TOTAL_EXPECTED
    print(f'target total_E2 = total_E3a = {target:,}  (tolerance {TOLERANCE:.0%})')
    print(f'{"c":>5} {"numel":>12} {"diff":>10} {"rel":>8}  <=5%')
    rows = []
    best = None
    for c in range(190, 226):
        n = numel(build_cnn(c))
        diff = n - target
        rel = abs(diff) / target
        ok = rel <= TOLERANCE
        rows.append({'conv_channels': c, 'numel': n, 'diff': diff, 'rel': rel, 'within_5pct': ok})
        if best is None or abs(diff) < abs(best['diff']):
            best = rows[-1]
        mark = 'yes' if ok else 'no'
        print(f'{c:>5} {n:>12,} {diff:>+10,} {rel:>8.4%}  {mark}')
    print()

    chosen = best['conv_channels']
    chosen_numel = best['numel']
    rel = abs(chosen_numel - target) / target
    assert rel <= TOLERANCE, (chosen, chosen_numel, rel)
    print(f'chosen conv_channels = {chosen}  (analytic estimate was 201, not 210)')
    print(f'  total_E2            = {chosen_numel:,}')
    print(f'  diff vs E3a total   = {chosen_numel - target:+,} ({chosen_numel / target - 1:+.4%})')
    print(f'  rel |diff|          = {rel:.4%}  (<= {TOLERANCE:.0%}: PASS)')
    print(f'  added params, E2    = 0 (no event branch)')
    print(f'  added params, E3a   = {n_added:,}')
    print(f'  added-parameter difference (E3a - E2) = {n_added:,}')

    # --- 4. branch-absent proof at the chosen channel count ---
    e2 = build_cnn(chosen)
    assert e2.event_branch is None, 'E2 Brain instantiated an event branch!'
    n_e2 = numel(e2)
    an_e2 = analytic_cnn(chosen, in_channels)
    assert n_e2 == chosen_numel
    assert n_e2 == an_e2, (n_e2, an_e2)
    assert not any(k.startswith('event_branch.') for k in e2.state_dict())
    print()
    print('branch-absent proof for E2:')
    print(f'  Brain(use_event_branch=False).event_branch is None : True')
    print(f'  measured numel == analytic CNN numel               : {n_e2:,} == {an_e2:,}')
    print(f'  event_branch.* tensors in state_dict               : 0')
    print(f'  toggle-on control (E3a) event_branch is not None   : {e3a.event_branch is not None}')
    print()
    print(f'RESULT conv_channels={chosen} total_E2={n_e2} total_E3a={n_e3a} '
          f'rel={rel:.6f} within_5pct={rel <= TOLERANCE}')

    if args.out:
        payload = {
            'obs_shape': list(obs_shape(VERSION)),
            'num_blocks': NUM_BLOCKS,
            'baseline_cnn': {'conv_channels': BASELINE_C, 'numel': n_base},
            'e3a': {'conv_channels': BASELINE_C, 'numel': n_e3a, 'added': n_added},
            'target_total': target,
            'tolerance': TOLERANCE,
            'scan': rows,
            'chosen': {
                'conv_channels': chosen,
                'total_e2': n_e2,
                'analytic_cnn': an_e2,
                'diff_vs_e3a': n_e2 - target,
                'rel': rel,
                'branch_is_none': True,
                'event_branch_tensors': 0,
            },
        }
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
