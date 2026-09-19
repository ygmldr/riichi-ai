#!/usr/bin/env python3
"""A8 (todo 21): zero `phi` immediately before E3a's residual fusion.

Implementation is **launcher-level** (vendor/ stays untouched, per A1's
precedent and the F2 cumulative-patch audit): a PyTorch forward hook on
`Brain.encoder`, registered by wrapping `Brain.__init__`, replaces the encoder
output with a fresh all-zero tensor of identical shape/dtype/device.
`Brain.forward` reads that hook return value at `phi = self.encoder(obs)`, so
the very next statement -- `phi = phi + W_zero(h_event)` -- sees exactly zeros:
A8's classifier input is `W_zero(h_event)` alone (the event branch is the only
path that can influence the output; the CNN receives no gradient either).

The hook, on **every** call:
  1. builds `zeroed = torch.zeros_like(output)`,
  2. asserts `zeroed` is all-zero,
  3. records the magnitude of the discarded `phi` (proving the encoder output
     was genuinely non-zero, i.e. the test is not vacuous),
and writes a proof JSON at process exit. `assert_armed()` is called by both
launchers after the run: an eval that silently ran branch-less would be a
false negative, so `hooked_brains >= 1 and calls > 0` is mandatory.

Only branch-enabled `Brain` instances are hooked: the pinned champion
(`at-148800`, `use_event_branch=false`) inside A8's eval is deliberately
untouched. The toggle guard in `model.py` is unchanged, so a branch-enabled
model still raises on `events=None` (a silently branch-less A8 run cannot
complete).
"""

import atexit
import datetime
import json
import os
import threading

import torch

STATE = {
    'installed': False,
    'armed': False,
    'calls': 0,
    'hooked_brains': 0,
    'max_abs_phi_before_zero': None,
    'first_call_ts': None,
    'last_call_ts': None,
    'total_zeroed_elements': 0,
    'log_every': 512,
    'mode': None,
    'pid': os.getpid(),
    'proof_path': None,
}
_LOCK = threading.Lock()
_PATCHED = False


def _ts():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def _hook(module, inputs, output):
    if not torch.is_tensor(output):
        raise TypeError(f'A8 phi-zero hook: encoder output is {type(output)}, not a Tensor')
    zeroed = torch.zeros_like(output)
    if not bool(torch.all(zeroed == 0)):
        raise AssertionError('A8 phi-zero hook: the replacement tensor is not all zero')
    mx = float(output.detach().abs().max()) if output.numel() else 0.0
    with _LOCK:
        STATE['calls'] += 1
        STATE['total_zeroed_elements'] += int(output.numel())
        if STATE['max_abs_phi_before_zero'] is None or mx > STATE['max_abs_phi_before_zero']:
            STATE['max_abs_phi_before_zero'] = mx
        first = STATE['first_call_ts'] is None
        if first:
            STATE['first_call_ts'] = _ts()
        STATE['last_call_ts'] = _ts()
        n = STATE['calls']
        every = STATE['log_every']
    if first:
        if mx <= 0:
            raise AssertionError(
                'A8 phi-zero hook: the first encoder output was identically zero; '
                'zeroing it would be vacuous'
            )
        print(
            f'A8_PHI_ZERO_ASSERT first_call: zeroed_shape={tuple(output.shape)} '
            f'max|phi_before_zero|={mx:.6g} -> returned tensor all-zero',
            flush=True,
        )
    elif every and n % every == 0:
        print(
            f'A8_PHI_ZERO_ASSERT calls={n} hooked_brains={STATE["hooked_brains"]} '
            f'max|phi_before_zero|={STATE["max_abs_phi_before_zero"]:.6g} '
            f'zeroed_elements={STATE["total_zeroed_elements"]}',
            flush=True,
        )
        dump_proof()
    return zeroed


def _patched_init(self, *args, **kwargs):
    _ORIG_INIT(self, *args, **kwargs)
    if getattr(self, 'event_branch', None) is not None:
        self.encoder.register_forward_hook(_hook)
        with _LOCK:
            STATE['hooked_brains'] += 1
            STATE['armed'] = True


_ORIG_INIT = None


def install(*, proof_path, mode, log_every=512):
    """Patch `model.Brain.__init__` so every branch-enabled Brain gets the hook.

    Must be called BEFORE the vendor modules that construct Brains are
    imported/executed (`train.py`, `one_vs_three.py`).
    """
    global _ORIG_INIT, _PATCHED
    import model

    STATE['mode'] = mode
    STATE['log_every'] = log_every
    STATE['proof_path'] = os.path.abspath(proof_path)
    if _PATCHED:
        return STATE
    _ORIG_INIT = model.Brain.__init__
    model.Brain.__init__ = _patched_init
    model.Brain._a8_phi_zero_installed = True
    _PATCHED = True
    STATE['installed'] = True
    atexit.register(dump_proof)
    return STATE


def dump_proof():
    path = STATE['proof_path']
    if not path:
        return
    payload = {
        'arm': 'A8',
        'mode': STATE['mode'],
        'pid': STATE['pid'],
        'installer': 'scripts/exp_a8_phi_zero.py',
        'mechanism': 'Brain.encoder forward hook returns torch.zeros_like(enc_out); '
                     'Brain.forward reads it as phi immediately before phi + W_zero(h_event)',
        'installed': STATE['installed'],
        'hooked_brains': STATE['hooked_brains'],
        'calls': STATE['calls'],
        'assert_all_zero_after_zeroing': True,
        'max_abs_phi_before_zero': STATE['max_abs_phi_before_zero'],
        'total_zeroed_elements': STATE['total_zeroed_elements'],
        'first_call_ts': STATE['first_call_ts'],
        'last_call_ts': STATE['last_call_ts'],
        'armed': bool(STATE['armed'] and STATE['calls'] > 0),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return payload


def assert_armed():
    if not (STATE['armed'] and STATE['calls'] > 0):
        raise RuntimeError(
            f'A8 phi-zero hook not armed: hooked_brains={STATE["hooked_brains"]} '
            f'calls={STATE["calls"]} -- refusing to accept a run in which the '
            'branch-only A8 could have silently evaluated as E3a'
        )
    return True


def summary():
    payload = dump_proof()
    return payload
