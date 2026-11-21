#!/usr/bin/env python3
"""PR-01 V3.3 P1 regression tests: EMA shadow and multi-kan second decision."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.eval import m5d_arena as A  # noqa: E402
from riichiformer.models.policy import MVPConfig, Policy  # noqa: E402
from riichiformer.train.b0 import ModelEMA  # noqa: E402
from riichiformer.train.checkpoint import config_hash, save_checkpoint  # noqa: E402


class EMAReadTests(unittest.TestCase):
    def _checkpoint(self, root: Path, *, with_ema: bool) -> Path:
        torch.manual_seed(7)
        config = MVPConfig(layers=1, hidden=32, heads=4, ffn=64,
                           state_hidden=32, name="ema_test")
        model = Policy(config)
        ema = None
        if with_ema:
            ema = ModelEMA(model, 0.5)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.25)
            ema.update(model)
        path = root / ("with_ema.pt" if with_ema else "raw_only.pt")
        save_checkpoint(
            path, model=model, optimizer=None, counters={"step": 1, "tokens": 0,
                                                          "decisions": 0},
            hashes={"schema_hash": "0" * 64, "action_map_hash": "1" * 64,
                    "normalizer_hash": "2" * 64},
            config={"seed": 1}, seed=1, ema=ema,
            model_config={field: getattr(config, field) for field in config.field_names()})
        return path

    def test_ema_is_loaded_from_shadow_and_differs_from_raw(self):
        root = Path(tempfile.mkdtemp(prefix="rf_ema_"))
        path = self._checkpoint(root, with_ema=True)
        raw = A.RFPolicyBackend.from_checkpoint(path, use_ema=False)
        ema = A.RFPolicyBackend.from_checkpoint(path, use_ema=True)
        self.assertTrue(ema.use_ema)
        raw_state = raw.policy.state_dict()
        ema_state = ema.policy.state_dict()
        self.assertFalse(torch.equal(raw_state["policy_head.linear.weight"],
                                     ema_state["policy_head.linear.weight"]))
        # The backend must have loaded the shadow, not a wrapper dict.
        payload = torch.load(path, weights_only=False, map_location="cpu")
        shadow = payload["ema"]["shadow"]
        self.assertTrue(torch.equal(ema_state["policy_head.linear.weight"],
                                    shadow["policy_head.linear.weight"]))
        self.assertTrue(all(torch.isfinite(value).all()
                            for value in ema_state.values()))

    def test_missing_ema_fails_closed(self):
        root = Path(tempfile.mkdtemp(prefix="rf_ema_missing_"))
        path = self._checkpoint(root, with_ema=False)
        with self.assertRaises(A.BridgeError):
            A.RFPolicyBackend.from_checkpoint(path, use_ema=True)


if __name__ == "__main__":
    unittest.main()
