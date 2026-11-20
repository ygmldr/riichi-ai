#!/usr/bin/env python3
"""Tests for `scripts/rf_bench.py` (RiichiFormer V2 PR-01).

Run:
  .venv/bin/python -m pytest -q tests/rf_test_bench.py
  .venv/bin/python tests/rf_test_bench.py

CPU tests use a 16-dim, 8-token toy block and complete in a few seconds. The GPU
test is skipped when CUDA is unavailable and uses a 2x32x64 shape so it coexists
with whatever else is using the 5070 Ti.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rf_bench as rb  # noqa: E402


def tiny_cfg(**overrides):
    cfg = {
        "schema": "rf-bench/1",
        "device": "cpu",
        "dtype": "float32",
        "seed": 7,
        "warmup_steps": 1,
        "measure_steps": 2,
        "model": {
            "n_layers": 1,
            "n_heads": 2,
            "ffn_multiplier": 2,
            "dropout": 0.0,
            "causal": True,
        },
        "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0},
        "shapes": [{"batch": 2, "seq": 8, "d_model": 16}],
        "phases": ["forward", "train", "step"],
        "disk": {"scan_paths": [], "zip_suffixes": [".zip"]},
    }
    cfg.update(overrides)
    return cfg


class BenchUnitTests(unittest.TestCase):
    def test_validate_accepts_tiny_config(self):
        rb.validate_config(tiny_cfg())

    def test_validate_rejects_bad_head_divisibility(self):
        cfg = tiny_cfg(shapes=[{"batch": 2, "seq": 8, "d_model": 15}])
        with self.assertRaises(rb.ConfigError):
            rb.validate_config(cfg)

    def test_validate_rejects_missing_train_phase(self):
        with self.assertRaises(rb.ConfigError):
            rb.validate_config(tiny_cfg(phases=["forward"]))

    def test_validate_rejects_unknown_dtype_and_phase(self):
        with self.assertRaises(rb.ConfigError):
            rb.validate_config(tiny_cfg(dtype="float8"))
        with self.assertRaises(rb.ConfigError):
            rb.validate_config(tiny_cfg(phases=["forward", "train", "backward"]))

    def test_validate_rejects_empty_shapes(self):
        with self.assertRaises(rb.ConfigError):
            rb.validate_config(tiny_cfg(shapes=[]))

    def test_percentile(self):
        self.assertEqual(rb._percentile([3.0, 1.0, 2.0], 0.50), 2.0)
        self.assertEqual(rb._percentile([3.0, 1.0, 2.0], 0.95), 3.0)
        self.assertIsNone(rb._percentile([], 0.5))

    def test_cpu_all_phases_are_measured(self):
        cfg = tiny_cfg()
        rec = rb.benchmark_one_shape(cfg["shapes"][0], cfg, torch.device("cpu"))
        self.assertEqual(rec["status"], "PASS", rec.get("error"))
        self.assertGreater(rec["param_count"], 0)
        for phase in ("forward", "train", "step"):
            info = rec["phases"][phase]
            self.assertEqual(info["status"], "PASS")
            self.assertGreater(info["ms_per_step"], 0)
            self.assertGreater(info["tokens_per_sec"], 0)
            self.assertLessEqual(info["p50_ms"], info["p95_ms"])
            self.assertIsNone(info["peak_allocated_mib"])  # CPU: not applicable
            self.assertEqual(info["autocast_dtype"], "float32")

    def test_oom_is_recorded_not_zero_and_next_shape_runs(self):
        cfg = tiny_cfg(shapes=[
            {"batch": 2, "seq": 8, "d_model": 16},
            {"batch": 2, "seq": 8, "d_model": 16},
        ])
        seen = {"train": 0}

        def fake_runner(**kwargs):
            if kwargs["phase"] == "train":
                seen["train"] += 1
                if seen["train"] == 1:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return {"status": "PASS", "phase": kwargs["phase"], "ms_per_step": 1.5,
                    "tokens_per_sec": 100.0, "peak_allocated_mib": 10.0}

        results = rb.benchmark_shapes(cfg, torch.device("cpu"), phase_runner=fake_runner)
        self.assertEqual(len(results), 2)
        first = results[0]
        self.assertEqual(first["status"], "OOM")
        self.assertEqual(first["failed_phase"], "train")
        self.assertIsNone(first["phases"]["train"]["ms_per_step"])
        self.assertIsNone(first["phases"]["train"]["tokens_per_sec"])
        self.assertIn("out of memory", first["phases"]["train"]["error"].lower())
        self.assertEqual(results[1]["status"], "PASS")
        self.assertEqual(rb.compute_status(results), "PASS")

    def test_status_blocked_when_no_training_shape_passes(self):
        bad = {"status": "OOM", "phases": {"forward": {"status": "PASS"},
                                           "train": {"status": "OOM"}}}
        self.assertEqual(rb.compute_status([bad]), "BLOCKED")
        self.assertEqual(rb.compute_status([]), "BLOCKED")

    def test_load_config_hashes_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.yaml"
            path.write_text(yaml.safe_dump(tiny_cfg()), encoding="utf-8")
            cfg, digest = rb.load_config(path)
            self.assertEqual(cfg["shapes"][0]["d_model"], 16)
            self.assertEqual(digest, rb.hashlib.sha256(path.read_bytes()).hexdigest())

    def test_disk_scan_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "datasets" / "extracted" / "2009").mkdir(parents=True)
            (root / "datasets" / "extracted" / "2009" / "a.mjson").write_bytes(b"x" * 10)
            (root / "datasets" / "2009.zip").write_bytes(b"z" * 20)
            cfg = tiny_cfg(disk={"scan_paths": ["datasets"], "zip_suffixes": [".zip"]})
            disk = rb.collect_disk(root, cfg)
            self.assertEqual(disk["datasets_zip_bytes"], 20)
            self.assertEqual(disk["datasets_zip_count"], 1)
            self.assertEqual(disk["datasets_extracted_bytes"], 10)
            self.assertTrue((root / "datasets" / "2009.zip").exists())
            self.assertEqual(disk["scan_paths"]["datasets"]["files"], 2)

    def test_cli_cpu_smoke_writes_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / "bench.yaml"
            cfg_path.write_text(yaml.safe_dump(tiny_cfg()), encoding="utf-8")
            out = root / "artifacts" / "gpu_baseline.json"
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = rb.main([
                    "--config", str(cfg_path),
                    "--out", str(out),
                    "--repo-root", str(root),
                    "--device", "cpu",
                    "--warmup-steps", "1",
                    "--measure-steps", "2",
                ])
            self.assertEqual(code, 0, stderr.getvalue())
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], rb.SCHEMA_VERSION)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["environment"]["selected_device"], "cpu")
            self.assertEqual(len(report["shapes"]), 1)
            self.assertIn("disk", report)
            self.assertEqual(report["safety"]["downloads"], False)
            self.assertEqual(report["config"]["sha256"],
                             rb.hashlib.sha256(cfg_path.read_bytes()).hexdigest())

    def test_cli_bad_config_exits_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / "bad.yaml"
            cfg_path.write_text("schema: nope\n", encoding="utf-8")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = rb.main(["--config", str(cfg_path), "--out", str(root / "x.json"),
                                "--repo-root", str(root)])
            self.assertEqual(code, 1)
            self.assertIn("config error", err.getvalue())

    def test_cpu_environment_is_json_serializable(self):
        env = rb.collect_environment(torch.device("cpu"), "deadbeef", "bench.yaml", REPO_ROOT)
        json.dumps(env)  # must not raise
        self.assertEqual(env["selected_device"], "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the GPU smoke test")
    def test_gpu_environment_is_json_serializable(self):
        env = rb.collect_environment(torch.device("cuda:0"), "deadbeef", "bench.yaml", REPO_ROOT)
        json.dumps(env)  # _CUuuid must already be str-converted
        self.assertGreater(env["gpu"]["total_memory_mib"], 0)
        self.assertTrue(env["gpu"]["uuid"].startswith("GPU-"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the GPU smoke test")
    def test_gpu_bf16_smoke(self):
        cfg = tiny_cfg(dtype="bfloat16", shapes=[{"batch": 2, "seq": 32, "d_model": 64}],
                       warmup_steps=1, measure_steps=2)
        rec = rb.benchmark_one_shape(cfg["shapes"][0], cfg, torch.device("cuda:0"))
        self.assertEqual(rec["status"], "PASS", rec.get("error"))
        train = rec["phases"]["train"]
        self.assertEqual(train["autocast_dtype"], "bfloat16")
        self.assertGreater(train["peak_allocated_mib"], 0)
        self.assertGreater(train["ms_per_step"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
