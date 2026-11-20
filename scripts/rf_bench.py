#!/usr/bin/env python3
"""RiichiFormer V2 PR-01: synthetic SDPA compute/VRAM benchmark (GPU capacity only).

This is deliberately **not** a mahjong model and **not** a performance claim for the
project. It measures a generic causal transformer block under the exact schedule that
PR-13/PR-23 will use (BF16 autocast + SDPA + AdamW), so the 16 GB budget can be
planned with real numbers instead of guesses.

Per shape ``(batch, seq, d_model)`` and per phase:

* ``forward`` — eval, ``torch.no_grad()``, autocast;
* ``train``   — forward + backward (no optimizer step);
* ``step``    — forward + backward + AdamW step (optimizer state allocated in warmup).

For each phase it records wall time, ms/step, tokens/s, p50/p95 latency and
``torch.cuda.max_memory_allocated/reserved``.  OOM is **recorded and continued**:
the failed phase gets ``ms_per_step: null`` (never 0 ms), the error text, and the
peak memory observed so far; the next shape still runs.  If no shape passes the
required ``train`` phase, the artefact status is ``BLOCKED`` and exit code is 2.

Safety: read-only w.r.t. the repo, no downloads, no installs, no background jobs,
no GPU setting changes.  The only write is ``--out`` (plus its parent directory).

Usage:
  .venv/bin/python scripts/rf_bench.py \
      --config configs/riichiformer/benchmark.yaml \
      --out artifacts/riichiformer/bench/gpu_baseline.json
  .venv/bin/python scripts/rf_bench.py --config ... --device cpu --max-shapes 1
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # PyYAML is a dev dependency of the new rf_* CLIs, not of the old Mortal code.
    import yaml
except ImportError:  # pragma: no cover - handled with a clear message in main()
    yaml = None

SCHEMA_VERSION = "rf-bench/1"
DEFAULT_OUT = "artifacts/riichiformer/bench/gpu_baseline.json"
PHASES = ("forward", "train", "step")
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class ConfigError(ValueError):
    """Invalid benchmark config / CLI override (fail closed, never guess)."""


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path: Path):
    if yaml is None:
        raise ConfigError("PyYAML is required: .venv/bin/pip install 'pyyaml==6.0.3'")
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    raw = path.read_bytes()
    try:
        cfg = yaml.safe_load(raw.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    validate_config(cfg)
    return cfg, hashlib.sha256(raw).hexdigest()


def validate_config(cfg: dict) -> None:
    if cfg.get("schema") != SCHEMA_VERSION:
        raise ConfigError(f"schema must be {SCHEMA_VERSION!r}, got {cfg.get('schema')!r}")
    model = cfg.get("model")
    if not isinstance(model, dict):
        raise ConfigError("model must be a mapping")
    for key in ("n_layers", "n_heads", "ffn_multiplier"):
        if not isinstance(model.get(key), int) or model[key] < 1:
            raise ConfigError(f"model.{key} must be a positive integer")
    if not 0.0 <= float(model.get("dropout", 0.0)) < 1.0:
        raise ConfigError("model.dropout must be in [0, 1)")
    if not isinstance(model.get("causal", True), bool):
        raise ConfigError("model.causal must be a boolean")

    if not isinstance(cfg.get("warmup_steps"), int) or cfg["warmup_steps"] < 0:
        raise ConfigError("warmup_steps must be a non-negative integer")
    if not isinstance(cfg.get("measure_steps"), int) or cfg["measure_steps"] < 1:
        raise ConfigError("measure_steps must be a positive integer")
    if cfg.get("dtype") not in DTYPES:
        raise ConfigError(f"dtype must be one of {sorted(DTYPES)}")

    shapes = cfg.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ConfigError("shapes must be a non-empty list")
    for i, spec in enumerate(shapes):
        if not isinstance(spec, dict):
            raise ConfigError(f"shapes[{i}] must be a mapping")
        for key in ("batch", "seq", "d_model"):
            if not isinstance(spec.get(key), int) or spec[key] < 1:
                raise ConfigError(f"shapes[{i}].{key} must be a positive integer")
        if spec["d_model"] % model["n_heads"] != 0:
            raise ConfigError(
                f"shapes[{i}].d_model={spec['d_model']} is not divisible by "
                f"model.n_heads={model['n_heads']}"
            )

    phases = cfg.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ConfigError("phases must be a non-empty list")
    if any(p not in PHASES for p in phases):
        raise ConfigError(f"phases must be a subset of {list(PHASES)}, got {phases}")
    if len(set(phases)) != len(phases):
        raise ConfigError("phases contains duplicates")
    if "train" not in phases:
        raise ConfigError("phases must include 'train' (PR-01 requires a training shape)")

    opt = cfg.get("optimizer")
    if not isinstance(opt, dict) or opt.get("name") != "adamw":
        raise ConfigError("optimizer.name must be 'adamw'")
    if float(opt.get("lr", 0.0)) <= 0:
        raise ConfigError("optimizer.lr must be > 0")

    disk = cfg.get("disk", {})
    if not isinstance(disk, dict):
        raise ConfigError("disk must be a mapping")
    paths = disk.get("scan_paths", [])
    if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
        raise ConfigError("disk.scan_paths must be a list of strings")


def resolve_device(requested: str) -> torch.device:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError("--device cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ConfigError(f"device must be auto|cuda|cpu, got {requested!r}")


# --------------------------------------------------------------------------- #
# synthetic model
# --------------------------------------------------------------------------- #
class SynthBlock(nn.Module):
    """Pre-norm transformer block using ``F.scaled_dot_product_attention``."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float, causal: bool):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, d = x.shape
        y = self.norm1(x)
        qkv = self.qkv(y).view(b, seq, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, is_causal=self.causal
        )
        attn = attn.transpose(1, 2).reshape(b, seq, d)
        x = x + self.out(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class SynthTransformer(nn.Module):
    """Feature-only stack: input hidden states, output hidden states (no vocab head)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, ffn_dim: int,
                 dropout: float, causal: bool):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.blocks = nn.ModuleList(
            SynthBlock(d_model, n_heads, ffn_dim, dropout, causal) for _ in range(n_layers)
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


def build_model(spec: dict, model_cfg: dict, device: torch.device) -> SynthTransformer:
    d = spec["d_model"]
    model = SynthTransformer(
        d_model=d,
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        ffn_dim=d * model_cfg["ffn_multiplier"],
        dropout=float(model_cfg.get("dropout", 0.0)),
        causal=bool(model_cfg.get("causal", True)),
    )
    return model.to(device)


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=DTYPES[dtype_name])


def _allocated_mib(device: torch.device):
    if device.type != "cuda":
        return None
    try:
        return round(torch.cuda.memory_allocated(device) / (1 << 20), 2)
    except Exception:  # pragma: no cover - corrupted context after OOM
        return None


def _peak_mib(device: torch.device):
    if device.type != "cuda":
        return None, None
    try:
        allocated = round(torch.cuda.max_memory_allocated(device) / (1 << 20), 2)
        reserved = round(torch.cuda.max_memory_reserved(device) / (1 << 20), 2)
        return allocated, reserved
    except Exception:  # pragma: no cover
        return None, None


def _percentile(values, q: float):
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1))
    return ordered[idx]


def _one_step(model, optimizer, x, phase, device, dtype):
    """One optimizer-aware step; autocast context is created per call (reentrant)."""
    if phase == "forward":
        with torch.no_grad(), _autocast(device, dtype):
            out = model(x)
            return out.float().square().mean()
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, dtype):
        out = model(x)
        loss = out.float().square().mean()
    loss.backward()
    if phase == "step":
        optimizer.step()
    return loss


def run_phase(*, model, optimizer, x, phase, warmup, measure, device, dtype):
    """Measure one phase; returns a JSON-safe record. Raises only on real errors."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    model.train(phase != "forward")

    for _ in range(warmup):
        _one_step(model, optimizer, x, phase, device, dtype)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_alloc = _allocated_mib(device)

    per_step = []
    _sync(device)
    wall_start = time.perf_counter()
    for _ in range(measure):
        t0 = time.perf_counter()
        _one_step(model, optimizer, x, phase, device, dtype)
        _sync(device)
        per_step.append(time.perf_counter() - t0)
    wall = time.perf_counter() - wall_start
    peak_alloc, peak_reserved = _peak_mib(device)

    tokens = x.shape[0] * x.shape[1] * measure
    return {
        "status": "PASS",
        "phase": phase,
        "steps": measure,
        "wall_seconds": round(wall, 6),
        "ms_per_step": round(1000.0 * wall / measure, 3),
        "tokens_per_sec": round(tokens / wall, 1) if wall > 0 else None,
        "p50_ms": round(1000.0 * _percentile(per_step, 0.50), 3),
        "p95_ms": round(1000.0 * _percentile(per_step, 0.95), 3),
        "start_allocated_mib": start_alloc,
        "peak_allocated_mib": peak_alloc,
        "peak_reserved_mib": peak_reserved,
        "autocast_dtype": dtype if device.type == "cuda" else "float32",
    }


def _failure_record(phase: str, exc: BaseException, device: torch.device):
    message = f"{type(exc).__name__}: {exc}"
    oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
    peak_alloc, peak_reserved = _peak_mib(device)
    return {
        "status": "OOM" if oom else "ERROR",
        "phase": phase,
        "error": message[:600],
        "ms_per_step": None,          # never fake a 0 ms success
        "tokens_per_sec": None,
        "wall_seconds": None,
        "p50_ms": None,
        "p95_ms": None,
        "start_allocated_mib": _allocated_mib(device),
        "peak_allocated_mib": peak_alloc,
        "peak_reserved_mib": peak_reserved,
        "autocast_dtype": None,
    }


def benchmark_one_shape(spec: dict, cfg: dict, device: torch.device, phase_runner=None):
    """Run all phases for one shape; OOM/errors are recorded, never raised."""
    started = time.perf_counter()
    batch, seq, d_model = spec["batch"], spec["seq"], spec["d_model"]
    rec = {
        "batch": batch,
        "seq": seq,
        "d_model": d_model,
        "tokens": batch * seq,
        "status": "PASS",
        "failed_phase": None,
        "param_count": None,
        "elapsed_seconds": None,
        "phases": {},
    }
    runner = phase_runner or run_phase
    model = optimizer = x = None
    try:
        torch.manual_seed(int(cfg.get("seed", 0)))
        model = build_model(spec, cfg["model"], device)
        rec["param_count"] = int(sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["optimizer"]["lr"]),
            weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
        )
        x = torch.randn(batch, seq, d_model, device=device)
        for phase in cfg["phases"]:
            try:
                phase_rec = runner(
                    model=model,
                    optimizer=optimizer,
                    x=x,
                    phase=phase,
                    warmup=int(cfg["warmup_steps"]),
                    measure=int(cfg["measure_steps"]),
                    device=device,
                    dtype=cfg["dtype"],
                )
            except Exception as exc:  # noqa: BLE001 - recorded as evidence
                phase_rec = _failure_record(phase, exc, device)
            rec["phases"][phase] = phase_rec
            if phase_rec.get("status") != "PASS":
                rec["status"] = phase_rec["status"]
                rec["failed_phase"] = phase
                break
    except Exception as exc:  # noqa: BLE001 - model construction / input OOM
        rec["status"] = "ERROR"
        rec["failed_phase"] = "<setup>"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:600]
    finally:
        del model, optimizer, x
        gc.collect()
        if device.type == "cuda":
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
    rec["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return rec


def benchmark_shapes(cfg: dict, device: torch.device, phase_runner=None, progress=None):
    results = []
    for index, spec in enumerate(cfg["shapes"]):
        if progress:
            progress(index, spec)
        results.append(benchmark_one_shape(spec, cfg, device, phase_runner=phase_runner))
    return results


def compute_status(results, required_phase: str = "train") -> str:
    for rec in results:
        phase = (rec.get("phases") or {}).get(required_phase, {})
        if rec.get("status") == "PASS" and phase.get("status") == "PASS":
            return "PASS"
    return "BLOCKED"


# --------------------------------------------------------------------------- #
# environment / disk
# --------------------------------------------------------------------------- #
def _nvidia_smi():
    candidates = [shutil.which("nvidia-smi"), "/usr/lib/wsl/lib/nvidia-smi"]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        proc = subprocess.run(
            [candidate, "--query-gpu=name,memory.total,memory.used,uuid,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        if proc.returncode != 0:
            continue
        rows = []
        for line in proc.stdout.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 6:
                rows.append({
                    "name": cols[0],
                    "memory_total_mib": _int_or_none(cols[1]),
                    "memory_used_mib": _int_or_none(cols[2]),
                    "uuid": cols[3],
                    "driver_version": cols[4],
                    "compute_cap": cols[5],
                })
        if rows:
            return {"available": True, "path": candidate, "gpus": rows}
    return {"available": False, "path": None, "gpus": []}


def _int_or_none(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def collect_environment(device: torch.device, config_sha: str, config_path: str, repo: Path):
    env = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "wsl": "microsoft" in platform.release().lower(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "numpy_available": _module_version("numpy"),
            "pyyaml_version": _module_version("yaml"),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None,
            "allow_bf16_reduced_precision_reduction": bool(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "gpu": None,
        "nvidia_smi": _nvidia_smi(),
        "cpu": {"count": os.cpu_count()},
        "ram_total_gib": _ram_gib(),
        "multiprocessing_start_method": _start_method(),
        "shared_memory": _shm_gib(),
        "config": {"path": config_path, "sha256": config_sha},
        "git_sha": _git_sha(repo),
        "selected_device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        env["gpu"] = {
            "name": props.name,
            "uuid": (
                "GPU-" + str(getattr(props, "uuid", ""))
                if getattr(props, "uuid", None) and not str(getattr(props, "uuid")).startswith("GPU-")
                else (str(getattr(props, "uuid", "")) or None)
            ),
            "total_memory_bytes": int(props.total_memory),
            "total_memory_mib": round(props.total_memory / (1 << 20)),
            "free_memory_mib_before": round(free / (1 << 20)),
            "used_memory_mib_before": round((total - free) / (1 << 20)),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": int(props.multi_processor_count),
        }
    return env


def _module_version(name: str):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return None


def _ram_gib():
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1 << 20), 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _start_method():
    try:
        import torch.multiprocessing as mp

        return mp.get_start_method()
    except Exception:  # noqa: BLE001
        return None


def _shm_gib():
    try:
        usage = shutil.disk_usage("/dev/shm")
        return {"total_gib": round(usage.total / (1 << 30), 1),
                "free_gib": round(usage.free / (1 << 30), 1)}
    except OSError:
        return None


def _git_sha(repo: Path):
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None
    except OSError:
        return None


def path_size(path: Path):
    total = 0
    files = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            try:
                st = os.stat(os.path.join(dirpath, name), follow_symlinks=False)
            except OSError:
                continue
            total += st.st_size
            files += 1
    return total, files


def collect_disk(repo: Path, cfg: dict):
    disk_cfg = cfg.get("disk", {})
    paths = disk_cfg.get("scan_paths", [])
    suffixes = tuple(disk_cfg.get("zip_suffixes", [".zip"]))
    entries = {}
    total = 0
    for rel in paths:
        target = repo / rel
        if not target.exists():
            entries[rel] = {"exists": False, "bytes": None, "files": None}
            continue
        if target.is_file():
            size, files = target.stat().st_size, 1
        else:
            size, files = path_size(target)
        entries[rel] = {"exists": True, "bytes": size, "files": files,
                        "gib": round(size / (1 << 30), 2)}
        total += size
    # classify datasets zips vs extracted (only if datasets is in scan_paths)
    datasets = repo / "datasets"
    zip_bytes, zip_count = 0, 0
    if datasets.is_dir():
        for p in datasets.rglob("*"):
            if p.is_file() and p.suffix.lower() in suffixes:
                try:
                    zip_bytes += p.stat().st_size
                    zip_count += 1
                except OSError:
                    pass
    extracted = datasets / "extracted"
    extracted_bytes = path_size(extracted)[0] if extracted.is_dir() else None
    usage = shutil.disk_usage(repo)
    return {
        "scan_paths": entries,
        "scanned_bytes": total,
        "scanned_gib": round(total / (1 << 30), 2),
        "datasets_zip_bytes": zip_bytes,
        "datasets_zip_count": zip_count,
        "datasets_extracted_bytes": extracted_bytes,
        "free_bytes": usage.free,
        "free_gib": round(usage.free / (1 << 30), 1),
        "total_bytes": usage.total,
        "note": "read-only sizing; no zip was deleted or moved",
    }


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _atomic_json(path: Path, obj: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".rf_bench.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_summary(report: dict) -> None:
    gpu = (report.get("environment") or {}).get("gpu") or {}
    print(f"rf_bench: status={report['status']} device={report['environment']['selected_device']} "
          f"gpu={gpu.get('name')} free_before={gpu.get('free_memory_mib_before')} MiB")
    for rec in report["shapes"]:
        parts = []
        for phase in PHASES:
            info = rec["phases"].get(phase)
            if not info:
                continue
            if info["status"] == "PASS":
                parts.append(f"{phase}={info['ms_per_step']}ms {info['peak_allocated_mib']}MiB")
            else:
                parts.append(f"{phase}={info['status']}")
        print(f"  B{rec['batch']} L{rec['seq']} d{rec['d_model']}: {rec['status']} " + " ".join(parts))
    print(f"  disk free={report['disk']['free_gib']} GiB scanned={report['disk']['scanned_gib']} GiB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RiichiFormer V2 synthetic SDPA benchmark (PR-01)")
    ap.add_argument("--config", default="configs/riichiformer/benchmark.yaml")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--device", default=None, help="auto|cuda|cpu (overrides YAML)")
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--measure-steps", type=int, default=None)
    ap.add_argument("--max-shapes", type=int, default=None)
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    try:
        cfg, config_sha = load_config((repo / args.config).resolve()
                                      if not Path(args.config).is_absolute() else Path(args.config))
        if args.warmup_steps is not None:
            if args.warmup_steps < 0:
                raise ConfigError("--warmup-steps must be >= 0")
            cfg["warmup_steps"] = args.warmup_steps
        if args.measure_steps is not None:
            if args.measure_steps < 1:
                raise ConfigError("--measure-steps must be >= 1")
            cfg["measure_steps"] = args.measure_steps
        if args.max_shapes is not None:
            if args.max_shapes < 1:
                raise ConfigError("--max-shapes must be >= 1")
            cfg["shapes"] = cfg["shapes"][: args.max_shapes]
        validate_config(cfg)
        device = resolve_device(args.device or cfg.get("device", "auto"))
        if device.type == "cuda" and cfg["dtype"] != "bfloat16":
            raise ConfigError("PR-01 requires dtype=bfloat16 for CUDA autocast")
    except ConfigError as exc:
        print(f"rf_bench: config error: {exc}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    environment = collect_environment(device, config_sha, args.config, repo)
    disk = collect_disk(repo, cfg)

    def progress(index, spec):
        print(f"[{index + 1}/{len(cfg['shapes'])}] B{spec['batch']} L{spec['seq']} "
              f"d{spec['d_model']}", file=sys.stderr, flush=True)

    try:
        shapes = benchmark_shapes(cfg, device, progress=progress)
    except Exception as exc:  # noqa: BLE001 - report, never fake PASS
        print(f"rf_bench: internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    status = compute_status(shapes)
    report = {
        "schema": SCHEMA_VERSION,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "required_phase": "train",
        "shapes": shapes,
        "environment": environment,
        "disk": disk,
        "config": {"path": args.config, "sha256": config_sha, "effective": {
            "warmup_steps": cfg["warmup_steps"],
            "measure_steps": cfg["measure_steps"],
            "dtype": cfg["dtype"],
            "phases": cfg["phases"],
            "model": cfg["model"],
            "optimizer": cfg["optimizer"],
        }},
        "duration_seconds": round(time.perf_counter() - started, 2),
        "safety": {
            "downloads": False,
            "installs": False,
            "background_jobs": False,
            "gpu_settings_changed": False,
            "writes": ["--out only"],
        },
    }
    try:
        artifact_sha = _atomic_json(Path(args.out), report)
    except OSError as exc:
        print(f"rf_bench: cannot write {args.out}: {exc}", file=sys.stderr)
        return 1
    print_summary(report)
    print(f"  artifact_sha256={artifact_sha}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
