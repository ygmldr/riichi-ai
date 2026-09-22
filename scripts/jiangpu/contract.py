"""单元 5: 输出契约(手写校验,零第三方依赖)。

要点:
  - 版本化(SCHEMA_VERSION)
  - 每个 provider 必须带 {provider, provider_version, checkpoint_sha256, unit}
  - rank_dist 矩阵每行必须和为 1(领域不变量,也是门 2 的一部分)
  - unavailable 元素必须带 reason(不允许静默缺省)
"""
import math

SCHEMA_VERSION = 1
ENGINE_VERSION = 1

TOP_KEYS = ("schema_version", "engine_version", "checkpoint", "record", "decisions",
            "kyoku_summaries", "game_summary", "key_moments", "unavailable")
PROVIDER_KEYS = ("provider", "provider_version", "checkpoint_sha256", "unit")
DECISION_KEYS = ("index", "player_id", "at_kyoku", "at_turn", "shanten",
                 "logged_action", "legal_actions", "metrics")
NO_CHECKPOINT_PROVIDERS = ("sp_table",)


def round_floats(obj, ndigits=6):
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def _check_provider(errors, where, name, payload):
    if not isinstance(payload, dict):
        errors.append(f"{where}: 应为对象")
        return
    for key in PROVIDER_KEYS:
        if key not in payload:
            errors.append(f"{where}: 缺少 {key}")
    if name not in NO_CHECKPOINT_PROVIDERS and payload.get("checkpoint_sha256") is None:
        errors.append(f"{where}: checkpoint_sha256 不得为 null")


def validate(doc):
    errors = []
    if not isinstance(doc, dict):
        return ["顶层应为对象"]
    for key in TOP_KEYS:
        if key not in doc:
            errors.append(f"顶层缺少 {key}")
    if doc.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 应为 {SCHEMA_VERSION},实为 {doc.get('schema_version')!r}")

    ck = doc.get("checkpoint") or {}
    for key in ("path", "sha256", "steps"):
        if key not in ck:
            errors.append(f"checkpoint 缺少 {key}")

    rec = doc.get("record") or {}
    for key in ("path", "source", "names", "seed", "key", "split"):
        if key not in rec:
            errors.append(f"record 缺少 {key}")

    for i, dec in enumerate(doc.get("decisions") or []):
        where = f"decisions[{i}]"
        for key in DECISION_KEYS:
            if key not in dec:
                errors.append(f"{where}: 缺少 {key}")
        metrics = dec.get("metrics") or {}
        if not isinstance(metrics, dict):
            errors.append(f"{where}.metrics 应为对象")
            continue
        for name, payload in metrics.items():
            _check_provider(errors, f"{where}.metrics.{name}", name, payload)
        rd = metrics.get("rank_dist")
        if isinstance(rd, dict) and "matrix" in rd:
            for r, row in enumerate(rd["matrix"]):
                if len(row) != 4:
                    errors.append(f"{where}.rank_dist.matrix[{r}]: 长度应为 4")
                    continue
                total = sum(float(v) for v in row)
                if not math.isclose(total, 1.0, abs_tol=1e-4):
                    errors.append(f"{where}.rank_dist.matrix[{r}]: row 和应为 1,实为 {total:.6f}")

    for i, item in enumerate(doc.get("unavailable") or []):
        if not isinstance(item, dict) or "metric" not in item or "reason" not in item:
            errors.append(f"unavailable[{i}]: 必须同时含 metric 与 reason")
    return errors
