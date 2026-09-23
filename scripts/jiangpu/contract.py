"""单元 5: 输出契约(手写校验,零第三方依赖)。

要点:
  - 版本化(SCHEMA_VERSION);v2 起新增必填键 no_hidden_info / basis(破坏性变更)
  - 每个 provider 必须带 {provider, provider_version, checkpoint_sha256, unit,
    no_hidden_info, basis};NO-REVEAL 铁律:basis 仅允许 self_hand / public_info_predictor,
    "hindsight_exact" 一律判错
  - opp_tenpai / danger 的形状、unit 与逐字段范围**引用** PREREG §4 的冻结字面量
    (scripts/exp_ex1_prereg.md, fenced block `json ex1-metric-contract`),本文件只镜像不重定义;
    tests/test_contract.py 解析 PREREG 并断言两者相等
  - rank_dist 矩阵每行必须和为 1(领域不变量,也是门 2 的一部分)
  - unavailable 元素必须带 reason(不允许静默缺省)
"""
import math

SCHEMA_VERSION = 2
ENGINE_VERSION = 1

TOP_KEYS = ("schema_version", "engine_version", "checkpoint", "record", "decisions",
            "kyoku_summaries", "game_summary", "key_moments", "unavailable")
PROVIDER_KEYS = ("provider", "provider_version", "checkpoint_sha256", "unit",
                 "no_hidden_info", "basis")
DECISION_KEYS = ("index", "player_id", "at_kyoku", "at_turn", "shanten",
                 "logged_action", "legal_actions", "metrics")
# 免费(无检查点)provider 白名单:t11/t12/t13;有检查点的 provider(含 opp_tenpai/danger)
# 不得加入;called_risk 的路线由 t18 决定后自行添加。
NO_CHECKPOINT_PROVIDERS = ("sp_table", "own_state", "context", "safety")

VALID_BASIS = ("self_hand", "public_info_predictor")
FORBIDDEN_BASIS = "hindsight_exact"

# PREREG §4 冻结契约的引擎侧镜像(只引用,不重新定义)。
FROZEN_METRIC_CONTRACT = {
    "opp_tenpai": {
        "unit": "probability",
        "fields": {
            "opp_tenpai_prob": {"shape": [3], "unit": "probability", "range": [0.0, 1.0]},
        },
    },
    "danger": {
        "unit": "mixed",
        "unit_by_field": {
            "deal_in": "probability",
            "wait_membership": "probability",
            "exp_loss": "pt_loss",
        },
        "fields": {
            "deal_in": {"shape": [3, 34], "unit": "probability", "range": [0.0, 1.0]},
            "wait_membership": {"shape": [3, 34], "unit": "probability", "range": [0.0, 1.0]},
            "exp_loss": {"shape": [3, 34], "unit": "pt_loss", "range": [0.0, 48000.0]},
        },
    },
}


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


def _check_range(errors, where, values, lo, hi):
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= float(v) <= hi):
            errors.append(f"{where}[{i}]: 取值应在 [{lo}, {hi}] 内,实为 {v!r}")


def _check_frozen_field(errors, where, value, spec):
    shape = spec["shape"]
    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        errors.append(f"{where}: 形状应为 {shape},实为 {value!r}")
        return
    if len(shape) == 1:
        _check_range(errors, where, value, *spec["range"])
        return
    for i, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != shape[1]:
            errors.append(f"{where}[{i}]: 长度应为 {shape[1]},实为 {row!r}")
            continue
        _check_range(errors, f"{where}[{i}]", row, *spec["range"])


def _check_frozen_contract(errors, where, name, payload):
    """按 PREREG §4 的冻结字面量校验 opp_tenpai / danger;其余 provider 不受约束。"""
    spec = FROZEN_METRIC_CONTRACT.get(name)
    if spec is None or payload.get("unavailable"):
        return
    if payload.get("unit") != spec["unit"]:
        errors.append(f"{where}: unit 应为 {spec['unit']!r},实为 {payload.get('unit')!r}")
    if "unit_by_field" in spec and payload.get("unit_by_field") != spec["unit_by_field"]:
        errors.append(f"{where}: unit_by_field 应为 {spec['unit_by_field']!r},"
                      f"实为 {payload.get('unit_by_field')!r}")
    for field, field_spec in spec["fields"].items():
        _check_frozen_field(errors, f"{where}.{field}", payload.get(field), field_spec)


def _check_provider(errors, where, name, payload):
    if not isinstance(payload, dict):
        errors.append(f"{where}: 应为对象")
        return
    for key in PROVIDER_KEYS:
        if key not in payload:
            errors.append(f"{where}: 缺少 {key}")
    if "no_hidden_info" in payload and payload["no_hidden_info"] is not True:
        errors.append(f"{where}: no_hidden_info 必须为 true(NO-REVEAL)")
    basis = payload.get("basis")
    if basis == FORBIDDEN_BASIS:
        errors.append(f"{where}: basis={FORBIDDEN_BASIS!r} 永远非法(NO-REVEAL)")
    elif basis is not None and basis not in VALID_BASIS:
        errors.append(f"{where}: basis 应为 {VALID_BASIS} 之一,实为 {basis!r}")
    if name not in NO_CHECKPOINT_PROVIDERS and payload.get("checkpoint_sha256") is None:
        errors.append(f"{where}: checkpoint_sha256 不得为 null")
    _check_frozen_contract(errors, where, name, payload)


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
