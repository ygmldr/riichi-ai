# EX1 Pre-Registration — explainability-metrics validity audit + no-reveal heads

Plan: `.omo/plans/2026-09-23-explainability-metrics-and-validity.md` (todos t1–t20, gates F1–F4).
This file is the **frozen contract** for EX1: every number below is a pre-committed line, not a
measurement. It was written before any EX1 number was computed.

Repo HEAD at pre-registration: `1a3368e36e0b52e75cfdbeeb19aa9c3c40888d2f`

`git status --porcelain` at write time (captured before this todo's two new files existed; they
will appear as untracked `?? scripts/exp_ex1_*` afterwards):

```
 M .omo/notepads/fork-roadmap-steps-1-10/decisions.md
 M .omo/notepads/fork-roadmap-steps-1-10/learnings.md
 M RUNBOOK.md
 M scripts/parse_eval.py
 M vendor/README.md
 M vendor/mortal
?? .zcode/
?? AGENTS.md
?? configs/AGENTS.md
?? configs/config.toml.bak-12
?? configs/config.toml.bak-19
?? configs/config.toml.bak-3
?? configs/config.toml.bak-5
?? configs/config.toml.bak-6
?? configs/config.toml.bak-7
?? configs/config.toml.rl-bak
?? configs/config.toml.sp-bak
?? docs/AGENTS.md
?? "docs/\347\253\213\347\233\264\351\272\273\345\260\206 AI \344\270\213\344\270\200\344\273\243\350\256\255\347\273\203\350\267\257\347\272\277\346\216\242\347\264\242.md"
?? "docs/\347\253\213\347\233\264\351\272\273\345\260\206 AI \344\270\213\344\270\200\344\273\243\350\256\255\347\273\203\350\267\257\347\272\277\346\216\242\347\264\242.md:Zone.Identifier"
?? "docs/\351\272\273\345\260\206 AI \347\240\224\347\251\266\350\267\257\347\272\277\344\272\244\345\217\211\350\257\204\345\256\241 Prompt.md"
?? "docs/\351\272\273\345\260\206 AI \347\240\224\347\251\266\350\267\257\347\272\277\344\272\244\345\217\211\350\257\204\345\256\241 Prompt.md:Zone.Identifier"
?? mjai-reviewer/
?? patches/0002-train-remove-per-step-cuda-sync.patch
?? patches/AGENTS.md
?? scripts/AGENTS.md
?? vendor/AGENTS.md
?? vendor/akochan/
```

Immutability: once committed, this file must not change. Every consuming script records its
sha256 at start and re-reads/re-hashes it at end; drift ⇒ `FATAL` (M8). Any edit requires a new
commit, a new sha256, and invalidates every artifact produced under the old hash.

---

## 1. Goal, objects, hard invariants

**Goal.** (a) Quantify the validity/calibration of the five metric payloads the M1+M2 commentary
engine already ships, each with a point estimate + file-cluster bootstrap CI + a negative control;
(b) add the missing no-reveal explainability metrics, each with its own validity statistic; (c)
produce the calibration product that makes `is_blunder` / `confidence` functional.

**Hard invariants (non-negotiable):**

1. **NO-REVEAL.** No engine metric may consume an opponent's concealed hand, real shanten/tenpai/
   waits, or the true tile order. Oracle replay is a **label source inside the validity harness
   only** (`scripts/exp_ex1_labels.py` must never be imported by `scripts/jiangpu/**`).
2. **No composite scalar.** Each metric is reported separately; no synthesized "validity score".
3. **Calibration ≠ discrimination.** ECE and AUC/PR-AUC are reported side by side; a low ECE from a
   constant predictor is never read as skill.
4. **Pre-registration.** Every acceptance line below is frozen. "Not met" ⇒ negative result
   recorded; lines are never relaxed after seeing numbers.
5. **No A2b result is copied.** EX1 numbers are recomputed from raw; the two-tier *convention*
   (triage + delivery) is adapted from `runs/exp/A2b/PREREG.md` §1, but every threshold below is
   EX1's own.

---

## 2. Data: corpus, splits, seeds, budgets

- **Holdout corpus** = `datasets/extracted/{2024,2025,2026}` (human games, outside the policy
  training years). Deterministic file-level reservoir sample, **seed 13** (t3).
- **Three disjoint subsets** (target allocation; sampler must hit these exactly when the pool
  allows, and record actual counts + year distribution + sha256 in `runs/exp/EX1/corpus_index.json`):

  | subset | role | target files | hard floor |
  |---|---|---|---|
  | `calib_sub` | **only** subset on which any threshold/parameter is fitted | 400 | 200 |
  | `select_sub` | head-to-head selection only (t16) | 400 | 200 |
  | `report_sub` | final reporting; evaluated **exactly once** per head route (t16) | 800 | 400 |

  Pairwise intersections must be empty. `split` ∈ {a,b,c,d} rotates to balance seat coverage.
  Shortfall rule (§9): if the pool cannot fill the targets, take all available and record the
  shortfall — never substitute training-year files.
- **Head-training manifest** (t2): fixed-seed (13) reservoir sample from
  `datasets/extracted/{2009..2022}`; target **20,000 games**, hard bounds **[10,000, 40,000]**,
  recorded in `runs/exp/EX1/train_manifest.json`. Both head routes (t14/t15) train on the **same**
  manifest. No seed pool is consumed; no self-play.
- **Threshold fitting subset** is pinned to `calib_sub` (MA-3/B1). Any number computed from
  `report_sub` is read-only and may never feed a fit or a selection (C-4).
- **Determinism seeds:** corpus/manifest sampling `13`; bootstrap `7`; shuffle negative control
  `0`; head training seed `13` (both routes).

---

## 3. Truth definitions (labels)

All labels come from replaying human records (t2). Tile index `t ∈ 0..33`, aka collapsed via
`tiles.deaka` (34 classes). Opponent index `j ∈ {0,1,2}` = **relative seat order**
`(player_id + j + 1) % 4` ⇒ shimocha / toimen / kamicha (`libriichi/gameplay.rs:279`,
`invisible.rs:152-209`).

- `shanten_j`, `waits_j`, `furiten_j`: per-decision, per-opponent oracle decode (A2 layout
  constants, `scripts/exp_a2_probe.py:143-203`), with the PREREG decode-verification gates
  (critical violation count must be 0).
- **`opp_tenpai_prob` truth:** `1{ shanten_j(d) == 0 }`.
- **`wait_membership` truth (dense):** `1{ shanten_j(d) == 0 AND t ∈ waits_j(d) }`.
  Furiten is **not** modelled (not publicly knowable); this is a documented limitation, and the
  payload must never be presented as a deal-in probability.
- **`deal_in` truth (sparse realized):** `1{ the tile discarded by us at d was ron'd by opponent j }`,
  from `hora{actor=j, target=us, actor≠target}` paired with the immediately preceding
  `dahai.pai` (or `kakan.pai` for chankan). Multi-ron counts once per `hora` event. Non-discarded
  tiles are `0.0` (unobserved ⇒ 0; sparse by construction).
- **`exp_loss` truth (continuous):** `exp_loss[j][t] = Σ (our point delta)` over all `hora` events
  with `actor=j, target=us` whose paired discard tile equals `t`; `0.0` when none. Point delta is
  read from `hora.deltas`; **sign convention: `exp_loss = -deltas[us]` (positive = our loss)**.
  Missing `hora.deltas` ⇒ fail loud (`analyze_alllast._deltas_field` behaviour) — never a silent 0
  (MINOR-1). Labels are recorded verbatim, **never clamped**; the nominal maximum is 48000
  (dealer yakuman ron), and honba bonuses (300/honba) can in principle push a realized label
  slightly above it. The `[0, 48000]` bound in §4 is the **payload** validation range, not a label
  clamp; head outputs must be range-bounded (e.g. `48000·sigmoid`) so contract validation passes.
- **`bad` outcome proxy (used by V2/V5/V6/F3/D1):** primary `bad = 1[the tile we played at d was
  ron'd by any opponent]`; secondary = `1[our seat's point delta over the rest of the kyoku < 0]`.
  If the primary proxy has < 50 positives on the evaluation subset, the secondary proxy is used and
  the switch is recorded.

---

## 4. Frozen metric contract — exact shape, `unit`, per-field range

Machine-readable (the audit harness parses this block and asserts equality with its in-code
expectation; `scripts/exp_ex1_audit.py --selftest` fails if it drifts):

```json ex1-metric-contract
{
  "opp_tenpai": {
    "unit": "probability",
    "fields": {
      "opp_tenpai_prob": {"shape": [3], "unit": "probability", "range": [0.0, 1.0]}
    }
  },
  "danger": {
    "unit": "mixed",
    "unit_by_field": {
      "deal_in": "probability",
      "wait_membership": "probability",
      "exp_loss": "pt_loss"
    },
    "fields": {
      "deal_in": {"shape": [3, 34], "unit": "probability", "range": [0.0, 1.0]},
      "wait_membership": {"shape": [3, 34], "unit": "probability", "range": [0.0, 1.0]},
      "exp_loss": {"shape": [3, 34], "unit": "pt_loss", "range": [0.0, 48000.0]}
    }
  }
}
```

Semantics and deviations from the spec (`docs/superpowers/specs/2026-09-22-commentary-analysis-engine-design.md`):

- `opp_tenpai_prob` is a `[3]` list: one probability per opponent `j` (self excluded), each in
  `[0.0, 1.0]`, `unit = "probability"`. **Spec deviation** (`spec:120` writes `opp_tenpai_prob[4]`,
  player-indexed incl. self): the EX1 payload is per-opponent; t20 records this deviation.
- `danger` is a **mixed-unit** payload: top-level `unit` is pinned to **`"mixed"`** (never
  `"probability"`, which would mislabel `exp_loss`), with `unit_by_field` required (mirrors
  `sp_table`). Fields: `deal_in[j][34]` and `wait_membership[j][34]` in `[0.0, 1.0]`
  (`probability`); `exp_loss[j][34]` continuous point loss, range `[0.0, 48000.0]`, unit `pt_loss`
  — **explicitly not `[0,1]`**. **Spec deviation** (`spec:121` lists only `deal_in`+`exp_loss`):
  `wait_membership` is carried as a third field and is never called a deal-in probability; t20
  records this deviation.
- `dangerous_tiles` lives at **document top level** (spec `:123`), not inside
  `decisions[*].metrics`; top-level extra keys are outside provider validation.
- All provider payloads carry `provider`, `provider_version`, `checkpoint_sha256`, `unit`; plus
  the t4-mandated `no_hidden_info: true` and `basis ∈ {self_hand, public_info_predictor}`
  (`basis == "hindsight_exact"` is always invalid).

---

## 5. Binning, statistics, negative controls

- **ECE** = 15 equal-width bins of `[0,1]` (`np.linspace(0,1,16)`, `np.digitize`), empty bins
  skipped: `Σ_b (n_b/N)·|acc_b − conf_b|`. Shared binary routine; top-1 / pooled / marginal
  variants reduce to it. Used for every `[0,1]` probability (V1, V3, V4, H1).
- **Quantile (equal-count) bins** for low-prevalence tile targets (`deal_in`, `wait_membership`,
  H2/H3) — equal-width ECE is meaningless at ~0.05–2 % prevalence (MA-2).
- **Value-binned table** for continuous targets (`exp_loss` H4, `sp_table.own_ev` V3): 15
  equal-count bins; per bin n / mean predicted / mean realized / bias; scalars MAE, Spearman ρ,
  `value_binned_calibration_error = Σ_b (n_b/N)·|mean_pred_b − mean_real_b|`, also normalized by
  the realized std. **This is not ECE and is not a probability calibration claim.**
- **Primary discrimination statistics:** AUC (binary), PR-AUC / average precision (sparse), Brier
  skill score `BSS = 1 − Brier/Brier_base` with `Brier_base = p̄(1−p̄)` and `p̄` = prevalence on the
  evaluation subset (climatology baseline), Spearman ρ (continuous), MAE-skill
  `1 − MAE/MAE_base` with `MAE_base` = MAE of the constant-mean predictor.
- **CIs:** 95 % **file-cluster bootstrap** (resample source files with replacement), `B = 2000`,
  seed 7. Sparse targets with < 50 positives on the evaluation subset ⇒ `not_estimable`.
- **Negative controls (anti-false-success):** label-shuffle control must satisfy
  `|AUC − 0.5| < 0.02` and `|BSS| < 0.01`; constant-input control must behave as a constant.
  A failing control **stops** the audit (§9). Every headline number is recomputed from raw data
  inside the EX1 scripts — never copied from another experiment.
- All point estimates are accompanied by their CI; "accuracy" is never claimed without both.

---

## 6. Per-metric acceptance lines (triage + delivery) — concrete numbers

**Convention.** *Triage line* = the minimum validity for the metric to ship/enable at all; below
it the metric is recorded as a negative result and the provider stays `unavailable`. *Delivery
line* = the quality bar the shipped artifact must clear, evaluated **once** on `report_sub` for the
winning head route; failing it (but clearing triage) ships with an explicit `low_confidence` /
degraded flag and a documented shortfall. Neither line is relaxed post hoc.

### 6.1 WS-V — audited existing payloads (read-only evaluation; `report_sub` allowed)

| id | metric | primary statistic | triage | delivery |
|---|---|---|---|---|
| V1 | `q_ev` leg ① — action consistency, `softmax(Q/τ=1)` vs recorded action | ECE(15) + Brier | ECE ≤ 0.10 | ECE ≤ 0.05 |
| V2 | `q_ev` leg ② — outcome, risk `= −logged_action_q_rel` vs `bad` | AUC + Spearman vs secondary proxy | AUC ≥ 0.55 | AUC ≥ 0.60 |
| V3 | `sp_table` — `own_tenpai_prob` / `own_win_prob` vs realized; `own_ev` value table | ECE(15); value-binned CE (× realized std) | ECE ≤ 0.10 | ECE ≤ 0.05; value-CE ≤ 0.25×std |
| V4 | `rank_dist` — 16-cell marginal + top-1 vs game-final rank | ECE(15) | marginal ≤ 0.10; top-1 ≤ 0.15 | marginal ≤ 0.05; top-1 ≤ 0.10 |
| V5 | `regret` / `delta_rel` — risk `= −delta_rel` vs `bad` | AUC + Spearman | AUC ≥ 0.55 | AUC ≥ 0.60 |
| V6 | aggregate `judgement_vs_outcome` — `judgement_score` vs `bad` (secondary proxy) | AUC; dispersion gate: std ≥ 0.05 **and** ≥ 10 distinct values, else `not_estimable` | AUC ≥ 0.55 | AUC ≥ 0.60 |

Additional mandatory V5 outputs: monotonicity check (bad-outcome rate in the most-negative
`delta_rel` quintile ≥ that in the least-negative quintile, point estimate). V2/V5 must also report
the `leg3_status: "not_estimable"` line with a non-empty reason ("no demonstrably stronger
reference exists").

### 6.2 WS-E — free no-reveal metrics

| id | metric | primary statistic | triage | delivery |
|---|---|---|---|---|
| F1 | `own_state` (own shanten + ukeire) | bit-exactness vs frozen obs fixture; coverage of readable decisions | 0 fixture mismatches; coverage ≥ 0.90 | 0 mismatches on `report_sub` sample; coverage ≥ 0.95; every `unavailable` has a non-empty reason |
| F2 | `context` (turn/round/scores/visible counts) | bit-exactness vs obs slices | 0 mismatches | 0 mismatches on all sampled `report_sub` decisions |
| F3 | `safety` heuristic score | AUC(risk `= −score`, `bad`) | AUC ≥ 0.55 | AUC ≥ 0.60 |

`heuristic_score` is **never** a probability: `unit == "heuristic_score"`, no key may contain
`prob`/`deal_in`/`danger`.

### 6.3 WS-E — head targets (t14/t15/t16); all lines on the same manifest

| id | target | primary statistic | triage (on `select_sub`) | delivery (on `report_sub`, winner, exactly once) |
|---|---|---|---|---|
| H1 | `opp_tenpai_prob` (binary) | BSS vs base-rate; AUC; ECE(15) | BSS ≥ 0.05 AND AUC ≥ 0.70 | BSS ≥ 0.10 AND AUC ≥ 0.75 AND ECE ≤ 0.05 |
| H2 | `deal_in[j][34]` (sparse-tile) | PR-AUC; BSS; quantile-bin ECE | PR-AUC ≥ 0.005 AND ≥ 3×prevalence AND BSS ≥ 0.005 | PR-AUC ≥ 0.01 AND ≥ 5×prevalence AND BSS ≥ 0.01 AND quantile-ECE ≤ 0.05 |
| H3 | `wait_membership[j][34]` (dense-tile) | PR-AUC; BSS; quantile-bin ECE | PR-AUC ≥ 0.03 AND ≥ 2×prevalence AND BSS ≥ 0.02 | PR-AUC ≥ 0.05 AND ≥ 3×prevalence AND BSS ≥ 0.05 AND quantile-ECE ≤ 0.05 |
| H4 | `exp_loss[j][34]` (continuous) | MAE-skill; Spearman ρ; value-binned CE (× realized std) | MAE-skill ≥ 0.02 AND ρ ≥ 0.05 | MAE-skill ≥ 0.05 AND ρ ≥ 0.10 AND value-CE ≤ 0.25×std |

`exp_loss` must never be silently dropped, and `wait_membership` must never be presented as a
deal-in probability (M8). All four families are reported separately with CIs.

### 6.4 Derived metrics (t17/t18)

| id | metric | primary statistic | triage | delivery |
|---|---|---|---|---|
| D1 | `dangerous_tiles` (top-level) | hit@5: fraction of realized deal-in events whose tile is in the top-5 list; random baseline = 5/34 ≈ 0.147 | hit@5 ≥ 0.22 | hit@5 ≥ 0.29 |
| D2 | `called_risk` (our discard called) — sparse | PR-AUC; BSS | PR-AUC ≥ 0.005 AND ≥ 3×prevalence AND BSS ≥ 0.005 | PR-AUC ≥ 0.01 AND ≥ 5×prevalence AND BSS ≥ 0.01 |
| D3 | `post_call_risk` (dealt in after our call) — sparse | PR-AUC; BSS | same as D2 | same as D2 |

`called_risk` must not be derived from `danger` (B2); the two metrics must not share a field name.

### 6.5 Selection pass line (t16, `select_sub` only)

A head route is **eligible** iff `S = BSS(opp_tenpai_prob) ≥ 0.05` AND `AUC ≥ 0.70` AND every
`danger` target (H2/H3/H4) clears its triage line. Winner = highest `S`; if `|ΔS| < 0.005`,
higher AUC; if `|ΔAUC| < 0.005`, **Route A (public features)** wins the tie (cheaper, no encoder
coupling) and the tie is recorded. Neither eligible ⇒ `winner: null`, both head-based providers
stay `unavailable` with a non-empty reason (fail-closed). The `report_sub` delivery evaluation is
called exactly once; the count is produced by a module-level counter `_REPORT_SUB_EVAL_CALLS` and
recorded in `headtohead.json` (never asserted by hand).

---

## 7. Threshold fitting protocol (calib_sub only)

The calibration product is `runs/exp/EX1/calibration_product.json`; schema is **flat top-level
floats** `gap_threshold` and `regret_blunder_threshold` (nested objects are a runtime `TypeError`
in the consumers, MA-1), plus a sibling `derivation` block
(`fit_subset == "calib_sub"`, `n_decisions`, `rule`, `status`, `ci`).

- `gap_threshold` — semantics fixed by `providers/q_ev.py:37-38`: `confidence = "low"` iff
  `abs(logged_action_q_rel) < gap_threshold`. Candidate grid = deciles 10..90 of
  `|logged_action_q_rel|` on `calib_sub`. Rule: smallest candidate `g` such that decisions with
  `|gap| < g` have outcome-AUC (V2 definition) inside `[0.45, 0.55]` **and** decisions with
  `|gap| ≥ g` have outcome-AUC ≥ 0.60. Fallback: median `|gap|` with
  `status = "fallback_median_no_separation"`.
- `regret_blunder_threshold` — semantics fixed by `providers/regret.py:38`:
  `is_blunder = delta_rel ≤ threshold` (threshold ≤ 0). Candidate grid = percentiles 1..20 of
  `delta_rel` on `calib_sub`. Rule: least-negative candidate `t` such that
  `P(bad | delta_rel ≤ t) ≥ 2 × P(bad)` on `calib_sub`. If the primary `bad` proxy has < 50
  positives on `calib_sub`, the secondary proxy is used and the switch recorded. Fallback: 1st
  percentile with `status = "fallback_p1_no_enrichment"`.
- Thresholds are data-derived; hardcoding is forbidden (D7). Both statuses must be recorded.

---

## 8. NO-REVEAL mechanical guard

`scripts/exp_ex1_labels.py` is the only label source and must have **0 imports** from
`scripts/jiangpu/**` into it (asserted by an import-graph test, t4). New providers must be
demonstrably runnable without oracle tensors. No `hindsight_*` basis may appear in any engine
payload.

---

## 9. Degradation rules

1. **Fail-closed.** Any metric that cannot be computed ⇒ `unavailable: true` + non-empty `reason`;
   never zero-filled payloads, never silent omission (including `dangerous_tiles`, which then
   appears in `unavailable[]`).
2. Below triage ⇒ negative result recorded; provider stays/becomes `unavailable`; the line is not
   relaxed.
3. Between triage and delivery ⇒ ships with an explicit `low_confidence`/degraded flag and a
   documented shortfall.
4. < 50 positives on the evaluation subset (sparse targets) ⇒ `not_estimable`; empty subset ⇒
   `not_estimable` (not a crash); V6 dispersion gate failure ⇒ `not_estimable`.
5. Missing `hora.deltas` ⇒ fail loud; `exp_loss` labels are never fabricated or silently zeroed.
6. Corpus shortfall vs §2 targets ⇒ record actual counts and the shortfall in
   `corpus_index.json`; never substitute training-year files.
7. Any negative control failing (§5) ⇒ **STOP**; the pipeline is broken; no numbers are reported
   until it is fixed.
8. `report_sub` is read-only for t5–t8; no number computed from it may feed any fit or selection
   (C-4). The single-consumption rule applies to the head-to-head delivery evaluation (t16).

---

## 10. Selftest (PREREG-pinned; the audit script refuses to run if it fails)

Synthetic checks executed by `scripts/exp_ex1_audit.py --selftest` before any real audit, reusing
the shared A2b routines (single implementation — never reimplemented):

1. **Perfect calibration:** `p = k/1000` for `k = 1..999`, 1000 samples per value, exactly `k`
   positives per value ⇒ `perfect_ece < 1e-6` (hard gate).
2. **Known miscalibration:** `p ≡ 0.7` with 50 % positives ⇒ `|known_miscal_ece − 0.2| < 1e-12`.
3. **Top-1 path:** 24-class construction with exactly `round(v·n)` hits per confidence level
   ⇒ `top1_perfect_ece < 1e-6`.
4. **Brier sanity:** `brier_score([0,1], [0,1]) == 0.0` ⇒ `brier_sanity == 0.0`.
5. **Cluster path:** `bin_sums` + `ece_from_sums` on one group reproduce the point ECE
   (`< 1e-12`).
6. **Frozen contract:** the §4 JSON block parses and equals the harness's in-code expectation
   (shape / `unit` / range of `opp_tenpai_prob`, `deal_in`, `wait_membership`, `exp_loss`;
   `danger.unit == "mixed"` + `unit_by_field`).

The audit script must exit non-zero with `FATAL` on stderr if the PREREG is missing, if the frozen
contract block drifts, or if the PREREG sha256 changes between the start and the end of a run.

---

## 11. Artifacts and hash discipline

- t1 artifacts: `runs/exp/EX1/prereg_manifest.json` (PREREG path + start/end sha256 + parsed frozen
  contract + selftest result + run-time HEAD). Later todos write their own
  `runs/exp/EX1/*_validity.json`, `calibration_product.json`, `headtohead.json`, `headA_metrics.json`,
  `headB_metrics.json`, `corpus_index.json`, `train_manifest.json`, and `F1.json`–`F4.json`.
- `runs/` is gitignored: artifacts live there, the PREREG and the audit scripts live in
  `scripts/` and are committed.
- Start-of-run: verify the PREREG exists, record sha256. End-of-run: re-read and re-hash; drift ⇒
  `FATAL` (M8 strengthening of the A2b pattern, which recorded but did not re-check).
