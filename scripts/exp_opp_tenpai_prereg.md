# OPP-TENPAI standalone shipping PREREG

Status: **APPROVED 2026-11-03** by the project owner ("你做吧"). This is a new, narrower
pre-registration. It does **not** relax any EX1 line; EX1's H2/H3/H4 negative results remain
recorded and EX1 §6.5 still makes the Route-A bundle ineligible.

## 1. Question

Can the frozen Route-A H1 head be shipped as a standalone `opp_tenpai_prob` provider, with a
validity claim measured exactly once on an unconsumed holdout?

## 2. Evidence from EX1 t14 (calib_sub, official run 2026-11-02)

- H1 `opp_tenpai_prob`: AUC 0.93703 (95% CI 0.93383–0.94009), BSS 0.53741 (0.52386–0.55010),
  ECE(15) 0.01213 (0.01043–0.01447), n=190,947 slots, prevalence 0.15205.
- Gate-2 invariants passed; shuffled-label control passed; the model was trained on 20,000 games
  (2009–2022), 3,287,306 decisions, 1,608 steps.
- H2 `deal_in` and H4 `exp_loss` failed triage; H3 passed triage but missed delivery. EX1 §6.5
  therefore makes Route A ineligible **as a bundle**. This prereg asks only about H1.

## 3. Frozen model and features

- Checkpoint: `runs/exp/EX1/headA/headA.pt`
  sha256 `2dcb6d6a9baadd34b8af6a5e059a23c36ebbdd1e83e41c6bb45c290a71ad1aa5`;
  weights sha256 `c9585d2164e33b1766e64abb584b12db25bf0c66f4728458bb306af6be7b426e`.
- Features: t12 `context` + t13 `safety`, 526 dimensions, per-opponent `opp_tenpai` head only.
- No retraining, no fine-tuning, no hyperparameter search, no feature changes. The model is the
  same object evaluated in EX1 t14.

## 4. Evaluation data and one-shot rule

- Subset: `report_sub` from `runs/exp/EX1/corpus_index.json` — 800 files, **130,060 label rows;
  129,890 real decisions** (170 kan-select pseudo-decisions excluded by t2's `real_mask`, the same
  convention as EX1 t14's calib_sub: 63,736 rows → 63,649 real); **389,670 opponent slots**; years
  2024–2026, splits a/b/c/d. Never used for training, threshold fitting, or head selection.
- `report_sub` is evaluated **exactly once** for this prereg. No second look, no model-variant
  comparison on it. If the result fails, record the negative result and require a new holdout for
  another attempt. The evaluation script refuses to overwrite an existing metrics report.
- Labels are oracle-derived (t2) and are used only inside the validity harness; the shipped
  provider consumes no oracle tensor.

**Amendment 1 (2026-11-03, before any `report_sub` scoring).** The draft's "130,060 decisions /
390,180 slots" is corrected above to distinguish label rows from real decisions. The t2 label
files contain 130,060 rows; 129,890 are real decisions, and H1 is scored on the `real_mask` rows
(the same convention as EX1 t14's calib_sub). No metric, pass line, model, split, or one-shot rule
changed; `report_sub` remains unconsumed at the time of this amendment.

## 5. Metrics and pass rule

Primary statistics (must all hold):

- BSS vs the base-rate predictor ≥ 0.10
- AUC ≥ 0.75
- ECE(15) ≤ 0.05

Robustness (must all hold for an unqualified ship):

- 95% file-cluster bootstrap lower bound for BSS ≥ 0.05
- 95% file-cluster bootstrap lower bound for AUC ≥ 0.70

Bootstrap: 2,000 draws, file-cluster, fixed seed; point estimate + 95% CI reported for AUC, BSS,
ECE, and PR-AUC. Report the reliability table and base rate.

Decision:

- Primary + robustness pass: ship `opp_tenpai_prob` with `confidence="calibrated"`.
- Primary pass, robustness fail: ship only with `low_confidence=true` and a written caveat.
- Primary fail: provider stays `unavailable` with a non-empty reason; record the negative result.

## 6. Controls

- Base-rate predictor (constant prevalence) must have AUC 0.5 and BSS 0.
- Trained-shuffle checkpoint `runs/exp/EX1/headA/headA_shuffle.pt`
  (sha256 `12de9bc028f63dc175326ac4893fdc27479709a7d0d9b691bdb17ef98a8cde87`) evaluated on the
  same `report_sub`; it must be near chance (AUC within 0.02 of 0.5, no positive BSS).
- No hidden-information ablation is needed: features are public by construction (t12/t13).

## 7. Provider contract

- New provider `scripts/jiangpu/providers/opp_tenpai.py`, loaded via `--head-checkpoint`
  (default `None` → `unavailable` with a reason, never a hardcoded path).
- Payload: `opp_tenpai_prob[j]` for the three opponents; `unit="probability"`;
  `basis="public_info_predictor"`; `no_hidden_info=true`; `checkpoint_sha256` set to the frozen
  checkpoint.
- Must not emit `deal_in`, `exp_loss`, `wait_membership`, `danger`, or any heuristic-safety field.
- CLI/contract/tests: `test_opp_tenpai.py`; update `test_cli.py` and `tests/test_m2.py` only as
  required by the new provider; keep `meld_decision_eval` in `unavailable[]`.
- Determinism: two runs on the same record must be byte-identical.

## 8. Artifacts

- `runs/exp/OPP_TENPAI/PREREG.md` (copy of this file, with sha256)
- `runs/exp/OPP_TENPAI/metrics_report.json` (one-shot report_sub evaluation; provenance + hashes)
- `runs/exp/OPP_TENPAI/report_sub_features.npy` + `report_sub_targets.npz` + meta
- Provider code + tests + docs; no changes to `runs/1v3/`, `runs/exp/M3/games/`, `datasets/`,
  `vendor/`, or pinned checkpoints.

## 9. Out of scope

- `danger`, `called_risk`, `post_call_risk`, `dangerous_tiles`: remain unavailable; EX1 negatives
  stand.
- No new training, no Route B, no change to EX1 §6.5.
- No LLM/commentary changes; this is W1 only.

## 10. Approval and immutability

Approval covers (a) the frozen model/features above, (b) one-shot `report_sub` consumption, (c) the
pass rule in §5, (d) the provider contract in §7. Any edit requires a new commit, a new sha256, and
a new holdout; the evaluation script records the sha256 at start and re-hashes at end.
