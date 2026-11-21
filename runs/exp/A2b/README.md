# A2b — Calibration audit (probe / GRP / DQN-Q), zero training

Date: 2026-09-17 · Plan: `.omo/plans/fork-roadmap-steps-1-10.md` todo 14 · Roadmap: `docs/FORK_ROADMAP.md` step 6
Artifacts: `runs/exp/A2b/{PREREG.md, results.json, README.md}` + per-sample npz
(`a2b_probe_val.npz`, `a2b_grp_rounds.npz`, `a2b_q_records.npz`) ·
Script: `scripts/exp_a2b_calibration.py` · Evidence: `.omo/start-work/evidence/T14.txt` ·
PREREG sha256 `92cd0d37bb130efa9f6da4ca8ab5dd291f407dcd4921339a53b10d6fcc2ce5d1`
(written before the run, unchanged after it; the run records the hash).

## TL;DR

**All three pre-registered ECEs are ≤ 0.05 → the pre-registered gate is `block`: this audit
finds no calibration gap that would justify the calibration-aware belief head (todo 29 /
roadmap step 10).** The "model isn't uncalibrated, it just has no signal" hypothesis is *not*
supported: the two genuine probability models are already well calibrated, and the third
(Q-values) is miscalibrated only under a temperature that is a deployment artifact, not a
probability statement.

| # | object | truth | primary metric | value | 95% CI (cluster bootstrap by file) |
|---|---|---|---|---|---|
| (a) | A2 probe `P(opp_any_tenpai)` | oracle-decoded label, held-out val | **ECE(15)** / Brier | **0.0099** / 0.0943 | [0.0086, 0.0180] |
| (b) | GRP 24-permutation softmax | game-final `rank_by_player` | **top-1 ECE(15)** / top-1 Brier | **0.0242** / 0.1467 | [0.0214, 0.0282] |
| (c) | DQN `softmax(Q/τ=1)` over legal actions | action recorded in the log | **ECE(15)** / Brier | **0.0364** / 0.1241 | [0.0346, 0.0382] |

Pre-committed verdicts (`results.json → gate`):
`gate_three_default = block`, `gate_two_models = block`.
Yardstick, with public mahjong calibration numbers quoted by the roadmap (different prediction
targets, so only a rough reference): Sasaki hand model ECE 0.00188 (diffusion) / 0.000801
(direct prediction), riichi-outcome model ECE 0.008511.
- (a) probe 0.0099 = 1.2× the riichi-outcome number and 5–12× the Sasaki numbers;
- (b) GRP 0.0242 = 2.8× the riichi-outcome number and 13–30× the Sasaki numbers;
- (c) Q τ=1 0.0364 = 4.3× the riichi-outcome number.

Neither gate-bearing object is an *order of magnitude* worse than the public numbers, which is
the roadmap's explicit "already close → do not build the belief head" condition (the exact
order-of-magnitude line is a judgement, and the 2.8×/13–30× spread on (b) is recorded here so
the call can be audited).

An honest negative is the result: **no `not_estimable` sub-object was needed**; the GRP truth
aligned exactly (10,000/10,000 games, 0 load errors, 0 bad-rank games, no last-3-rounds
degradation), and all three numbers are measured, not proxied.

## Which of these numbers may influence the belief gate (explicit)

Pre-registered in `PREREG.md` §7 before the run:

1. **Gate-bearing (headline, both verdicts):**
   - `E_a = 0.0099` — A2 probe, primary label, held-out val.
   - `E_b = 0.0242` — GRP top-1 confidence ECE.
2. **Gate-bearing in the literal three-object rule (`gate_three_default`) only:**
   - `E_c = 0.0364` — Q action-confidence ECE at τ = 1. It did not change the verdict
     (0.0364 ≤ 0.05), but its *concept* is weaker: it needs a temperature assumption.
3. **Informational — must NOT be read as an ECE gate trigger:**
   - `E_c(τ = 0.05) = 0.1897`: the fork's deployed play temperature (config
     `boltzmann_temp = 0.05`, `boltzmann_eps = 0.005`) turns softmax(Q) into an almost
     0/1 confidence (mean 0.9855) while the model still agrees with the recorded action only
     79.6 % of the time. This is a *property of a decision rule that is not a probability*,
     reported for transparency; it is why the τ = 1 arm, not this one, is the primary.
   - Value-table diagnostics (§(c2)), multiclass Brier, AUCs, per-round breakdowns: context.

**Two thresholds, not to be conflated** (also in `PREREG.md` §1 and in `results.json → gate`):
this audit's triage gate is **ECE > 0.05 ⇒ allow**; the belief-head *delivery* gate
(todo 29) is **ECE ≤ 0.15** for a model that already exists. Note the irony recorded here on
purpose: the τ = 0.05 arm (0.1897) would fail *both*; it is still not a gate trigger because
it is not a pre-registered probability of the model.

This audit is an **independent (ECE) route** to the same decision A2 reached by AUC
(`runs/exp/A2/README.md`: `pipeline_valid=false`, `gate_belief=block`). It neither re-opens
nor overrides A2 — it agrees with it, for a different reason: not "the embedding lacks
hidden-state signal" but "the probabilities that do exist are already calibrated".

## Results

### (a) A2 probe probabilities — ECE 0.0099, Brier 0.0943

Re-derived with A2's exact protocol and **reproduced A2 bit-for-bit**
(`delta_auc = delta_ece = delta_brier = 0.0`, `reproduced_within_1e-3 = true`); sample
262,339 decision points / 400 files / 209,610 train / 52,729 val, identical to
`runs/exp/A2/results.json`. Val-set, all five A2 labels:

| label | AUC | ECE | Brier |
|---|---|---|---|
| **`opp_any_tenpai` (gate-bearing)** | 0.9334 | **0.0099** | 0.0943 |
| `opp_any_shanten_le1` | 0.8982 | 0.0153 | 0.1180 |
| `opp_any_shanten_le2` | 0.9101 | 0.0222 | 0.0604 |
| `opp_any_aka_in_hand` | 0.6055 | 0.0138 | 0.1699 |
| `ctrl_opp_any_meld_visible` | 0.7947 | 0.0188 | 0.1821 |

Reliability (primary label, 15 equal-width bins): gaps are ≤ 0.038 everywhere and the two
extreme bins hold 46 % of the mass with gaps +0.0002 (`[0,0.07)`) and +0.0018 (`[0.93,1.0]`);
the mild negative gaps in the middle (e.g. conf 0.5000 vs acc 0.4744) mean the probe is
slightly *under*-confident around 0.3–0.6. The full table is in
`results.json → objects.a_probe.labels.opp_any_tenpai.reliability` and in
`a2b_probe_val.npz` (per-sample probabilities and labels).

Caveat that stays attached to this number: the probe is a fresh 1024→1 logistic readout fit
on 209,610 points, i.e. an *upper bound* on what a belief head could extract and a *lower*
bound on the calibration problem. It is not the network's own uncertainty, and it does not
re-open A2's substantive negative (the 24-feature public baseline scored AUC 0.9399 > 0.9334).

### (b) GRP 24-permutation distribution — top-1 ECE 0.0242, top-1 Brier 0.1467

Sample: **8,000 games / 85,144 round predictions** (`files[2000:]` of the pre-registered
k = 10,000, seed-13 reservoir draw; the first 2,000 files were used by the pre-run timing
probe and are reported separately). Truth: the game's own final ranking — one prediction per
kyoku, no subsetting, no proxy.

| metric | value |
|---|---|
| top-1 accuracy / mean confidence | 0.2220 / 0.2009 |
| **top-1 ECE** | **0.0242** (CI [0.0214, 0.0282]) |
| top-1 Brier | 0.1467 |
| multiclass Brier (24 classes) | 0.8623 (uniform 0.9583) |
| pooled one-vs-rest ECE (all 24 class probs) | **0.0019** |
| pooled OvR Brier | 0.0359 |
| marginal P(player finishes r) pooled OvR ECE (16 cells) | 0.0066 |
| marginal ECE, rank 1st only | 0.0082 |
| uniform-predictor reference | top-1 ECE 0.1804 |
| constant train-frequency reference (50/50 file split, held-out half) | top-1 ECE 0.0064 / Brier 0.0399, acc 0.0416, mc Brier 0.9584 |

Reference caveat, worth stating because it disciplines the reading of every ECE here: the
constant predictor posts a *lower* top-1 ECE (0.0064) than the GRP (0.0242) while being nearly
useless (accuracy 0.0416 vs 0.2220). **ECE is a calibration metric, not a discrimination
metric**; the GRP's value is that it is *both* accurate and calibrated, and a low ECE alone
must never be read as "the model is good".

Reliability structure: at low confidence the GRP is accurate within ~0.005; above ~0.4 the
observed hit rate *exceeds* the stated confidence (e.g. `[0.53,0.60)`: conf 0.5615 vs acc
0.7089; `[0.67,0.73)`: 0.677 vs 0.909 on 11 rounds) — the model is **under-confident in its
confident tail**, but that tail is thin: it never predicts above 0.69 (max 0.688) and only 248
rounds (0.3 %) land above 0.60. Broken down by round index, accuracy climbs from 0.044 (round 0,
chance) to 0.393 (rounds 8+) and the ECE grows from 0.0007 to 0.0638 — late rounds are where
the head is sharp and where it under-states itself.

Subsets (transparency, `objects.b_grp`): the probed 2,000-file subset gives ECE 0.0245
(21,573 rounds) — indistinguishable from the primary sample; the 400-file A2 sample gives
0.0454 (16,936 rounds), i.e. sample-to-sample scatter of that order. Loader integrity:
0 errors, 0 zero-round games, 0 non-permutation ranks ⇒ `truth_alignment =
"game_final_rank_by_player (exact; no degradation needed)"`.

Bottom line: the GRP is a calibrated probability model (OvR ECE 0.0019 is remarkable) that is
slightly *under*-confident; it is not a source of a calibration gap.

### (c) DQN Q values — action-confidence ECE 0.0364; value scale over-dispersed

**(c1) Action-confidence calibration.** p = softmax(Q/τ) over legal actions; truth = the
action recorded in the log (human play), i.e. **agreement calibration, not optimality**:

| arm | match rate | mean confidence | ECE | Brier |
|---|---|---|---|---|
| **τ = 1.0 (primary)** | 0.7959 | 0.7594 | **0.0364** (CI [0.0346, 0.0382]) | 0.1241 |
| τ = 0.05 (deployed temperature, informational) | 0.7959 | 0.9855 | 0.1897 (CI [0.1875, 0.1918]) | 0.1919 |

At τ = 1 the shape is a mild, systematic over-confidence in the mid range (gaps +0.04…+0.06
for conf 0.4–0.9) and near-perfect calibration in the huge top bin (33 % of decisions at
conf 0.9856, acc 0.9914). The `[0.93,1]` bin plus the wide 0.4–0.9 band is what keeps the ECE
at 0.036. Held-out subset agrees (ECE 0.0370). The τ = 0.05 arm shows the deployed rule
turning Q into near-certainty it cannot support — an artifact of the temperature, not of Q
probabilities, and not a gate input.

**(c2) Value-scale reliability (diagnostic, not ECE).** Predicted `Q(a_recorded)` vs the
exact training target `γ^steps_to_done · kyoku_reward[at_kyoku]` (checkpoint's own
`γ = 1`, `pts = [6,4,2,0]`), n = 262,339, 15 equal-count bins:

| bin (predicted Q) | n | mean predicted | mean realized | bias |
|---|---|---|---|---|
| [−9.096, −0.805) | 17,490 | −1.806 | −0.085 | **−1.721** |
| [−0.805, −0.263) | 17,489 | −0.479 | −0.120 | −0.359 |
| [−0.263, −0.115) | 17,489 | −0.178 | −0.126 | −0.052 |
| [−0.115, −0.034) | 17,489 | −0.072 | −0.093 | +0.021 |
| [−0.034, +0.028) | 17,489 | −0.002 | −0.060 | +0.058 |
| [+0.028, +0.085) | 17,490 | +0.057 | −0.042 | +0.099 |
| [+0.085, +0.138) | 17,488 | +0.112 | −0.028 | +0.140 |
| [+0.138, +0.191) | 17,490 | +0.164 | −0.011 | +0.176 |
| [+0.191, +0.244) | 17,489 | +0.218 | −0.000 | +0.218 |
| [+0.244, +0.300) | 17,489 | +0.272 | +0.015 | +0.257 |
| [+0.300, +0.360) | 17,490 | +0.329 | +0.035 | +0.294 |
| [+0.360, +0.428) | 17,489 | +0.393 | +0.035 | +0.358 |
| [+0.428, +0.513) | 17,489 | +0.468 | +0.060 | +0.408 |
| [+0.513, +0.646) | 17,489 | +0.573 | +0.116 | +0.456 |
| [+0.646, +3.015) | 17,490 | +0.855 | +0.419 | +0.436 |

Overall: mean bias +0.053 points (8 % of the target std 0.670), MAE 0.608, RMSE 0.875,
**Spearman ρ = 0.145**, `value_binned_calibration_error = 0.337 points = 0.50 × target std`.
Reading: on average Q is nearly unbiased, but its **spread is much wider than the realized
value spread** (over-dispersion): the lowest bin predicts −1.81 and realizes −0.09, while the
top bin predicts +0.86 and realizes +0.42. Within-round ordering by Q is nearly uninformative
(ρ = 0.145). Two caveats stay attached: the target is per-kyoku constant, and for non-final
rounds it is itself GRP-derived (so this measures fit to the training objective, not to
realized outcomes).

## Pipeline self-check (PREREG §6)

Run before the real audit and re-embedded in `results.json → self_check`:

| check | result |
|---|---|
| perfectly calibrated synthetic data (p = k/1000, exactly k positives each) | **ECE = 9.3e-18 < 1e-6** (999,000 samples) |
| known miscalibration (p ≡ 0.7, 50 % positives) | ECE = 0.20000000000000007 (exact) |
| top-1 path (24-class, exact hits per confidence level) | ECE = 6.0e-17 < 1e-6 |
| Brier sanity (p=0,y=0 / p=1,y=1) | 0.0 |
| cluster-bootstrap path reproduces the point ECE | yes (1 group) |

## Interpretation

1. **The calibration hypothesis loses.** Roadmap step 6 asked: is the model "not learning"
   because it is "not calibrated"? On this stack the answer is no: the A2 probe (0.0099) and
   the GRP rank head (0.0242, and 0.0019 one-vs-rest) are well calibrated (compare the ~0.8–3×
   spread against the public yardstick in the TL;DR), so a calibration-aware training objective
   has little headroom to win here. What A2 already showed is the binding constraint: the
   *signal*, not its calibration.
2. **The DQN's Q probabilities are the only visibly off part** — and only because a
   temperature was assumed to read a value as a probability. At τ = 1 the ECE is 0.036 (still
   ≤ 0.05); at the deployed τ = 0.05 it is 0.19, which says the deployed decision rule is
   effectively deterministic and its implied confidence is meaningless, not that Q *values*
   are wrong. The value-scale table is the substantive Q finding: **almost unbiased on
   average, badly over-dispersed, and nearly uninformative for within-round ordering
   (ρ = 0.145)**.
3. **Gate recommendation for todo 15:** `gate_calibration = block`. Nothing in this audit
   justifies the belief-head line on calibration grounds, and A2 independently reached
   `gate_belief = block` on AUC grounds. The two routes agree.
4. **What would change this conclusion:** a belief head that is *not* a linear readout
   (A2's limitation), a different checkpoint/temperature in the fork's own training regime,
   or a value target that is not GRP-derived. None of these is measured here.

## Limitations / honest caveats

- One policy checkpoint (`runs/exp/PINNED/mortal.pth.at-175200`, step 175,200), one GRP
  checkpoint (`checkpoints/grp.pth`, step 78,000), one corpus, obs v4, non-oracle inference.
- (a) is a logistic readout, not a belief head; its ECE is a lower bound on the calibration
  problem and its AUC a representation-level result already qualified by A2.
- (b) rounds within a game are strongly correlated; CIs are cluster (file) bootstraps. The
  confident tail of the GRP is thin (248 rounds > 0.60, max 0.688), so "under-confident tail"
  rests on small n.
- (c) match rate is agreement with recorded human actions, not optimality; τ = 1 is an
  assumption; the value target is GRP-derived for intermediate rounds; the Q net was
  evaluated in float32.
- The `[0.93,1]` τ = 0.05 bin contains 94.5 % of points; its ECE contribution is dominated by
  that single bin.
- ECE binning is the standard 15 equal-width choice, shared by all three objects; ECE is
  sensitive to binning by construction, which is exactly why the binning was frozen in
  `PREREG.md` before the run.
- Reproducibility of (a) is exact (`delta = 0.0` vs A2); (b) and (c) are deterministic
  given the seeds (no fitting is involved).

## Reproduction

```bash
source scripts/activate_env.sh
python scripts/exp_a2b_calibration.py --selftest   # metric self-check only, seconds
python scripts/exp_a2b_calibration.py             # full audit, 233 s, peak RSS 6.58 GB
```

Determinism anchors: `random.Random(13)` for both reservoir samples (400 files for (a)/(c),
10,000 for (b)), split seed 13, probe seed 0, bootstrap seed 7, GRP/policy checkpoints by
sha256 (`results.json → checkpoints`). `a2b_grp_rounds.npz` stores the raw 24-way
probabilities so every (b) number can be recomputed without a model forward.
