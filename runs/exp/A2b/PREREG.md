# A2b Pre-Registration — Calibration Audit (probe / GRP / DQN-Q), zero training

Plan: `.omo/plans/fork-roadmap-steps-1-10.md` todo 14 · Roadmap: `docs/FORK_ROADMAP.md` step 6.
Status: **written before any A2b calibration number was computed.** No ECE, Brier,
reliability table or calibration error for any of the three objects below existed anywhere
when this file was written; the only pre-run measurements were format/size probes, disclosed
in full in §8 (one of them, a GRP timing probe, incidentally reported top-1 accuracy and
multiclass Brier on a file subset — both are repeated verbatim in §8, and that subset is
excluded from the primary object-(b) sample, see §4).

Repo HEAD at pre-registration: `f9800fe0616e0dafe090eb4026d178dc523f1eef` (other tasks may
leave unrelated files dirty; this audit only writes new paths under `runs/exp/A2b/` plus one
new script `scripts/exp_a2b_calibration.py` and may import `scripts/exp_a2_probe.py`).

**Zero training.** Nothing is trained or fine-tuned by this audit. Object (a) *re-derives* the
A2 probe probabilities (logistic readout fit) because A2 persisted only aggregates; that fit
is the pre-registered A2 protocol replayed with identical seeds, not new model training.

---

## 1. Question, scope, gate

This audit asks one question per object: **"when the model says X %, does X % happen?"**
It measures and nothing else — no model is fitted, selected, or tuned on the outcome.

Pre-committed gate (feeds plan todo 15): **ECE > 0.05 on any of the three primary numbers ⇒
`gate_calibration = allow`** (a calibration gap exists, so the calibration-aware belief head
of step 10 has a justification); **all three ≤ 0.05 ⇒ `block`** (close the belief line as a
negative result). Exact composite rules in §7.

**Two thresholds that must not be conflated:**
- this audit's *readiness* gate: **ECE > 0.05 ⇒ allow** (todo 15 / roadmap step 6);
- the belief head's *delivery* gate (plan todo 29 / roadmap step 10): the trained head must
  reach **ECE ≤ 0.15**. 0.15 is the acceptance bar for a model that exists; 0.05 is the
  triage threshold for whether building it is warranted. A model at ECE 0.10 would pass the
  delivery gate and still have failed this audit's triage.

| # | object | probability | truth | primary metric |
|---|---|---|---|---|
| (a) | A2 linear probe, re-derived with A2's seeds | `P(opp_any_tenpai)` | oracle-decoded label | ECE(15) + Brier |
| (b) | GRP 24-permutation head | softmax over the 24 permutations | game-final `rank_by_player` | top-1 ECE(15) + top-1 Brier |
| (c) | DQN Q values | softmax(Q/τ=1) over legal actions | action recorded in the log | ECE(15) + Brier |

Object (c) additionally carries the value-scale reliability table against the exact Q training
target (§5) — reported as a table, never as "ECE".

## 2. Metric definitions (code-level)

- **ECE (binary / top-1 / pooled one-vs-rest)**: 15 equal-width bins of [0,1]
  (`np.linspace(0,1,16)`), assignment `np.digitize(p, edges[1:-1])`, empty bins skipped:
  `ECE = Σ_b (n_b/N)·|acc_b − conf_b|`, `conf_b` = mean predicted probability in bin b,
  `acc_b` = empirical frequency of the positive event in bin b. Shared single implementation;
  the top-1, pooled and marginal variants reduce to it (verified in §6).
- **Brier (binary / top-1)**: `mean((p − y)²)`, `y ∈ {0,1}`.
- **Multiclass Brier** (object b): `mean_i Σ_k (p_ik − y_ik)²` over the 24 one-hot classes.
- **Top-1 confidence** (object b): `conf_i = max_k p_ik`; `y_i = 1[true_i ∈ argmax_k p_ik]`
  with a 1e-6 tie tolerance; then the binary ECE/Brier functions are applied to `(conf, y)`.
- **Pooled OvR ECE** (object b): flatten all `(p_ik, 1[k = true_i])` pairs and apply the binary
  ECE once (24 classes × n rounds samples).
- **Marginals** (object b): `M[i, r] = Σ_k p_ik · 1[perm_k[i] = r]` (identical to
  `GRP.calc_matrix`); 16 (player, rank) OvR cells pooled for one ECE, plus the per-rank-row
  reliability table for `R=1st` (the decision-relevant row).
- **Value-scale table** (object c, diagnostic): 15 **equal-count** (quantile) bins of the
  predicted `q(a_recorded)`; per bin: n, mean predicted, mean realized target, bias
  (predicted − realized). Scalars: mean bias, MAE, RMSE, Spearman ρ, and
  `value_binned_calibration_error_pts = Σ_b (n_b/N)·|mean_pred_b − mean_real_b|` in points,
  also reported normalised by the target's std. **This quantity is not ECE and is not
  gate-bearing.**
- Every ECE is accompanied by its reliability table: `[lo, hi)`, n, weight, mean confidence,
  empirical frequency, gap (`acc − conf`).

## 3. Object (a) — A2 probe probabilities (re-derived and persisted)

`runs/exp/A2/results.json` persisted aggregates only (AUC/ECE/Brier), not per-sample
probabilities, so they are re-derived with the **identical pre-registered protocol**, by
importing `scripts/exp_a2_probe.py`:

- reservoir sample **k=400, `random.Random(13)`** over
  `glob.iglob("datasets/extracted/**/*.mjson", recursive=True)` (A2 §3.1);
- `libriichi.dataset.GameplayLoader(version=4, oracle=True)`, all 4 POVs, no filtering;
- labels + decode-verification gates from A2's `verify_and_label`;
- `phi = brain.encoder(obs)` of `runs/exp/PINNED/mortal.pth.at-175200` (frozen, `eval()`,
  `inference_mode`, float32, non-oracle path);
- file-level 80/20 split, `random.Random(13).shuffle`, all points of a file on one side;
- standardisation with train mean/std (σ floor 1e-6);
- probe `nn.Linear(1024,1)`, AdamW lr=1e-3, wd=1e-5, batch 4096, 100 epochs, seed 0.

Metrics are computed on the **held-out val split**. Gate-bearing: the pre-registered primary
label `opp_any_tenpai`. Secondary: `opp_any_shanten_le1`, `opp_any_shanten_le2`,
`opp_any_aka_in_hand`, `ctrl_opp_any_meld_visible`.

**Reproduction check (mandatory):** primary AUC / ECE / Brier must reproduce
`runs/exp/A2/results.json` = 0.9334 / 0.0099 / 0.0943 within 1e-3; a larger deviation means
the sample or fit drifted, and the audit must report the difference (never silently
substitute numbers).

**Caveat stated up front:** the probe is a *fresh logistic readout* of the frozen
representation, i.e. an upper bound on what a belief head could extract. Its ECE is therefore
its own honest number, not the network's; the A2 README's substantive negative (a 24-feature
public baseline at AUC 0.9399 ≥ phi's 0.9334) is a different question and is not re-derived
here. A2b does not re-run A2's permuted-label controls (AUC validity is A2's claim).

Per-sample val probabilities (all 5 labels) and val labels are persisted to
`runs/exp/A2b/a2b_probe_val.npz`.

## 4. Object (b) — GRP 24-permutation distribution

- **Model**: `checkpoints/grp.pth`, sha256 recorded in `results.json`; hidden 64, 2 layers,
  float64, `eval()`; the same file the pinned policy's own config points at
  (`config.grp.state_file`).
- **Input**: for a game with feature matrix `F` (n_rounds × 7) =
  `[grand_kyoku, honba, kyotaku, score_i/1e4 for i=0..3]`, the prediction for round index `t`
  uses the prefix `F[:t+1]` — exactly the training-time prefix (`train_grp.py`,
  `reward_calculator.calc_grp`).
- **Probability**: the raw 24-way softmax of `GRP.forward` (`logits.softmax(-1)`), index order
  `itertools.permutations(range(4))`.
- **Truth**: `Grp.take_rank_by_player()` of the **same game** (game-final ranking from
  `Rankings::new` over final scores), mapped to the permutation index exactly as
  `GRP.get_label` does. Alignment is exact for every round — no proxy, no subsetting, and
  every round of every loaded game is scored.
- **Degradation rule (plan QA, recorded here pre-run)**: if a game's `rank_by_player` cannot
  be decoded or its final ranking is undefined, truth falls back to the **last three kyoku of
  each game** and the group is labelled `truth_alignment: "last3_rounds"`. If neither form is
  defensible, the group is written as `"not_estimable"` with the concrete reason; no invented
  proxy is substituted.
- **Sample**: reservoir sample **k=10000, `random.Random(13)`** (declared here, before the
  run) loaded with `libriichi.dataset.Grp.load_gz_log_files`; ≈ 10.7 rounds/game ⇒ ≈ 107k
  round predictions expected. The primary (b) results use **files[2000:] of that draw**
  (≈ 85k rounds) because the first 2000 files were used by the pre-run timing probe (§8);
  that probed subset is reported separately as a transparency replication, not as the
  headline. A chunk that fails to load is retried file-by-file and every skipped file is
  counted and reported (`load_errors`).
- **Primary (gate-bearing)**: top-1 confidence ECE + top-1 Brier, all rounds pooled.
- **Secondary**: multiclass Brier; pooled OvR ECE over the 24 classes; marginal 4×4 OvR ECE;
  per-rank-row reliability (row R=1st); top-1 accuracy/confidence and ECE broken down by round
  index (early vs late rounds).
- **References**: uniform (1/24) predictor; constant "train-frequency" predictor fit on a
  file-level 50/50 split (`Random(13)`) and evaluated on the held-out half.
- Round-level probabilities + truth + file/round indices persisted to
  `runs/exp/A2b/a2b_grp_rounds.npz`.

## 5. Object (c) — DQN Q values

- **Model**: `state['current_dqn']` of `runs/exp/PINNED/mortal.pth.at-175200`
  (v4, `net = Linear(1024, 47)`), `q = DQN(Mish(phi), mask)`, per decision point of the same
  400-file A2 sample as object (a). Nothing is fitted for (c).
- **(c1) Primary probability**: `p = softmax(q_legal / τ)` with **τ = 1.0** (the Gibbs reading
  of Q), softmax restricted to the legal mask; `conf = max_a p(a)`;
  `y = 1[a_recorded ∈ argmax_legal q]` (1e-6 tie tolerance). Truth = the action recorded in
  the log, so this measures **agreement calibration with the recorded (human) behaviour
  policy, not optimality**; the temperature is an assumption, not a likelihood of the DQN.
- **(c1-sensitivity, secondary, not gate-bearing)**: τ = 0.05 — the fork's deployed play
  temperature (`[train_play.default] boltzmann_temp`, `configs/config.toml`) — same metrics.
- **(c2) Value-scale reliability (diagnostic, no gate)**: predicted `q(a_recorded)` vs the
  exact training target `γ^steps_to_done · kyoku_rewards[at_kyoku]`
  (`dataloader.py:91-122`; `train.py:236`), computed with the checkpoint's own saved `env`
  (`γ=1`, `pts=[6,4,2,0]`) via `RewardCalculator` and the same GRP checkpoint.
  **Stated up front:** for non-final rounds that target is itself GRP-derived (expected point
  deltas), so (c2) measures *fit to the training objective*, not fit to realized outcomes.
- **Sample**: all 400 files (no fitting ⇒ no held-out requirement); the val subset is also
  reported for comparability with (a).
- Per-decision records persisted to `runs/exp/A2b/a2b_q_records.npz`.

## 6. Pipeline self-check (mandatory; if it fails, fix the metric — do not report numbers)

Synthetic, executed before the real audit:

1. **Perfect calibration**: `p = k/1000` for k = 1..999, 1000 samples per value, with exactly
   `k` positives per value ⇒ `ECE < 1e-6` (required hard gate).
2. **Known miscalibration**: `p ≡ 0.7` with 50 % positives ⇒ `ECE = 0.2` (±1e-12).
3. **Top-1 path**: 24-class construction in which each confidence level `v` has exactly
   `round(v·n)` hits ⇒ top-1 ECE < 1e-6.
4. **Brier sanity**: `(p=0, y=0)` and `(p=1, y=1)` ⇒ 0.

All three objects' ECE paths are the same shared binary routine (top-1 and pooled reduce to
it), so (1)–(3) validate every reported ECE.

## 7. Gate rules (pre-committed)

Let `E_a` = object (a) primary-label ECE, `E_b` = object (b) top-1 ECE,
`E_c` = object (c) τ=1 action-confidence ECE.

- `gate_three_default` (literal plan rule): `allow` iff `max(E_a, E_b, E_c) > 0.05`, else `block`.
- `gate_two_models` (pre-committed robustness reading): `allow` iff `max(E_a, E_b) > 0.05`,
  else `block`. Rationale: (a) and (b) are probability models with exact binary outcomes; (c)'s
  τ=1 softmax is a temperature *assumption*, so if it is the only ECE above 0.05 the audit
  reports the τ=0.05 sensitivity arm and the value table instead of treating that alone as a
  calibration gap warranting the belief line.

Both verdicts are written to `results.json`. A2's own AUC-based verdict
(`pipeline_valid=false`, `gate_belief=block`) is an **independent route** to the same gate and
is neither re-opened nor overridden here.

## 8. Disclosed pre-run probes (format/size only)

All run on ≤ 2000 files, before this file; none computed an ECE:

- Loader formats: obs `(n,1012,34)`, masks `(n,46)`, actions int list, `at_kyoku`/`at_turns`
  bytes (`np.frombuffer(..., np.uint8)`), `take_rank_by_player` bytes, `take_shantens` list,
  GRP feature `(n_rounds,7)` float64.
- One game (file 1 of the A2 sample): 10 rounds, `cap.rank_by_player = [1,2,3,0]`;
  the per-round GRP probability of the true permutation rising 0.043 → 0.436;
  `calc_matrix` rows sum to 1; `calc_delta_pt` length = n_rounds.
- `Grp.load_gz_log_files`: 1000 files in 0.11 s; 10000 files → 106,717 rounds
  (mean 10.67 rounds/game, min 1, max 21).
- GRP forward throughput: 2000 games (8000 rounds) in 7.8 s.
  **Incidental numbers from that probe, disclosed verbatim:** top-1 accuracy **0.5946**,
  multiclass Brier **2.3321**, mean top-1 confidence **0.1981** (quantiles
  0.044/0.089/0.167/0.281/0.393/0.698) on the first 2000 files of the k=10000 draw.
  These 2000 files are **excluded** from the primary object-(b) sample (§4).
- No reliability table, no ECE, and no per-object-a/(c) metric was computed pre-run.

## 9. Runtime / resource discipline

This audit is the lower-priority job (roadmap todo 7 training smoke may run concurrently):
`nice -n 10`; GPU chunks ≤ 2048; file batches of 3; `MemAvailable` monitored with batch
shrinkage below ~4 GB; stdlib + numpy + torch only, no pip installs; no writes to
`runs/exp/A2/**`, `configs/config.toml`, `checkpoints/**`, `datasets/**`, `runs/1v3/**`,
`runs/variance/**`, `runs/exp/EVAL_A_175200/**`, `runs/exp/PINNED/**`, `vendor/**`.

## 10. Artifacts

`runs/exp/A2b/{PREREG.md, results.json, README.md}`, per-sample npz
(`a2b_probe_val.npz`, `a2b_grp_rounds.npz`, `a2b_q_records.npz`), evidence
`.omo/start-work/evidence/T14.txt`, commit
`feat(exp): A2b calibration audit (probe/GRP/Q)`.

`results.json` records `prereg.sha256` (this file) so the pre-registration can be verified
against the run.
