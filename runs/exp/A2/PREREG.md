# A2 Pre-Registration — Linear Probe on the Frozen 1024-d Embedding

Status: **written before the probe run.** No probe (AUC/ECE) number was computed before
this file was committed to disk. A layout-verification pilot (structural checks only, no
probe fit) was run first; its observations are disclosed in §6.

Repo HEAD at pre-registration: `a41370d547c5d8019eb295432b86a78886da22bb` (todo 6 eval and
other tasks are running concurrently; uncommitted changes to `configs/config.toml`,
`RUNBOOK.md`, `scripts/parse_eval.py`, `vendor/**` are NOT touched by this experiment).

## 1. Question / probe target

Does the **frozen 1024-d representation** produced by the pinned policy network already
linearly encode **opponents' hidden information** (their concealed-hand state), which is
not part of the network input?

Interpretation is two-sided and pre-committed:
- positive result → the representation carries opponent hidden state; a belief head
  (roadmap todo 28) has a basis;
- negative result → **the embedding does NOT (linearly) encode opponent hidden state**;
  this is a legitimate, reportable outcome, not a failure.

Primary target: `opp_any_tenpai` = `min(shanten[opp0], shanten[opp1], shanten[opp2]) <= 0`
(at least one opponent is tenpai), where shanten is decoded from the oracle tensors (§3).

## 2. Model / embedding (frozen — no training, no fine-tuning)

- Checkpoint: `runs/exp/PINNED/mortal.pth.at-175200`, 130,780,507 B,
  sha256 starts `f0a611b97f318e0b…` (full hash recorded in `results.json`).
- Loaded with `torch.load(..., weights_only=True)`; weights under `state['mortal']`;
  architecture from `state['config']`: `version=4`, `resnet={conv_channels:192,num_blocks:40}`.
  The first conv weight is `(192, 1012, 3)` ⇒ the model is **not oracle** at inference:
  it sees only the 1012-channel observation. No oracle channels are fed to the network.
- Embedding: `phi = brain.encoder(obs)` (`vendor/mortal/mortal/model.py:156`), i.e. the
  1024-d pre-activation encoder output, not the post-Mish `Brain.forward` output.
- `brain.eval()`, `torch.inference_mode()`, float32, no AMP. Weights, BatchNorm running
  stats and everything else stay frozen (nothing is trained).

## 3. Data and labels

### 3.1 Sampling

- Loader: `libriichi.dataset.GameplayLoader(version=4, oracle=True)`.
- Corpus: `datasets/extracted/**/*.mjson` (gzip mjson, 2,512,424 files at prereg time).
- File sample: **reservoir sample of K=400 files** with `random.Random(13)` over
  `glob.iglob("datasets/extracted/**/*.mjson", recursive=True)` (Algorithm R, deterministic).
  If the final decision-point count is `< 200,000`, additional independent reservoir
  samples of 100 files with `random.Random(14), Random(15), …` are appended until ≥ 200,000.
- Every decision point of every one of the 4 POVs per game is used. **No filtering.**
  (The loader's kan-select duplicate entries are kept.)
- Expected yield ≈ 1,000 decision points/file (pilot measured 998.2), so ≈ 300k points;
  the pre-registered requirement is **N ≥ 200,000 decision points** (task hard floor).

### 3.2 Oracle-tensor layout (verified against Rust source; pilot in §6)

`take_invisible_obs()` returns `(217, 34)` per decision point. Rows `[0,51)` encode the 3
opponents in relative seat order `shimocha, toimen, kamicha`, each with a 17-row block
(`base = 17*k`), per `vendor/mortal/libriichi/src/dataset/invisible.rs:163-209`:

| rows | meaning |
|---|---|
| `base+0 .. base+3` | concealed-hand tile-count planes: row `base+c`, column `t` is 1 iff hand holds `>c` copies of kind `t`; per-kind count = column sum over these 4 rows (`assign_rows`) |
| `base+4 .. base+6` | red-five-in-hand planes 5mr/5pr/5sr (all-0 or all-1 row) |
| `base+7 .. base+13` | shanten one-hot (7 rows; exactly one row is all-ones → shanten `n ∈ [0,6]`, `fill(idx+n)`) |
| `base+14` | `n/6` (normalized shanten) |
| `base+15` | waits one-hot over tile kinds |
| `base+16` | furiten flag (all-0 or all-1 row) |

Rows `[51,217)` are wall/rinshan/dora/ura (not used here).

### 3.3 Labels

| label | definition | kind |
|---|---|---|
| `opp_any_tenpai` **(primary)** | `min_k shanten_k <= 0` | hidden |
| `opp_any_shanten_le1` | `min_k shanten_k <= 1` | hidden |
| `opp_any_shanten_le2` | `min_k shanten_k <= 2` | hidden |
| `opp_any_aka_in_hand` | any opponent concealed hand contains a red five (aka rows) | hidden |
| `ctrl_opp_any_meld_visible` | any opponent concealed tile count `< 13` (i.e. has a called meld) | **visible positive control** (derivable from public melds; AUC expected ≥ 0.9) |

Concealed hand size = sum over kinds of decoded counts.

### 3.4 Decode-verification gates (critical; failure ⇒ degrade, see §3.5)

Run over **all** drawn decision points:
1. shanten block: exactly one all-ones row per opponent — 0 violations required;
2. `norm == shanten/6` — 0 violations required;
3. counts integer, each in `[0,4]`; aka/furiten/waits planes binary — 0 violations required;
4. own shanten: obs one-hot planes `[862,869)` (own shanten, `obs_repr.rs` IntegerEncoder)
   equals `take_shantens()` (clamped to `[0,6]`) — 0 violations required;
5. concealed hand size ∈ `{13 - 3*fuuro_groups - 4*ankan_kinds, +1}`, where fuuro/ankan
   groups are counted from the public obs planes `[771,831)` / `[832,835)`
   (`obs_repr.rs:303-329`) — violation rate ≤ 1e-4 of opponent observations allowed
   (rare mid-kan edge states); anything above ⇒ degrade.

`waits_nonzero ⇔ shanten == 0` is recorded as **informational** (Rust has known rare
exceptions; pilot saw 4/939 tenpai points with an empty waits row). It does not gate.

### 3.5 Degradation rule (mandatory, from the plan QA rule)

If any critical gate in §3.4 fails beyond tolerance, we do **not** silently relabel:
the primary target becomes **`own_tenpai` = own shanten (`take_shantens()`) <= 0** and
`results.json` + `README.md` state explicitly that the label is the **trainee's own
shanten**, never a claimed opponent label. (`label_source: "own_shanten (decode degraded)"`.)

## 4. Probe, split, metrics, control

- Train/val split: **by source file**, 80/20, `random.Random(13).shuffle(file_ids)`
  (all points of a file, hence all 4 POVs of a game, stay on one side — no same-game leakage).
- Features: `phi` (1024-d), standardized with **train** mean/std (σ floor 1e-6).
- Probe: plain logistic regression `nn.Linear(1024, 1)`, AdamW `lr=1e-3`, `weight_decay=1e-5`,
  batch 4096, **100 epochs** (see Amendment A1), model-init/shuffle seed 0 — identical recipe
  for every label and control.
- Metrics on the held-out split: AUC (Mann–Whitney, tie-averaged ranks), ECE (15 equal-width
  probability bins, standard Σ n_b/N·|acc_b−conf_b|), Brier; primary AUC gets a bootstrap
  95% CI (1000 resamples, seed 7).
- Control (pipeline sanity): labels globally permuted (same split, same recipe):
  seeds **100, 101, 102**; `auc_random` = mean of the three; per-seed values reported.
  Secondary labels get one control fit (seed 100).

## 5. Decision rule (pre-committed)

1. **Pipeline valid** iff for the primary label `|auc_random − 0.5| < 0.02` AND the
   positive-control label `ctrl_opp_any_meld_visible` has val AUC ≥ 0.90.
   If invalid → the pipeline is broken: fix it; do not report a probe number.
2. If valid, evidence tiers for the primary AUC:
   - **`gate_belief = allow`** (embedding linearly encodes opponent hidden state):
     primary AUC ≥ 0.65 **and** bootstrap CI lower bound > 0.5;
   - **`gate_belief = block`**: primary AUC < 0.65 or CI includes 0.5.
   (Numbers recorded for todo 15; A2 is a linear-probe result only — a weak/null result
   does not exclude non-linear decodability.)

## 6. Disclosed pre-registration pilot (structural only)

6 files, 5,989 decision points, 17,967 opponent observations:
- zero critical violations for §3.4 gates 1–4 and 5;
- hand-size distribution `{4:26, 7:251, 10:947, 13:16743}`; own-shanten obs decode == `take_shantens()` for all points;
- label prevalence (all-opponents min shanten): `le0=15.7%`, `le1=33.4%`, `le2=70.4%`;
  `ctrl_opp_any_meld_visible=17.2%`;
- informational: `waits≠∅ ⇔ shanten==0` held for 99.6% of tenpai opponent observations (4 exceptions).
No AUC, ECE or any probe fit was computed during the pilot.

## 7. Artifacts & runtime discipline

- Outputs: `runs/exp/A2/{PREREG.md,results.json,README.md}`; evidence `.omo/start-work/evidence/T13.txt`.
- Embedding computed in GPU chunks ≤ 4096, file batches of 3; embeddings+labels ≈ 1–1.5 GB RAM;
  `MemAvailable` monitored, chunk size reduced if available RAM < 4 GB.
- A 12k-game GPU evaluation (roadmap todo 6) is running concurrently; this probe is the
  lower-priority job and must not disturb it.

## 8. Amendments (recorded before the probe run)

- **A1 (planned value 30 → 100 epochs).** A synthetic convergence pre-check
  (200k×1024, 20 sparse signal features, prevalence 0.24, same optimizer/batch/seed
  recipe) gave val AUC 0.9275 @30 epochs, 0.9282 @100, 0.9281 @300 — AUC is converged by
  30 epochs, but probability calibration (loss 0.330 @30 vs 0.285 @100) benefits from 100.
  The recipe was therefore fixed at **100 epochs** before any probe fit. This pre-check
  used synthetic data only.
- **A2 (sample size K 300 → 400).** After A1, a 20-file reservoir sample (seed 13)
  measured a mean yield of **670 decision points/file** (13,403 total), i.e. K=300 would
  land at ≈201k — too close to the 200k floor. K was raised to **400** (expected ≈268k)
  before any probe fit; the fallback draw rule is unchanged.
