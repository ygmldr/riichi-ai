# A2 — Linear probe on the frozen 1024-d embedding

Date: 2026-09-17 · Plan: `docs/…` / `.omo/plans/fork-roadmap-steps-1-10.md` todo 13
Artifacts: `runs/exp/A2/{PREREG.md,results.json,supplementary.json,README.md}` ·
Scripts: `scripts/exp_a2_probe.py` (pre-registered run) and `scripts/exp_a2_supp.py`
(post-hoc supplementary).

## TL;DR

**The frozen 1024-d representation does NOT provide evidence of encoding opponents' hidden
state beyond what plain public information already gives — and the honest negative is the
right conclusion here.**

- Pre-registered primary probe (`opp_any_tenpai`: at least one opponent is tenpai, decoded
  from the oracle tensors over 262,339 decision points / 787,017 opponent observations):
  **AUC 0.9334** (bootstrap 95% CI [0.9312, 0.9354]), **ECE 0.0099**, Brier 0.0943,
  **random-label control AUC 0.5007** (seeds 0.4994 / 0.5002 / 0.5025) → the required
  control gate `|auc_random − 0.5| < 0.02` **passes**.
- **But** a 24-feature *public-information* logistic baseline (tiles left, own turn counter,
  opponent riichi flags, own shanten, round/honba, visible dora count) reaches
  **AUC 0.9399** on the same split — *higher than the embedding*. The embedding adds no
  hidden-state information beyond public game state; if anything it exposes slightly less.
- A genuinely hidden target, `opp_any_aka_in_hand` (does an opponent hold a red five):
  embedding AUC **0.6055**, public baseline AUC **0.6158** — public features again do at
  least as well.
- Positive-control check intended to prove the pipeline (a plainly *visible* label,
  "some opponent has a called meld", from `phi`): AUC **0.7947**, below the
  pre-registered ≥ 0.90 expectation → the prereg's `pipeline_valid=false /
  gate_belief=block` verdict fires. Post-hoc supplementary checks show the measurement path
  itself is sound (`exact meld planes → AUC 1.0000`; permuted labels ≈ 0.5); the shortfall
  is a property of the frozen encoder, not of the probe.
- **Recommendation for todo 15: `gate_belief = block`.** This is what the pre-registered
  rule emits, and it is also what the substance says: the belief-head line (todo 28) is not
  justified by A2.

## Setup

| item | value |
|---|---|
| checkpoint | `runs/exp/PINNED/mortal.pth.at-175200`, 130,780,507 B, sha256 `f0a611b97f318e0b32fd7ce97965da0a9299b959742da54f64051677c7dc93a9`, step 175,200 |
| model | `version=4`, `resnet={conv_channels:192,num_blocks:40}`; first conv `(192, 1012, 3)` ⇒ **non-oracle**, input is the 1012-channel obs only |
| embedding | `phi = brain.encoder(obs)` (`vendor/mortal/mortal/model.py:156`), 1024-d, frozen, `eval()`, float32, `inference_mode` |
| data | `libriichi.dataset.GameplayLoader(version=4, oracle=True)`; reservoir sample k=400 files (`random.Random(13)`) from `datasets/extracted/**/*.mjson`, all 4 POVs, no filtering |
| sample | **262,339 decision points** (655.8/file mean), 2010–2026 files, no duplicate files (400 unique md5) |
| split | by source file, 80/20 (`Random(13)`) ⇒ 209,610 train / 52,729 val points (320/80 files); all 4 POVs of a game stay on one side |
| probe | logistic regression `nn.Linear(1024,1)`, AdamW lr 1e-3, wd 1e-5, batch 4096, 100 epochs, seed 0, train-set standardization; identical recipe for all labels/controls |
| runtime | 786 s, peak RSS 6.49 GB, device cuda (batches ≤2048; ran niced alongside the todo-6 12k eval) |

Labels decoded from the oracle tensor rows (`invisible.rs:163-209`):
hidden hand counts (rows `base+0..3`), red fives (`base+4..6`), shanten one-hot
(`base+7..13`), normalized shanten (`base+14`), waits (`base+15`), furiten (`base+16`),
for the 3 relative seats shimocha/toimen/kamicha (`base = 17k`).

## Decode verification (mandatory gate from the plan)

Over all 262,339 points (787,017 opponent observations), violations of the critical gates:

| check | violations |
|---|---|
| shanten block: exactly one all-ones row | 0 |
| normalized shanten == n/6 | 0 |
| hand counts integer, each in [0,4] | 0 |
| aka / furiten / waits planes binary | 0 |
| own shanten (obs planes [862,869)) == `take_shantens()` | 0 |
| concealed hand size == `13 − 3·fuuro − 4·ankan` (+0/+1) from the public obs planes | **0** (out of 787,017) |
| informational: `waits ≠ ∅ ⇔ shanten == 0` | 225 (0.029%; known Rust-side semantics; not a label source) |

The hand-size-vs-public-meld check ties every decoded opponent block to the correct relative
seat, and the own-shanten check independently validates the one-hot shanten encoding
semantics. **No degradation was needed; `label_source = "opponents (oracle decode verified)"`.**

## Results

Pre-registered run (`results.json`); val = 52,729 held-out points:

| label | prevalence (val) | AUC | ECE | Brier | random-label AUC |
|---|---|---|---|---|---|
| **`opp_any_tenpai` (primary)** | 0.348 | **0.9334** | 0.0099 | 0.0943 | **0.5007** (3 seeds) |
| `opp_any_shanten_le1` | 0.687 | 0.8982 | 0.0153 | 0.1180 | 0.4995 |
| `opp_any_shanten_le2` | 0.904 | 0.9101 | 0.0222 | 0.0604 | 0.5058 |
| `opp_any_aka_in_hand` | 0.776 | 0.6055 | 0.0138 | 0.1699 | 0.5006 |
| `ctrl_opp_any_meld_visible` (positive control) | 0.425 | 0.7947 | 0.0188 | 0.1821 | 0.4973 |

Supplementary (post-hoc, not pre-registered; same sample/split; `supplementary.json`):

| probe | features | AUC | ECE |
|---|---|---|---|
| primary label | 24 public features | **0.9399** | 0.0181 |
| primary label | `Mish(phi)` (what the DQN head consumes) | 0.9221 | 0.0139 |
| primary label | `phi` (reproduction) | 0.9334 | 0.0099 |
| aka label | 24 public features | **0.6158** | 0.0177 |
| aka label | `phi` | 0.6055 | 0.0138 |
| meld control | raw meld planes (exact determinant) | **1.0000** | 0.0258 |
| meld control | `Mish(phi)` | 0.7725 | 0.0236 |

## Interpretation

**Positive reading (what the embedding does).** `phi` linearly predicts "an opponent is
tenpai" with AUC 0.933 and impressively low ECE (0.010); it also weakly predicts a genuinely
hidden fact, opponents' concealed red fives (0.606, ≈35 SE above chance). So the network's
representation *does* carry opponent-relevant structure.

**Negative reading (what the controls show, and why it dominates).**

1. The primary label is **largely a public-information readout**: 24 plainly public features
   score *higher* (0.9399) than the 1024-d embedding (0.9334). The strong AUC is therefore
   not evidence of a learned opponent model; it is mostly game-phase + riichi + own-shanten
   information that is visible to every player.
2. The same holds for the hidden aka target (public 0.6158 ≥ phi 0.6055).
3. The positive control shows the probe on `phi` is not even able to linearly recover a
   fully visible fact at 0.90: "some opponent has a meld" scores 0.795 from `phi`, while the
   six raw meld planes that exactly determine the label score 1.0000. The frozen encoder
   compresses/rotates visible information such that a linear readout loses some of it —
   so `phi`-probe AUCs should be read as a *lower bound*, and their comparison against a
   direct public baseline is the informative statement.
4. **Conclusion: A2 does not support opening the belief-head line. The embedding does not
   demonstrably encode opponent hidden state beyond public information.**

**Pre-registration honesty.** PREREG §5 made the positive control (≥0.90) part of
"pipeline valid"; it failed (0.7947), so the machine-written verdict in `results.json` is
`pipeline_valid=false`, `gate_belief=block` and was **not** rewritten after the fact. The
post-hoc supplementary evidence (random-label controls ≈ 0.5 for every label; exact-determinant
label → AUC 1.0000; primary AUC reproduced bit-for-bit) indicates the *measurement path* is
sound, i.e. the prereg's ≥0.90 prior about the encoder was miscalibrated. Both readings
nevertheless converge on `block` for the belief-head gate, because the substantive answer is
negative regardless of how the pipeline-validity technicality is resolved.

## Limitations / honest caveats

- **Linear probes only.** A non-linear readout could extract more; A2 cannot rule that out
  (Mish(phi) with a linear probe scored 0.922, still ≤ public baseline).
- The public baseline is small (24 features) and untuned; it is a lower bound on public
  predictability, which makes the "public ≥ embedding" statement conservative.
- One checkpoint (step 175,200), one obs version (v4), non-oracle inference path.
- Labels are outcomes of `PlayerState::shanten()`/hand state at the same decision point;
  the probe is correlational, not causal.
- Prevalence varies little by year (0.33–0.41 for the primary label; see supplementary.json),
  so the sample mix is not driving the numbers; 2009 games are absent from the sample.
- `waits ≠ ∅ ⇔ shanten == 0` has 0.029% Rust-side exceptions (informational only).

## Reproduction

```bash
source scripts/activate_env.sh
python scripts/exp_a2_probe.py                 # ~13 min, writes runs/exp/A2/results.json
python scripts/exp_a2_probe.py --selftest      # decode + metric unit checks, no products
python scripts/exp_a2_supp.py                  # post-hoc supplementary, ~8 min, writes supplementary.json
```
