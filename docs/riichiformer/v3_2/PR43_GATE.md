# RiichiFormer V3.2 — PR-43 G3 gate (Wave 1 integrated)

Date: 2026-11-21 (Asia/Shanghai). Gate status is deliberately split; it is **not** a
single unconditional PASS.

| Component | Status |
|---|---|
| `CODE_PASS` | **PASS** — 682 passed, 2 skipped, 281 subtests in the final integrated tree |
| `SYNTHETIC_E2E_PASS` | **PASS** — synthetic/private H-02 10-game shard → 6-layer MVP-S GPU smoke → strict resume |
| `PRIVATE_H2_TRAIN_SMOKE` | **PASS** (operator-private profile, real H-02 rows, not a strength run) |
| `OWN_SELFPLAY_TRAIN_PASS` | **NOT_RUN** — selfplay/arena plumbing ran, POV → shard → trainer roundtrip not implemented |
| `THIRD_PARTY_SOURCE_STATUS` | **UNCHANGED** — 11 rows, 0 `ALLOWED`; private profile is an operator attestation, not a licence |
| `STRENGTH_NOT_RUN` | **NOT RUN** — no arena strength, dan/rank, Mortal comparison, or “九段/天凤位” claim |

## 1. Integrated baseline

| Item | Value |
|---|---|
| Branch | `rf/v32-integration` |
| HEAD | see `git rev-parse rf/v32-integration` (integration branch; includes the P1 fixes and this gate record) |
| Accepted base | `cbf166a` (`rf/v3.2-local`; remote `a27df099…`) |
| Merged lanes | A `87afe78`, C `8248496`, B `4e4754b`, D `4e380a3` |
| Python | `/home/ygsldr/riichi-ai/.venv/bin/python` (3.12, torch 2.11.0+cu128) |
| GPU | RTX 5070 Ti 16 GB |
| Vendor | pinned `0cff2b5`; local integration applied `patches/0014-rule-r02-r25-r26.patch` and used the patched `libriichi.so` built in `worktrees/v32-rules` |
| Schema/action/normalizer | `12503d36…` / `6cf1efc5…` / `37df861b…` |
| Policy config | `72ac377f2fa44297c18228aeeba59160de48f67c65bc3b3b242a3e9b889251b0` (`rf-local-policy/1`) |

The patch is tracked as `patches/0014-rule-r02-r25-r26.patch`; the integration worktree's
`vendor/mortal` source was **not committed** as a submodule pointer change. A fresh
checkout must apply the patch and rebuild `libriichi` before relying on R02/R25/R26.

## 2. CODE_PASS

Command (from `worktrees/v32-integration`):

```bash
PYTHONPATH=/home/ygsldr/riichi-ai/worktrees/v32-rules/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q \
  $(git ls-files 'tests/rf_test_*.py' | tr '\n' ' ')
```

Result: **682 passed, 2 skipped, 1 warning, 281 subtests passed**, 198.10 s, exit 0.
Log: `docs/riichiformer/v3/logs/V3_2_W1_full_suite.log`. The two skips are the
pre-existing `NativeBoardCapabilityTests` / board-injection capability gaps; they are
never counted as passes.

Independent reviews on the lane commits: PR-37 `PASS_WITH_P1` (P1 fixed in `c426ac7`),
PR-40 `PASS_WITH_P1` (evidence P1s fixed in `c426ac7`), PR-38/39 and PR-41/42 under
separate review. No P0 was reported for any lane.

## 3. SYNTHETIC_E2E_PASS / PRIVATE_H2_TRAIN_SMOKE

Real H-02 canonical/split data, protocol L, 10 train rows:

- `rf-shard/3`, 10 games / **8,504 decisions**, 12 MB.
- shard sha256 `18a776f232c01590121d0dae5904f880d92c34fc7652ade82b6868815c19eb83`
- content sha256 `e1d7df0988a04af91eba8eacb9c8a5b7865acd42358cea2b0299ebc8f697d5e1`
- `rights_status=UNKNOWN`, `external_license_status=UNKNOWN`, `train_allowed=false`
  (the original facts are preserved), `train_eligible_under_policy=true`,
  `research_policy=operator_private_research`, `policy_version=rf-local-policy/1`.
- `licensed_only` build of the same rows still exits 2; a private shard without an
  explicit policy config/hash is refused by `ShardReader`.

GPU smoke (default six-layer MVP-S, not tiny):

```bash
.venv/bin/python scripts/rf_train.py --shard /tmp/rf43_private10.rfshard --train \
  --out /tmp/rf43_private10-smoke.pt --device cuda --max-steps 2 --batch-size 4 \
  --policy-config configs/riichiformer/local_research_policy.example.yaml
```

Observed: `PR14_MODEL layers=6 ... parameters=5217582`, `step=1 loss=2.286412`,
`step=2 loss=1.477174`, `tokens=204 decisions=8`, checkpoint sha
`559b98aa109a4e51f4983e1a11e4bc826bba6b2b2e5f0815c31153e64c32c5d5`, 17.58 s wall,
max RSS 2.02 GB. Strict resume to step 3: `loss=0.486104`, checkpoint sha
`0a91dcc013e2b44558f5c278a7024fd0283504595c179effeceeb90813ce0b1f`.

Strict resume refuses a protocol-R shard: exit **1**
(`checkpoint provenance mismatch for split_manifest_sha256`), no output checkpoint.
`--init-weights-only` was exercised separately: it loads only model weights, starts
step 0, and does not restore optimizer/RNG.

## 4. OWN_SELFPLAY_TRAIN_PASS = NOT_RUN

What did run (CPU, local, no network):

- `scripts/rf_selfplay.py` with `libriichi_tenhou_approx/1`, seed 1000, 4 games
  (4 seat rotations): 3,141 decisions, **0 illegal / 0 rejected / 0 fallbacks**,
  4/4 score sums = 100000; summary labels `tenhou_equivalent=false`,
  `strength_claim=none`.
- `scripts/rf_arena.py` (1v3, 4 games, seed 5000, 200 resamples): `GameAudit.ok=true`,
  no violation; R02/R25/R26 deviations 0; R29 proxy 1 (explicitly unverified).
  A single seed block correctly reports `block bootstrap needs at least 2 seed blocks`
  instead of inventing a CI.
- The raw engine game logs are full-information (four haipai and ura markers); the
  summary labels them as provenance artifacts, **not POV-safe training rows**.

Why the pass is NOT_RUN: there is no implemented POV → shard → `ShardReader`/`rf_train`
conversion for this selfplay output yet. The 10-game H-02 shard above is the private
human-log smoke, not an own-selfplay train pass. No loss/strength claim is made for
selfplay.

## 5. THIRD_PARTY_SOURCE_STATUS

`configs/riichiformer/source_rights.csv` is unchanged: **11 rows, 0 `ALLOWED`**.
The private profile is an operator research choice; `external_license_status` remains
`UNKNOWN`/`RESTRICTED` and `rf_check_policy.py` still DENIES `distribute`. No data,
weights, or derived samples were published or uploaded; no online Tenhou/雀魂 AI play.
This gate is **not** a statement that third-party training rights were obtained.

## 6. STRENGTH_NOT_RUN

No model-vs-bot arena, no official dan/rank estimate, no Mortal/akochan comparison, and
no “九段/天凤位” claim. BC loss decreasing on 8 decisions is a plumbing signal, not
strength. PR-44 B0 and the later candidate/risk/RL lines remain future work.

## 7. Remaining P1/P2 (do not hide)

P1:
1. R29 ippatsu timing remains **UNVERIFIED**; R02 continuation (honba/dealer/kyotaku) is
   source-read, not executed by the native settlement fixture.
2. PR-38 multi-decision forward is not yet wired into `data/loader.py`/`rf_train.py`;
   it has CPU parity/benchmark tests only. CUDA parity/VRAM was not measured.
3. Own-selfplay POV → shard → trainer roundtrip is not implemented (hence NOT_RUN).
4. The 0014 rules patch must be applied and `libriichi` rebuilt in a fresh checkout;
   the integration worktree's local vendor state is not committed.

P2:
- Private-policy authentication covers `purpose=train`; audit/diagnostic readers do not
  require the policy hash (documented; read-only path).
- Sidecar metadata is not signed and `content_sha256` is not cross-checked against the
  DB; the local-operator threat model is trusted.
- `--expected-policy-hash` without a config pins the hash but cannot check the source
  allowlist; use `--policy-config` for the full check.
- R02/R25/R26 labels are now `tenhou-parity-verified` only within the stated settlement
  branch; keep `tenhou_approx` for arena/strength reports until R29 and continuation
  are executable.

## 8. Next gate

1. Independent review of PR-38/39 and PR-41/42 integration.
2. Wire `MultiDecisionBatch` into the loader/trainer behind a versioned interface
   (Lead RFC), then rerun CUDA parity with flash-SDP disabled.
3. Implement POV → shard conversion for own selfplay (POV-safe only; never use the
   full-information raw logs as training rows), then rerun PR-43 selfplay smoke.
4. Apply 0014 to the main vendor and rebuild before any arena/rules claim.
5. Only then start PR-44 B0 with a preregistered data/config/seed/policy and a Lead GPU
   lock. Do not reuse confirm seeds.
