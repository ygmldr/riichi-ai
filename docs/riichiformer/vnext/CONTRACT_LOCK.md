# VNext Contract Lock (PR-02)

**Status**: FROZEN for PR-03..PR-11 development
**Owner**: single Lead
**Base**: `rf3.3-w2` / `8a915d1`
**Contract module**: `riichiformer/vnext/contracts.py`
**Manifest hash**: `6ce138249a06173e02804aea7e81e6c4b95cdaa22b57a03984c172eab0883e58`
**Model architecture hash**: `012f037139346913433c5129f6720f70e7708618a31af75fcd206f05d719f5df`

## 1. Frozen identifiers

| Field | Value |
|---|---|
| `model_arch_id` | `rf-vnext-crossattn/1` |
| `encoder_contract_id` | `current_kyoku_bos/1` |
| `state_contract_id` | `tile34_global/1` |
| `grp_contract_id` | `grp24-joint/1` |
| `rl_contract_id` | `rf-rl-row/1` |
| native action space | 46 (unchanged) |
| native action map hash | `6cf1efc5ba99533ec4844f38184ef51a7941174164f9588f21d33ca70a873a2e` |
| schema hash | `12503d36e2571449d4e4bdd4c8780e00f285d3a2ec81f9e26af184d24c8ef43c` |
| normalizer hash | `37df861b08b3958c3030c53dcbfa27762892c41332b383060caf46788f9ad444` |

The model default is `6L / d256 / 4 heads / ffn 704`, 1 cross-attention layer,
factorized tile embedding, tile-scored 37 + other 9 head, and 4-class Critic
with `critic_grad_mode ∈ {joint, detach_actor}` (default `detach_actor`).
`cross_attn_layers=0`, `head_type=linear46`, `tile_embedding=identity` and
`model_mode=state_only` are explicit control configurations.

## 2. Forward-visible contracts (no labels)

`VNextVisibleState`:

* `tile_features: [34, 26]` — registry in `contracts.TILE_FEATURE_NAMES`;
* `global_features: [38]` — registry in `GLOBAL_FEATURE_NAMES`;
* `candidate_features: [37, 4]` — normal candidates 0..33, aka candidates 34..36
  for 5m/5p/5s, with a distinct `is_aka_candidate` and base-kind column;
* `legal_mask: bool[46]`, phase-aware; at least one legal action.

`VNextModelBatch`:

* event fields `[B,L]`: type id (BOS=16, PAD=17, 0..15 schema events), actor,
  tile kind, aka, tsumogiri, kyoku position, valid-prefix mask;
* event meld fields `[B,L,3]`: public consumed tile kinds and aka flags
  (UNKNOWN padding), so chi/pon/daiminkan/kakan public tiles are not collapsed;
* decision fields `[N]`: `decision_to_sequence`, inclusive
  `prefix_token_offsets`, `phases`;
* state fields `[N,...]`: `x_tile`, `x_global`, `candidate_features`,
  `legal_mask`;
* **no target/rank/oracle/future field**; targets are a separate
  `VNextTargets` object.

`VNextPolicyOutput`: raw `logits [N,46]`, `masked_logits` with `-inf` on
illegal actions, normalized `logprobs`, `legal_mask`, optional
`rank_logits [N,4]`.

## 3. Current-kyoku/BOS rule

* `encode_stream` drops `start_game` and earlier kyoku events, starts with
  `BOS_EVENT_ID=16`, and appends one token per visible event of the current
  kyoku.
* For `p` current-kyoku events, `prefix_token_offset = p` (BOS is token 0).
* Cross-attention keys are exactly `0 <= token <= offset` for that decision's
  sequence; no key from another sequence or a later token may be used.
* `SELF_TURN` and `KAN_SELECTION` decisions may share one prefix; their
  decision IDs/order and phase embeddings remain distinct.
* Multi-kan action 42 is **not** decoded as `kan_options[0]`; the runtime must
  issue a real `KAN_SELECTION` decision (tile action 0..36).

## 4. GRP and PT

* 24 fixed lexicographic permutations in `GRP_PERMUTATIONS`;
  `permutation_index` is the only label encoder.
* `GRP24` is a separate module: `GRU(7 -> 64, 2 layers) -> 24`, no Actor import.
* `rank_marginals` returns `[N,4 seats,4 ranks]`; each seat sums to 1 and each
  rank sums to 1 over seats.
* `phi_from_marginals` with a zero-sum PT table must satisfy `sum(phi)=0`.
* PT 5D is `[+75,+30,0,-105]`; PT 7D is `[+90,+45,0,-135]`; IDs are
  `pt5d/75,30,0,-105` and `pt7d/90,45,0,-135`.
* 1v3 identity: `relativePT = RF_PT - mean(3 opponents) = (4/3) RF_PT`.

## 5. RL behavior and reward

* `GuardSpec` records temperature/top-p/epsilon/forced/effective mask and a
  guard version.
* `behavior_probs` returns the **actual executed** distribution; illegal actions
  have zero mass, a forced action has probability 1, and the stored
  `behavior_logprob` must equal `log(mu[executed])`.
* `RLDecisionRow` separates `legal_mask`, `effective_mask`, raw logits, full
  behavior probabilities, executed action, guard reason and policy/GRP/rules
  SHAs.
* `mortal_delta_rewards` and `pbrs_rewards` are distinct reward records:
  neither may be combined with an extra terminal reward; the telescoping
  identity is `sum(rewards)=R_final-Phi_0` per seat.
* `two_timescale_gae` uses `lambda_inter` only at a kyoku boundary; terminal
  transitions do not bootstrap.

## 6. Ownership map

| Path | Owner | Rule |
|---|---|---|
| `riichiformer/schemas/action_v3.py`, `schema_v3.py` | Lead | native 46 order; no worker write |
| `riichiformer/vnext/contracts.py` | Lead | ABI changes need a new contract hash and RFC |
| `riichiformer/vnext/model.py`, `observable.py` | Model worker under Lead review | no `value/`/`rl/` edits |
| `riichiformer/vnext/grp.py`, `critic.py` | Value worker under Lead review | no Actor edits |
| `riichiformer/vnext/runtime.py`, `train.py`, `data.py` | Runtime/Model under Lead review | no shared contract edits |
| `riichiformer/vnext/rl.py` | RL worker under Lead review | no Actor/model edits |
| `configs/riichiformer/source_rights.csv`, seed registry | Lead | fail-closed rights; never rewrite ALLOWED |
| `docs/riichiformer/vnext/**` | Lead | evidence, not an implementation claim |

## 7. Checkpoint rule

A VNext checkpoint must store:

* `model_config` with `VNextConfig` and its `model_config_hash`;
* the three frozen schema hashes **and** `vnext_contract_hash`;
* source/split/provenance and counters;
* optimizer/scheduler/EMA/RNG for strict resume.

Old MVP-S checkpoints are not strict-loadable.  `load_vnext_checkpoint` refuses
a payload without `d_model`/`cross_attn_layers`, verifies the sidecar SHA and
optionally the VNext contract hash.

## 8. Sign-off

Signed by the single Lead on 2026-11-22.  Worker changes to any frozen field
require a new contract hash, a migration note, and a fresh reviewer pass.
