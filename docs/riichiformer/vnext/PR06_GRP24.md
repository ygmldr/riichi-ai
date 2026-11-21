# PR-06 — Independent 24-way GRP

**Code**: `riichiformer/vnext/grp.py`  
**Tests**: `tests/rf_test_vnext_grp.py` (7 passing, CPU)

## Contract

* Input is **public kyoku-start state**: `grand_kyoku, honba, kyotaku,
  score0..score3` (7 values); `scores_abs` sum with `kyotaku*1000` to 100,000.
* Model is `GRU(7 -> 64, 2 layers) -> Linear(64 -> 24)`, independent of the
  Actor; no Actor import.
* Output is 24 joint final-rank permutations in the fixed lexicographic table
  `GRP_PERMUTATIONS`.
* `rank_marginals` projects to `[N,4 seats,4 ranks]`; each seat row sums to 1
  and each rank column across seats sums to 1.
* `phi_from_marginals` with a zero-sum PT table satisfies `sum(phi)=0` in all
  tested rows.
* `grp_loss` uses the inverse permutation index; target labels are final ranks
  only.
* `calibration_report` returns 24-way NLL/Brier/ECE.
* `grp_state_sha` hashes parameter names, shapes and bytes; save/load is stable.

## Tests

* 24 unique permutations and inverse `permutation_index`;
* marginal row/column normalization and zero-sum `phi`;
* exact one-hot PT projection (75/-105);
* true permutation gets near-zero CE, wrong permutation does not;
* perfect calibration gives NLL/Brier/ECE ≈ 0;
* seat-permutation transform of the probability vector;
* state hash stable across `state_dict` roundtrip;
* independent `rf-grp-checkpoint/1` save/load with config/state/sidecar hashes.

## NOT_RUN

* Training/fitting GRP on the real Train Split;
* game-count / score-gap / final-scenario calibration on real histories;
* frozen GRP checkpoint registration and on-policy drift monitoring;
* MLP control and a large GPU fit.

The group is intentionally **not** declared better than any alternative.  A
real fit needs user authorization and a measured data/Census budget.
