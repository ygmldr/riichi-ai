# PR-07 — Decision Critic and gradient isolation

**Code**: `riichiformer/vnext/critic.py`; `VNextPolicy` optionally emits
`rank_logits` and `VNextPolicyOutput.rank_logits`.  
**Tests**: `tests/rf_test_vnext_critic.py` (4 passing, CPU)

## Contract

* `DecisionRankCritic(z) -> [N,4]` predicts the current seat's final rank.
* `critic_ce` is weighted CE over four rank classes; target ranks are 0..3.
* `critic_features(z, "detach_actor")` returns `z.detach()`, so a Critic-only
  backward cannot put any gradient on the Actor trunk.
* `critic_features(z, "joint")` leaves the graph connected.
* `expected_pt` projects rank probabilities with the frozen 5D table.

## Tests

* A synthetic `nn.Linear` trunk plus Critic: `detach_actor` gives trunk gradient
  norm 0 while the Critic still learns; `joint` gives a non-zero trunk gradient.
* Invalid gradient mode fails closed.
* Weighted CE and expected PT (75 / -105) are correct.

## NOT_RUN

* Real BC rank calibration/ECE on held-out games.
* On-policy Critic recalibration before RL.
* Joint-vs-detach quality comparison at matched budget.
* `GRP + residual` research variant.

The Critic is not claimed to improve Actor CE or PT.
