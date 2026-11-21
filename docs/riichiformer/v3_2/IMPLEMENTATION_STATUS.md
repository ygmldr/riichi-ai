# V3.2 remote implementation status

- Branch: `rf/v3.2-policy-parallel-plan` based on `riichiformer-v3-filtered`.
- **Implemented:** isolated pure policy scaffolding `riichiformer/policy/research_policy.py` and tests. External source status retained independent of operator-private routing. No old H-03 source registry row was changed.
- **Not yet integrated:** `data/split.py`, `data/shard.py`, `data/loader.py`, `scripts/rf_build_shards.py`, `scripts/rf_train.py` all still require the old ALLOWED-based contract. Thus production G3 remains **BLOCKED_RIGHTS**.
- **Lead acceptance 2026-11-21: PASS_WITH_P1** — the remote tree was checked out in
  `worktrees/v3.2-policy` from `origin/rf/v3.2-policy-parallel-plan`; full suite
  **494 passed, 2 skipped, 242 subtests**; real H-02 10-game private shard built
  (`rf-shard/3`, 8,504 decisions), six-layer 5,217,582-param smoke trained
  (loss 2.193→1.712), strict resume advanced to step 3, and split-mismatch resume was
  refused. Details and P1 findings: `ACCEPTANCE_LEAD.md`.
- Scope: private-research policy is an operator choice, **not** proof of third-party permission or redistribution rights.
- Remaining P1s: policy hash not authenticated against a policy config, permissive
  builder default, no `--policy-config`; these are PR-37 follow-up, not permission to
  rewrite source status.
- Before merging: close the PR-37 P1s and run the legacy/negative suite again; then
  develop PR-38/39/40/41/42 according to continuation roadmap.
