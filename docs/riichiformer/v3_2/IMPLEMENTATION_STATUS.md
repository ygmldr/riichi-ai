# V3.2 remote implementation status

- Branch: `rf/v3.2-policy-parallel-plan` based on `riichiformer-v3-filtered`.
- **Implemented:** isolated pure policy scaffolding `riichiformer/policy/research_policy.py` and tests. External source status retained independent of operator-private routing. No old H-03 source registry row was changed.
- **Not yet integrated:** `data/split.py`, `data/shard.py`, `data/loader.py`, `scripts/rf_build_shards.py`, `scripts/rf_train.py` all still require the old ALLOWED-based contract. Thus production G3 remains **BLOCKED_RIGHTS**.
- **No user-machine tests have been run on this branch**; local standalone scaffold under the handoff pack passed 8/8 but this remotely uploaded code has not been executed here.
- Scope: private-research policy is an operator choice, **not** proof of third-party permission or redistribution rights.
- Before merging: implement PR-37 integration and shard-v3 schema, run legacy and negative suite, verify no source status was misrepresented; then develop PR-38/39/40/41/42 according to continuation roadmap.
