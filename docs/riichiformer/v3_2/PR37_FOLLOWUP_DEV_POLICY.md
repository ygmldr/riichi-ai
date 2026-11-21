# PR-37 follow-up — Dev-POLICY implementation record

Date: 2026-11-21 (Asia/Shanghai). TASK_ID: `PR-37-FOLLOWUP`.
Branch: `rf/v32-policy`. BASE_SHA: `cbf166a356f83c739b31f0b5a0b241b954ef9e0a`.
Status: implemented + static/CPU tests pass; GPU training and full-corpus builds were
**not** run (Lead owns the GPU). This record closes the three P1 findings in
`ACCEPTANCE_LEAD.md`. It does **not** claim any third-party licence, model strength, or
publication right.

## Scope and ownership

Files touched (all within the Dev-POLICY ownership list):

| File | Change |
|---|---|
| `riichiformer/policy/config.py` (new) | typed `PolicyConfig`, strict YAML/JSON loader, canonical `policy_hash` |
| `riichiformer/policy/research_policy.py` | `ResearchPolicy` backed by `PolicyConfig`; `PolicyDecision` contract extended; `decide()` matrix |
| `riichiformer/policy/__init__.py` | policy exports |
| `riichiformer/data/shard.py` | `policy_version`, `external_license_status`, `train_eligible_under_policy` metadata; private-train eligibility gate |
| `riichiformer/data/loader.py` | `expected_policy_hash` + `allowed_local_sources`; private train fails closed without an authenticated policy |
| `scripts/rf_build_shards.py` | strict `licensed_only` default; `--policy-config`; allowlist coverage; policy version/hash in metadata |
| `scripts/rf_train.py` | `--policy-config` / `--expected-policy-hash` / `--policy`; policy warning; policy fields in checkpoints; resume provenance unchanged |
| `scripts/rf_check_policy.py` (new) | one decision as JSON; explicit exit codes 0/1/2 |
| `configs/riichiformer/local_research_policy.example.yaml` (new) | committable example (no personal data, no secrets) |
| `tests/rf_test_research_policy.py` | 49 tests (config strictness, hash, matrix, serialisation) |
| `tests/rf_test_shards.py` | private fixtures rebuilt around a real config; builder/loader gate tests |
| `tests/rf_test_training.py` | private-shard CLI + resume provenance tests |
| `tests/rf_test_check_policy.py` (new) | CLI exit-code / JSON tests |

`configs/riichiformer/source_rights.csv` was **not** modified (11 rows, 0 ALLOWED).
No vendor pointer, model, encode, `train/bc.py`, remote branch, or existing tag changed.

## Typed policy config

```yaml
policy_version: rf-local-policy/1
profile: operator_private_research            # licensed_only | operator_private_research
scope: local_nonpublic_only                    # private requires this exact scope
operator_reference: example-user-authored-local-research-policy
allowed_local_sources: [tenhou_old, ...]       # explicit allowlist
release_allowed: false                         # must be false for private
network_upload_allowed: false                  # must be false for private
```

`PolicyConfig.from_mapping` / `load_policy_config` reject unknown fields, missing fields,
duplicate YAML/JSON keys, wrong types, unknown profiles/scopes, non-list allowlists,
duplicate/blank source ids, and private profiles with release/upload enabled. Errors are
`PolicyConfigError` with stable codes (`POLICY_UNKNOWN_FIELD`, `POLICY_MISSING_FIELD`,
`POLICY_PRIVATE_RELEASE_FORBIDDEN`, `POLICY_INVALID_ALLOWLIST`, ...).

`policy_hash = sha256(canonical_json(payload))` where the payload is exactly
`policy_version, profile, scope, operator_reference, allowed_local_sources(sorted),
release_allowed, network_upload_allowed`. Key order and allowlist order do not matter;
any semantic change does. `ResearchPolicy.policy_hash` delegates to the same function,
so legacy-style direct construction and config-backed construction agree.

Example config hash:
`72ac377f2fa44297c18228aeeba59160de48f67c65bc3b3b242a3e9b889251b0`.

## Decision contract

`PolicyDecision` now carries `selected_profile`, `scope`, `evidence_ref`, `reason`, and
a `source_rights_status` alias, in addition to `source_id`, `action`,
`operational_allow`, `external_license_status`, `basis`, `policy_hash`.
`to_dict()`/`to_json()` are stable. Actions: `audit`, `compile`, `synthetic_train`,
`private_train`, `distribute`.

The external status is copied verbatim and never rewritten to `ALLOWED`. The private
training branch keeps the required basis string
`operator_attestation_not_external_license`; the reason text says explicitly that the
attestation is not an external licence. `operator_private_research` denies `distribute`
even for a row whose external status is `ALLOWED`.

## Shard / loader / trainer gate

* Builder default is `licensed_only` (the former fail-open default is removed).
  `operator_private_research` requires `--policy-config`; the config profile must match
  `--research-policy`; every selected source must be in `allowed_local_sources` or the
  build exits 2 before writing anything.
* `rf-shard/3` keeps all previous metadata fields and adds `policy_version`,
  `external_license_status` (normalised to `rights_status`, and required to match it),
  and `train_eligible_under_policy`. `rights_status`/`train_allowed` remain the real
  external facts. Private train metadata requires eligibility; licensed train metadata
  keeps the old `ALLOWED` gate.
* `ShardReader(purpose="train")` on a private shard now requires
  `expected_policy_hash` (from `--policy-config` or `--expected-policy-hash`) and
  compares it to `research_policy_hash`; when an allowlist is supplied it also checks
  every metadata source against it. Missing authentication, wrong hash, wrong
  eligibility flag, or an uncovered source raises `ShardRightsError` before any row is
  yielded.
* `rf_train.py` accepts `--policy-config`, `--expected-policy-hash`, `--policy`; logs
  `external_license_status` + the operator-attestation warning; stores policy fields in
  the checkpoint hashes. Checkpoint resume still compares the same provenance keys
  (`split_manifest_sha256`, `source_id`, `research_policy`, `research_policy_hash`,
  `rights_status`), so a checkpoint from a different policy config is refused with
  exit 1.

## Real H-02 evidence (CPU, read-only over the main checkout)

Commands were run from this worktree with `PYTHONPATH` pointing at the pinned vendor
release and `MAIN=/home/ygsldr/riichi-ai`:

| Command | Result |
|---|---|
| `rf_build_shards.py ... --purpose train --limit 10 --research-policy operator_private_research --policy-config configs/riichiformer/local_research_policy.example.yaml --dry-run` | exit 0, `rows=10 sources=tenhou_old rights=UNKNOWN train_allowed=False policy_version=rf-local-policy/1 policy_hash=72ac377f... train_eligible_under_policy=True` |
| same without `--policy-config` | exit 2, `operator_private_research requires an explicit --policy-config` |
| same with default `licensed_only` | exit 2, `training shard selected blocked row ... train_allowed=False` |
| `--write` (10 games, `/tmp/rf37_private10.rfshard`) | exit 0, `games=10 decisions=8504 written=10 skipped=0`, sha256 `be76e8a6dca8b823a99407e0f32f18faa376c94af6daedeb11c8ad6bef6fe683` |
| `ShardReader(path, purpose="train")` without expected policy | `ShardRightsError`: refusing an unauthenticated private shard |
| `ShardReader(..., expected_policy_hash=config.hash, allowed_local_sources=config.sources)` | 10 games / 8504 decisions, `external_license_status=UNKNOWN`, `train_eligible_under_policy=True` |
| `ShardReader(..., expected_policy_hash="e"*64)` | `ShardRightsError`: `research_policy_hash mismatch` |
| `rf_train.py --dry-run --policy-config ...` | exit 0, `PR14_POLICY_NOTE ... basis=operator_attestation_not_external_license external_license_status=UNKNOWN` |
| `rf_train.py --dry-run` (no config) | exit 2, private shard requires authentication |

New artifact hashes (this run): shard `be76e8a6…`, content `e1d7df09…`,
schema `12503d36…`, action `6cf1efc5…`, normalizer `37df861b…`,
policy config `72ac377f…`. These are CPU data-build artifacts only; no model was
trained from them here.

## Tests actually run

```bash
export PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release
.venv/bin/python -m pytest -q tests/rf_test_research_policy.py tests/rf_test_shards.py \
  tests/rf_test_training.py tests/rf_test_source_rights.py
# 151 passed, 1 warning, 13 subtests passed — exit 0

.venv/bin/python -m pytest -q tests/rf_test_research_policy.py tests/rf_test_shards.py \
  tests/rf_test_training.py tests/rf_test_source_rights.py tests/rf_test_check_policy.py
# 167 passed, 1 warning, 13 subtests passed — exit 0

.venv/bin/python -m pytest -q $(git ls-files 'tests/rf_test_*.py' | tr '\n' ' ')
# 576 passed, 2 skipped, 1 warning, 242 subtests passed — exit 0
```

The two skips are the pre-existing board-injection skips in
`tests/rf_test_ruleset_parity.py`. The one warning is the pre-existing
`float(result.loss)` warning in `rf_train.py`; it is cosmetic, not a test failure.

## P1 closure mapping

| Lead P1 | Status | Evidence |
|---|---|---|
| 1. private hash not tied to a policy config | **closed** | `PolicyConfig` canonical hash; `ShardReader` requires `expected_policy_hash`, compares it, and checks the source allowlist; wrong/missing hash tests |
| 2. builder default is permissive | **closed** | `--research-policy` defaults to `licensed_only`; private without `--policy-config` exits 2; parser-default test |
| 3. policy hash derivation is builder-local | **closed** | hash derives from the strict config payload with a committed example and a known-vector test; no hardcoded operator reference remains in the builder |

## Residual risks / P2 notes

* `--expected-policy-hash` without `--policy-config` authenticates the exact policy hash
  but has no allowlist to check. It is a hash-pinned mode for callers that already know
  the exact policy; integration should prefer `--policy-config`.
* Pre-existing private train shards (including the Lead's `/tmp/rf32_private10.rfshard`)
  are intentionally not loadable: they lack `train_eligible_under_policy` and their old
  builder-local hash does not match any config hash. Rebuild them from the same split
  with `--policy-config`.
* Shard/manifest files are not cryptographically signed; the gate authenticates the
  policy hash, not the producer. A writer with filesystem access could still rewrite
  both the DB and manifest. The threat model remains the trusted local operator.
* A cosmetic pre-existing `float(tensor)` warning remains in `rf_train.py`.

## NOT_RUN

* GPU training / evaluation / checkpoint promotion (Lead-exclusive).
* Full-corpus shard build; only the 10-game real H-02 dry-run and write above.
* Migration of pre-existing private shards or checkpoints.
* Any `git push`, merge, tag, vendor update, or edits to other worktrees.
* Any download, upload, cloud training, or online platform play.

## Next gate

Lead/Reviewer: review this branch, rebuild the private shard with an operator config,
re-run the full suite from the committed tree, record the new policy hash in the PR-43
gate, and only then merge PR-37 ahead of PR-38/39. No claim of external permission or
model strength is made by this document.
