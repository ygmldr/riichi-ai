#!/usr/bin/env python3
"""PR-37 policy decision CLI: one source + action + profile -> PolicyDecision.

Reads the static source-rights registry as *facts* and the operator policy
config as a *local routing choice*.  It never writes, downloads, trains, or
changes a rights row.  ``operator_private_research`` can permit local audit or
private training for allowlisted sources, but it can never authorize
``distribute``: the external status is reported separately and stays unchanged.

Exit codes
----------
0  operational allow (this project's local run chain may proceed)
1  operational deny (the policy decided DENY; the external fact is still shown)
2  invalid input (malformed config/registry, unknown source/action/profile)

The distinction matters: a DENY is a valid policy answer (exit 1), while an
unreadable policy file or unknown source is an error (exit 2), never an
implicit allow.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.policy.config import (  # noqa: E402
    PROFILES, PolicyConfigError, load_policy_config,
)
from riichiformer.policy.research_policy import (  # noqa: E402
    ACTIONS, ResearchPolicy, ResearchPolicyError, decide,
)

DEFAULT_RIGHTS = REPO_ROOT / "configs/riichiformer/source_rights.csv"

EXIT_ALLOWED = 0
EXIT_DENIED = 1
EXIT_INVALID = 2

REQUIRED_COLUMNS = ("source_id", "status")
PRIVATE_PROFILE = "operator_private_research"


class PolicyCheckError(Exception):
    """Unusable input for the policy check; exit 2."""


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def load_registry(path: str | Path) -> dict[str, dict]:
    """Read the source-rights registry as immutable facts (read-only)."""
    registry_path = Path(path)
    if not registry_path.is_file():
        raise PolicyCheckError(f"rights registry not found: {registry_path}")
    try:
        with registry_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise PolicyCheckError("rights registry has no header row")
            missing = [name for name in REQUIRED_COLUMNS
                       if name not in reader.fieldnames]
            if missing:
                raise PolicyCheckError(
                    f"rights registry missing required column(s): {missing}")
            rows: dict[str, dict] = {}
            for lineno, raw in enumerate(reader, start=2):
                source_id = _text(raw.get("source_id"))
                if not source_id:
                    raise PolicyCheckError(
                        f"rights registry line {lineno}: empty source_id")
                if source_id in rows:
                    raise PolicyCheckError(
                        f"rights registry duplicate source_id {source_id!r}")
                rows[source_id] = {
                    key: _text(value) for key, value in raw.items() if key is not None
                }
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise PolicyCheckError(f"cannot read rights registry {registry_path}: {exc}") from exc
    if not rows:
        raise PolicyCheckError(f"rights registry has no data rows: {registry_path}")
    return rows


def build_policy(args) -> tuple[ResearchPolicy, Path | None]:
    config_path = Path(args.config) if args.config else None
    profile = args.profile
    if config_path is not None:
        config = load_policy_config(config_path)
        if profile is not None and profile != config.profile:
            raise ResearchPolicyError(
                f"--profile {profile!r} does not match policy config profile "
                f"{config.profile!r}", code="POLICY_PROFILE_MISMATCH")
        return config.to_research_policy(), config_path
    if profile is None:
        raise PolicyCheckError(
            "provide --profile or --config; a policy decision needs an explicit "
            "profile")
    if profile not in PROFILES:
        raise PolicyCheckError(f"unknown profile {profile!r}; choose one of {PROFILES}")
    # No config means an empty allowlist: private decisions deny every source
    # (fail closed) instead of inventing an operator attestation.
    policy = ResearchPolicy(
        profile=profile,
        local_source_allowlist=frozenset(),
        operator_reference="command-line-profile-without-policy-config",
        scope="local_nonpublic_only",
    )
    return policy, None


def check(args, registry: dict, policy: ResearchPolicy):
    return decide(registry, policy, args.source, args.action,
                  evidence_ref=getattr(args, "evidence_ref", None))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--action", required=True, choices=ACTIONS)
    parser.add_argument("--profile", default=None, choices=PROFILES)
    parser.add_argument("--config", default=None,
                        help="strict YAML/JSON policy config (optional for licensed_only, "
                             "recommended for operator_private_research)")
    parser.add_argument("--source-rights", default=str(DEFAULT_RIGHTS),
                        help="read-only source-rights CSV providing external facts")
    parser.add_argument("--evidence-ref", default=None,
                        help="optional explicit evidence/provenance reference")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy, config_path = build_policy(args)
        registry = load_registry(args.source_rights)
        decision = check(args, registry, policy)
    except (PolicyConfigError, ResearchPolicyError, PolicyCheckError) as exc:
        print(f"PR37_INVALID {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INVALID
    payload = decision.to_dict()
    payload["config_path"] = str(config_path) if config_path is not None else None
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        verdict = "ALLOW" if decision.operational_allow else "DENY"
        print(
            f"{verdict} source={decision.source_id} action={decision.action} "
            f"profile={decision.selected_profile} "
            f"external_license_status={decision.external_license_status} "
            f"basis={decision.basis} policy_hash={decision.policy_hash}")
        print(f"reason: {decision.reason}")
    return EXIT_ALLOWED if decision.operational_allow else EXIT_DENIED


if __name__ == "__main__":
    sys.exit(main())
