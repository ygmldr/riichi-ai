"""Pure project routing policy; never represents a third-party licence grant.

The routing policy answers one question: *may this project's run chain use this
source for this action under the operator's explicit research profile?*  It is
deliberately separate from the external source-rights registry.  A private
research decision is an operator attestation, **not** an external licence, and
:class:`PolicyDecision` therefore keeps the original ``external_license_status``
untouched and never rewrites it to ``ALLOWED``.

This module contains no I/O, no network access, and no implicit environment
variables.  Stable error codes are attached to :class:`ResearchPolicyError`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from .config import (
    POLICY_VERSION,
    PRIVATE_SCOPE,
    PROFILES,
    PolicyConfig,
    PolicyConfigError,
)

#: Actions understood by :func:`decide`.
ACTIONS = ("audit", "compile", "synthetic_train", "private_train", "distribute")

STATUS_VALUES = frozenset({"ALLOWED", "UNKNOWN", "RESTRICTED"})

#: Software code sources are governed by their own software licences; they are
#: never a training corpus.
CODE_SOURCES = frozenset({"tenhou_to_mjai_code", "mortal_code", "akochan_code"})

#: Locally generated data whose provenance is a generator/seed hash, not an HTTP
#: licence URL.  ``synthetic_*``/``selfplay_*`` identify first-party generators.
SELF_GENERATED_SOURCES = frozenset({"selfplay_own"})
SELF_GENERATED_PREFIXES = ("selfplay_", "synthetic_")

#: ``basis`` strings are part of the stable decision contract.
BASIS_VERIFIED_EXTERNAL = "verified_external"
BASIS_OPERATOR_ATTESTATION = "operator_attestation_not_external_license"
BASIS_SELF_GENERATED = "self_generated"
BASIS_LOCAL_AUDIT = "local_audit_only"
BASIS_CODE_POLICY = "code_use_policy"
BASIS_NOT_COVERED = "not_covered"
BASIS_NOT_TRAINING_DATA = "software_is_not_training_data"
BASIS_UNVERIFIED_EXTERNAL = "unverified_external"
BASIS_NOT_DISTRIBUTABLE = "not_authorized_for_distribution"

#: The public/redistribution path must never rely on an operator-private profile.
PRIVATE_PROFILE = "operator_private_research"
LICENSED_PROFILE = "licensed_only"


class ResearchPolicyError(ValueError):
    """Invalid policy input or decision request; fail closed with a stable code."""

    def __init__(self, message: str, *, code: str = "POLICY_INVALID"):
        super().__init__(message)
        self.code = code


def _self_generated(source_id: str) -> bool:
    return source_id in SELF_GENERATED_SOURCES or source_id.startswith(
        SELF_GENERATED_PREFIXES)


def _row_status(row: Mapping) -> str:
    status = str(row.get("status", "")).upper()
    if status not in STATUS_VALUES:
        raise ResearchPolicyError(
            f"invalid rights status {row.get('status')!r}; expected one of "
            f"{sorted(STATUS_VALUES)}", code="POLICY_INVALID_STATUS")
    return status


@dataclass(frozen=True)
class ResearchPolicy:
    """Profile + explicit local-source allowlist used by the run chain.

    Construct directly (as tests and callers historically did) or from a typed
    :class:`~riichiformer.policy.config.PolicyConfig`.  The policy hash is the
    config hash, so a shard/checkpoint can be authenticated against the exact
    operator policy that produced it.
    """

    profile: str
    local_source_allowlist: frozenset[str]
    operator_reference: str
    release_allowed: bool = False
    network_upload_allowed: bool = False
    scope: str = PRIVATE_SCOPE
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        try:
            config = PolicyConfig(
                profile=self.profile,
                operator_reference=self.operator_reference,
                allowed_local_sources=self.local_source_allowlist,
                scope=self.scope,
                release_allowed=self.release_allowed,
                network_upload_allowed=self.network_upload_allowed,
                policy_version=self.policy_version,
            )
        except PolicyConfigError as exc:
            raise ResearchPolicyError(str(exc), code=exc.code) from exc
        object.__setattr__(self, "profile", config.profile)
        object.__setattr__(self, "operator_reference", config.operator_reference)
        object.__setattr__(self, "local_source_allowlist",
                           config.allowed_local_sources)
        object.__setattr__(self, "scope", config.scope)
        object.__setattr__(self, "policy_version", config.policy_version)

    @classmethod
    def from_config(cls, config: PolicyConfig) -> "ResearchPolicy":
        if not isinstance(config, PolicyConfig):
            raise ResearchPolicyError(
                "from_config requires a PolicyConfig", code="POLICY_INVALID_CONFIG")
        return cls(
            profile=config.profile,
            local_source_allowlist=config.allowed_local_sources,
            operator_reference=config.operator_reference,
            release_allowed=config.release_allowed,
            network_upload_allowed=config.network_upload_allowed,
            scope=config.scope,
            policy_version=config.policy_version,
        )

    def to_config(self) -> PolicyConfig:
        return PolicyConfig(
            profile=self.profile,
            operator_reference=self.operator_reference,
            allowed_local_sources=self.local_source_allowlist,
            scope=self.scope,
            release_allowed=self.release_allowed,
            network_upload_allowed=self.network_upload_allowed,
            policy_version=self.policy_version,
        )

    @property
    def policy_hash(self) -> str:
        return self.to_config().policy_hash

    def covers(self, source_id: str) -> bool:
        return source_id in self.local_source_allowlist


@dataclass(frozen=True)
class PolicyDecision:
    """One explicit, serialisable routing decision.

    ``operational_allow`` only says whether this project's local run chain may
    proceed; it is never a statement that a third party granted a licence.
    ``external_license_status`` is copied verbatim from the rights registry.
    """

    source_id: str
    action: str
    operational_allow: bool
    external_license_status: str
    basis: str
    policy_hash: str
    selected_profile: str = ""
    scope: str = ""
    evidence_ref: str | None = None
    reason: str = ""

    @property
    def source_rights_status(self) -> str:
        """Spec-compatible alias; the fact is the external status, unchanged."""
        return self.external_license_status

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "action": self.action,
            "selected_profile": self.selected_profile,
            "scope": self.scope,
            "operational_allow": bool(self.operational_allow),
            "allow": bool(self.operational_allow),
            "external_license_status": self.external_license_status,
            "source_rights_status": self.source_rights_status,
            "basis": self.basis,
            "policy_hash": self.policy_hash,
            "evidence_ref": self.evidence_ref,
            "reason": self.reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def _evidence_ref(row: Mapping, basis: str) -> str | None:
    if basis == BASIS_VERIFIED_EXTERNAL or basis == BASIS_CODE_POLICY:
        value = row.get("evidence_url") or row.get("evidence_ref")
    elif basis == BASIS_SELF_GENERATED:
        value = row.get("provenance_ref") or row.get("generator_sha256")
    else:
        value = None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _reason(detail: str, status: str, policy: ResearchPolicy, action: str) -> str:
    text = (
        f"{detail}; action={action}; profile={policy.profile}; "
        f"external_license_status={status}"
    )
    if policy.profile == PRIVATE_PROFILE:
        text += (
            "; operator_private_research is a local research attestation and "
            "does not grant an external license or redistribution right"
        )
    return text


def _make(registry: Mapping, policy: ResearchPolicy, source_id: str, action: str,
          *, allow: bool, basis: str, detail: str,
          evidence_ref: str | None = None) -> PolicyDecision:
    status = _row_status(registry[source_id])
    ref = evidence_ref.strip() if isinstance(evidence_ref, str) and evidence_ref.strip() \
        else _evidence_ref(registry[source_id], basis)
    return PolicyDecision(
        source_id=source_id,
        action=action,
        operational_allow=bool(allow),
        external_license_status=status,
        basis=basis,
        policy_hash=policy.policy_hash,
        selected_profile=policy.profile,
        scope=policy.scope,
        evidence_ref=ref,
        reason=_reason(detail, status, policy, action),
    )


def decide(registry: Mapping, policy: ResearchPolicy, source_id: str, action: str,
           *, evidence_ref: str | None = None) -> PolicyDecision:
    """Return an explicit decision; the external status is never rewritten."""
    if not isinstance(policy, ResearchPolicy):
        raise ResearchPolicyError(
            "policy must be a ResearchPolicy", code="POLICY_INVALID_POLICY")
    if action not in ACTIONS:
        raise ResearchPolicyError(
            f"unknown action {action!r}; expected one of {list(ACTIONS)}",
            code="POLICY_UNKNOWN_ACTION")
    if source_id not in registry:
        raise ResearchPolicyError(
            f"unknown source {source_id!r}", code="POLICY_UNKNOWN_SOURCE")
    row = registry[source_id]
    if not isinstance(row, Mapping):
        raise ResearchPolicyError(
            f"registry row for {source_id!r} must be a mapping",
            code="POLICY_INVALID_REGISTRY")
    status = _row_status(row)
    private = policy.profile == PRIVATE_PROFILE

    def result(allow: bool, basis: str, detail: str) -> PolicyDecision:
        return _make(registry, policy, source_id, action, allow=allow, basis=basis,
                     detail=detail, evidence_ref=evidence_ref)

    if action == "audit":
        return result(
            True, BASIS_LOCAL_AUDIT,
            "local audit/indexing only; no training or redistribution is authorized")

    if action == "compile":
        allow = source_id in CODE_SOURCES and (
            (private and policy.covers(source_id)) or status == "ALLOWED")
        if allow:
            return result(True, BASIS_CODE_POLICY,
                          "software code may be built under its own code licence")
        return result(
            False, BASIS_NOT_COVERED,
            "source is not a code source covered by the selected profile/status")

    if action == "synthetic_train":
        if source_id in CODE_SOURCES:
            return result(
                False, BASIS_NOT_TRAINING_DATA,
                "software code is not a training corpus; use the code build path")
        if _self_generated(source_id):
            return result(
                True, BASIS_SELF_GENERATED,
                "first-party generated data may be used with generator provenance")
        if status == "ALLOWED" and str(row.get("train_allowed", "")).upper() == "ALLOWED":
            return result(True, BASIS_VERIFIED_EXTERNAL,
                          "externally ALLOWED source may back synthetic training")
        return result(
            False, BASIS_UNVERIFIED_EXTERNAL,
            "synthetic_train is limited to first-party generated data or an "
            "externally ALLOWED training source; use private_train for "
            "operator-private third-party corpora")

    if action == "private_train":
        if source_id in CODE_SOURCES:
            return result(
                False, BASIS_NOT_TRAINING_DATA,
                "software code is not a training corpus; its software licence "
                "does not license game data")
        if private:
            if policy.covers(source_id):
                return result(
                    True, BASIS_OPERATOR_ATTESTATION,
                    "local private training is allowed by the operator policy "
                    "allowlist only; this is not an external license")
            return result(
                False, BASIS_NOT_COVERED,
                "source is not in the operator policy allowlist")
        allow = (status == "ALLOWED"
                 and str(row.get("train_allowed", "")).upper() == "ALLOWED")
        if allow:
            return result(
                True, BASIS_VERIFIED_EXTERNAL,
                "verified ALLOWED source permits training under licensed_only")
        return result(
            False, BASIS_UNVERIFIED_EXTERNAL,
            "licensed_only requires external status=ALLOWED and "
            "train_allowed=ALLOWED")

    # distribute
    if private:
        return result(
            False, BASIS_NOT_DISTRIBUTABLE,
            "the operator_private_research profile never authorizes publication, "
            "redistribution, or network upload; the external status is reported "
            "separately")
    allow = (policy.release_allowed and status == "ALLOWED"
             and str(row.get("redistribute_allowed", "")).upper() == "ALLOWED")
    if allow:
        return result(
            True, BASIS_VERIFIED_EXTERNAL,
            "verified ALLOWED redistribution row under a release-enabled "
            "licensed_only policy")
    return result(
        False, BASIS_NOT_DISTRIBUTABLE,
        "licensed_only distribution requires release_allowed plus external "
        "status=ALLOWED and redistribute_allowed=ALLOWED")


__all__ = [
    "ACTIONS", "BASIS_CODE_POLICY", "BASIS_LOCAL_AUDIT", "BASIS_NOT_COVERED",
    "BASIS_NOT_DISTRIBUTABLE", "BASIS_NOT_TRAINING_DATA",
    "BASIS_OPERATOR_ATTESTATION", "BASIS_SELF_GENERATED",
    "BASIS_UNVERIFIED_EXTERNAL", "BASIS_VERIFIED_EXTERNAL", "CODE_SOURCES",
    "LICENSED_PROFILE", "PRIVATE_PROFILE", "PolicyDecision", "PROFILES",
    "ResearchPolicy", "ResearchPolicyError", "SELF_GENERATED_SOURCES",
    "STATUS_VALUES", "decide",
]
