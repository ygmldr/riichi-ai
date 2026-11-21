"""Pure project routing policy; never represents third-party licence grant."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass

class ResearchPolicyError(ValueError):
    pass

@dataclass(frozen=True)
class ResearchPolicy:
    profile: str
    local_source_allowlist: frozenset[str]
    operator_reference: str
    release_allowed: bool = False
    network_upload_allowed: bool = False

    def __post_init__(self):
        if self.profile not in {"licensed_only", "operator_private_research"}:
            raise ResearchPolicyError("unknown profile")
        if not self.operator_reference:
            raise ResearchPolicyError("missing operator reference")
        if self.profile == "operator_private_research" and (self.release_allowed or self.network_upload_allowed):
            raise ResearchPolicyError("private profile cannot grant release/upload")

    @property
    def policy_hash(self) -> str:
        payload = json.dumps({
            "profile": self.profile,
            "sources": sorted(self.local_source_allowlist),
            "reference": self.operator_reference,
            "release": self.release_allowed,
            "upload": self.network_upload_allowed,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

@dataclass(frozen=True)
class PolicyDecision:
    source_id: str
    action: str
    operational_allow: bool
    external_license_status: str
    basis: str
    policy_hash: str

_CODE_SOURCES = {"tenhou_to_mjai_code", "mortal_code", "akochan_code"}

def decide(registry: dict, policy: ResearchPolicy, source_id: str, action: str) -> PolicyDecision:
    if source_id not in registry:
        raise ResearchPolicyError("unknown source")
    if action not in {"audit", "compile", "private_train", "distribute"}:
        raise ResearchPolicyError("unknown action")
    row = registry[source_id]
    status = row["status"]
    if status not in {"ALLOWED", "UNKNOWN", "RESTRICTED"}:
        raise ResearchPolicyError("invalid rights status")
    if action == "audit":
        allow, basis = True, "local_audit_only"
    elif action == "compile":
        allow = source_id in _CODE_SOURCES and (
            (policy.profile == "operator_private_research" and source_id in policy.local_source_allowlist)
            or status == "ALLOWED")
        basis = "code_use_policy" if allow else "not_covered"
    elif action == "private_train":
        if source_id in _CODE_SOURCES:
            allow, basis = False, "software_is_not_training_data"
        elif policy.profile == "operator_private_research" and source_id in policy.local_source_allowlist:
            allow, basis = True, "operator_attestation_not_external_license"
        else:
            allow = status == "ALLOWED" and row["train_allowed"] == "ALLOWED"
            basis = "verified_external" if allow else "unverified_external"
    else:
        allow = policy.profile == "licensed_only" and policy.release_allowed and (
            status == "ALLOWED" and row["redistribute_allowed"] == "ALLOWED")
        basis = "verified_external" if allow else "not_authorized_for_distribution"
    return PolicyDecision(source_id, action, bool(allow), status, basis, policy.policy_hash)
