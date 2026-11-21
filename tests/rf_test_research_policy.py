from riichiformer.policy.research_policy import ResearchPolicy, decide, ResearchPolicyError
import pytest

REG = {
    "tenhou_old": {"status": "RESTRICTED", "train_allowed": "RESTRICTED", "redistribute_allowed": "RESTRICTED"},
    "selfplay_own": {"status": "UNKNOWN", "train_allowed": "UNKNOWN", "redistribute_allowed": "UNKNOWN"},
    "mortal_code": {"status": "UNKNOWN", "train_allowed": "UNKNOWN", "redistribute_allowed": "UNKNOWN"},
}
def policy(profile="operator_private_research"):
    return ResearchPolicy(profile, frozenset(REG), "user private research preference")

def test_private_policy_preserves_license_evidence():
    d = decide(REG, policy(), "tenhou_old", "private_train")
    assert d.operational_allow and d.external_license_status == "RESTRICTED"
    assert d.basis == "operator_attestation_not_external_license"

def test_private_profile_cannot_distribute():
    assert not decide(REG, policy(), "tenhou_old", "distribute").operational_allow

def test_strict_profile_still_blocks_unlicensed_training():
    assert not decide(REG, policy("licensed_only"), "tenhou_old", "private_train").operational_allow

def test_code_does_not_become_a_training_corpus():
    assert decide(REG, policy(), "mortal_code", "compile").operational_allow
    assert not decide(REG, policy(), "mortal_code", "private_train").operational_allow

def test_unknown_source_is_not_accepted():
    with pytest.raises(ResearchPolicyError):
        decide(REG, policy(), "unregistered", "private_train")

def test_invalid_private_release_flag_fails():
    with pytest.raises(ResearchPolicyError):
        ResearchPolicy("operator_private_research", frozenset(), "user", release_allowed=True)

def test_policy_hash_tied_to_source_scope():
    assert policy().policy_hash != ResearchPolicy("operator_private_research", frozenset({"selfplay_own"}), "user private research preference").policy_hash
