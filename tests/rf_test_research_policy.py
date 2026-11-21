"""PR-37 policy tests: typed config, canonical hash, and decision matrix.

No source_rights.csv row is modified and no external status is ever rewritten
to ALLOWED in these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from riichiformer.policy.config import (
    POLICY_VERSION,
    PolicyConfig,
    PolicyConfigError,
    load_policy_config,
    parse_policy_config_text,
)
from riichiformer.policy.research_policy import (
    ACTIONS,
    ResearchPolicy,
    ResearchPolicyError,
    decide,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "configs/riichiformer/local_research_policy.example.yaml"
EXAMPLE_HASH = "72ac377f2fa44297c18228aeeba59160de48f67c65bc3b3b242a3e9b889251b0"
BASE_TEST_HASH = "aebae1382c96a62b1a46a75bc36c40ce649caa06e427b70af758bdb1af14b088"

REG = {
    "tenhou_old": {
        "status": "RESTRICTED",
        "train_allowed": "RESTRICTED",
        "redistribute_allowed": "RESTRICTED",
    },
    "selfplay_own": {
        "status": "UNKNOWN",
        "train_allowed": "UNKNOWN",
        "redistribute_allowed": "UNKNOWN",
    },
    "mortal_code": {
        "status": "UNKNOWN",
        "train_allowed": "UNKNOWN",
        "redistribute_allowed": "UNKNOWN",
    },
    "fixture_licensed": {
        "status": "ALLOWED",
        "train_allowed": "ALLOWED",
        "redistribute_allowed": "ALLOWED",
        "evidence_url": "https://example.invalid/fixture-license",
    },
}


def policy(profile="operator_private_research"):
    return ResearchPolicy(profile, frozenset(REG), "user private research preference")


def config_mapping(**overrides):
    data = {
        "policy_version": POLICY_VERSION,
        "profile": "operator_private_research",
        "scope": "local_nonpublic_only",
        "operator_reference": "unit-test-operator",
        "allowed_local_sources": ["tenhou_old", "selfplay_own"],
        "release_allowed": False,
        "network_upload_allowed": False,
    }
    data.update(overrides)
    return data


def licensed_mapping(**overrides):
    data = {
        "policy_version": POLICY_VERSION,
        "profile": "licensed_only",
        "scope": "external_licensed_only",
        "operator_reference": "unit-test-licensed-reviewer",
        "allowed_local_sources": ["fixture_licensed"],
        "release_allowed": True,
        "network_upload_allowed": False,
    }
    data.update(overrides)
    return data


def write_yaml(tmp_path: Path, mapping: dict, name: str = "policy.yaml") -> Path:
    import yaml
    path = tmp_path / name
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path


def write_json(tmp_path: Path, mapping: dict, name: str = "policy.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Decision matrix (kept from the first PR-37 scaffold; behaviour unchanged)
# ---------------------------------------------------------------------------

def test_private_policy_preserves_license_evidence():
    d = decide(REG, policy(), "tenhou_old", "private_train")
    assert d.operational_allow and d.external_license_status == "RESTRICTED"
    assert d.source_rights_status == "RESTRICTED"
    assert d.basis == "operator_attestation_not_external_license"
    assert d.selected_profile == "operator_private_research"


def test_private_profile_cannot_distribute():
    assert not decide(REG, policy(), "tenhou_old", "distribute").operational_allow


def test_private_profile_cannot_distribute_even_a_licensed_row():
    d = decide(REG, policy(), "fixture_licensed", "distribute")
    assert not d.operational_allow
    assert d.external_license_status == "ALLOWED"  # fact preserved, never a grant


def test_strict_profile_still_blocks_unlicensed_training():
    assert not decide(REG, policy("licensed_only"), "tenhou_old", "private_train").operational_allow


def test_strict_profile_allows_verified_allowed_training():
    d = decide(REG, policy("licensed_only"), "fixture_licensed", "private_train")
    assert d.operational_allow
    assert d.basis == "verified_external"
    assert d.external_license_status == "ALLOWED"


def test_code_does_not_become_a_training_corpus():
    assert decide(REG, policy(), "mortal_code", "compile").operational_allow
    assert not decide(REG, policy(), "mortal_code", "private_train").operational_allow


def test_selfplay_synthetic_train_requires_no_http_evidence():
    d = decide(REG, policy(), "selfplay_own", "synthetic_train")
    assert d.operational_allow
    assert d.basis == "self_generated"
    assert d.evidence_ref is None
    assert d.external_license_status == "UNKNOWN"


def test_thirdparty_synthetic_train_is_denied():
    d = decide(REG, policy(), "tenhou_old", "synthetic_train")
    assert not d.operational_allow
    assert d.external_license_status == "RESTRICTED"


def test_unknown_source_is_not_accepted():
    with pytest.raises(ResearchPolicyError) as ctx:
        decide(REG, policy(), "unregistered", "private_train")
    assert ctx.value.code == "POLICY_UNKNOWN_SOURCE"


def test_unknown_action_is_rejected_with_stable_code():
    with pytest.raises(ResearchPolicyError) as ctx:
        decide(REG, policy(), "tenhou_old", "publish")
    assert ctx.value.code == "POLICY_UNKNOWN_ACTION"


def test_invalid_status_is_rejected():
    registry = {"x": {"status": "MAYBE", "train_allowed": "MAYBE"}}
    with pytest.raises(ResearchPolicyError) as ctx:
        decide(registry, policy(), "x", "private_train")
    assert ctx.value.code == "POLICY_INVALID_STATUS"


def test_invalid_private_release_flag_fails():
    with pytest.raises(ResearchPolicyError) as ctx:
        ResearchPolicy("operator_private_research", frozenset(), "user", release_allowed=True)
    assert ctx.value.code == "POLICY_PRIVATE_RELEASE_FORBIDDEN"


def test_policy_hash_tied_to_source_scope():
    assert policy().policy_hash != ResearchPolicy(
        "operator_private_research", frozenset({"selfplay_own"}),
        "user private research preference").policy_hash


def test_decision_serialises_all_contract_fields():
    d = decide(REG, policy(), "tenhou_old", "private_train")
    payload = json.loads(d.to_json())
    assert payload["source_id"] == "tenhou_old"
    assert payload["action"] == "private_train"
    assert payload["operational_allow"] is True
    assert payload["external_license_status"] == "RESTRICTED"
    assert payload["source_rights_status"] == "RESTRICTED"
    assert payload["basis"] == "operator_attestation_not_external_license"
    assert payload["policy_hash"] == policy().policy_hash
    assert payload["selected_profile"] == "operator_private_research"
    assert "not an external license" in payload["reason"]
    assert set(ACTIONS) == {"audit", "compile", "synthetic_train",
                            "private_train", "distribute"}


def test_decision_from_config_matches_policy_hash():
    config = PolicyConfig.from_mapping(config_mapping())
    d = decide(REG, config.to_research_policy(), "tenhou_old", "private_train")
    assert d.policy_hash == config.policy_hash
    assert d.scope == "local_nonpublic_only"


# ---------------------------------------------------------------------------
# Typed PolicyConfig strict loading + canonical hash
# ---------------------------------------------------------------------------

class TestPolicyConfigLoading:
    def test_yaml_and_json_load_to_the_same_config_and_hash(self, tmp_path):
        mapping = config_mapping()
        yaml_config = load_policy_config(write_yaml(tmp_path, mapping))
        json_config = load_policy_config(write_json(tmp_path, mapping))
        assert yaml_config == json_config
        assert yaml_config.policy_hash == json_config.policy_hash

    def test_example_config_loads_with_stable_hash(self):
        assert EXAMPLE_CONFIG.is_file()
        config = load_policy_config(EXAMPLE_CONFIG)
        assert config.profile == "operator_private_research"
        assert config.scope == "local_nonpublic_only"
        assert config.policy_version == POLICY_VERSION
        assert config.release_allowed is False
        assert config.network_upload_allowed is False
        assert "tenhou_old" in config.allowed_local_sources
        assert config.policy_hash == EXAMPLE_HASH

    def test_known_canonical_hash_vector(self):
        config = PolicyConfig.from_mapping(config_mapping())
        assert config.policy_hash == BASE_TEST_HASH

    def test_hash_ignores_key_and_allowlist_order(self, tmp_path):
        forward = config_mapping()
        reverse = {
            "network_upload_allowed": False,
            "allowed_local_sources": list(reversed(forward["allowed_local_sources"])),
            "operator_reference": forward["operator_reference"],
            "scope": forward["scope"],
            "profile": forward["profile"],
            "policy_version": forward["policy_version"],
            "release_allowed": False,
        }
        assert PolicyConfig.from_mapping(forward).policy_hash == \
            PolicyConfig.from_mapping(reverse).policy_hash

    def test_hash_covers_every_semantic_field(self):
        base = PolicyConfig.from_mapping(config_mapping())
        variants = [
            PolicyConfig.from_mapping(config_mapping(operator_reference="someone-else")),
            PolicyConfig.from_mapping(config_mapping(
                allowed_local_sources=["tenhou_old"])),
            PolicyConfig.from_mapping(config_mapping(
                allowed_local_sources=["tenhou_old", "selfplay_own", "mortal_code"])),
            PolicyConfig.from_mapping(licensed_mapping()),
        ]
        hashes = {base.policy_hash} | {item.policy_hash for item in variants}
        assert len(hashes) == 5

    def test_hash_covers_scope_and_flags_on_licensed_profile(self):
        base = PolicyConfig.from_mapping(licensed_mapping())
        other_scope = PolicyConfig.from_mapping(licensed_mapping(
            scope="local_nonpublic_only"))
        upload = PolicyConfig.from_mapping(licensed_mapping(
            network_upload_allowed=True))
        no_release = PolicyConfig.from_mapping(licensed_mapping(
            release_allowed=False))
        hashes = {base.policy_hash, other_scope.policy_hash, upload.policy_hash,
                  no_release.policy_hash}
        assert len(hashes) == 4

    def test_unknown_field_rejected(self, tmp_path):
        mapping = config_mapping()
        mapping["release_override"] = True
        with pytest.raises(PolicyConfigError) as ctx:
            load_policy_config(write_yaml(tmp_path, mapping))
        assert ctx.value.code == "POLICY_UNKNOWN_FIELD"
        assert "release_override" in str(ctx.value)

    def test_missing_field_rejected(self):
        mapping = config_mapping()
        del mapping["profile"]
        with pytest.raises(PolicyConfigError) as ctx:
            PolicyConfig.from_mapping(mapping)
        assert ctx.value.code == "POLICY_MISSING_FIELD"

    @pytest.mark.parametrize("overrides,code", [
        ({"release_allowed": True}, "POLICY_PRIVATE_RELEASE_FORBIDDEN"),
        ({"network_upload_allowed": True}, "POLICY_PRIVATE_UPLOAD_FORBIDDEN"),
        ({"scope": "external_licensed_only"}, "POLICY_INVALID_SCOPE"),
        ({"profile": "operator"}, "POLICY_INVALID_PROFILE"),
        ({"policy_version": "rf-local-policy/0"}, "POLICY_INVALID_VERSION"),
        ({"operator_reference": "   "}, "POLICY_INVALID_OPERATOR_REFERENCE"),
        ({"allowed_local_sources": "tenhou_old"}, "POLICY_INVALID_ALLOWLIST"),
        ({"allowed_local_sources": [1]}, "POLICY_INVALID_ALLOWLIST"),
        ({"allowed_local_sources": ["tenhou_old", "tenhou_old"]},
         "POLICY_INVALID_ALLOWLIST"),
        ({"allowed_local_sources": ["has space"]}, "POLICY_INVALID_ALLOWLIST"),
        ({"release_allowed": "false"}, "POLICY_FIELD_TYPE"),
        ({"network_upload_allowed": "false"}, "POLICY_FIELD_TYPE"),
    ])
    def test_invalid_fields_rejected_with_stable_codes(self, overrides, code):
        with pytest.raises(PolicyConfigError) as ctx:
            PolicyConfig.from_mapping(config_mapping(**overrides))
        assert ctx.value.code == code

    def test_unknown_top_level_type_rejected(self):
        with pytest.raises(PolicyConfigError) as ctx:
            PolicyConfig.from_mapping(["not", "a", "mapping"])
        assert ctx.value.code == "POLICY_NOT_A_MAPPING"

    def test_duplicate_yaml_keys_rejected(self, tmp_path):
        path = tmp_path / "duplicate.yaml"
        path.write_text(
            "policy_version: rf-local-policy/1\n"
            "profile: operator_private_research\n"
            "profile: licensed_only\n"
            "scope: local_nonpublic_only\n"
            "operator_reference: unit-test-operator\n"
            "allowed_local_sources: [selfplay_own]\n",
            encoding="utf-8")
        with pytest.raises(PolicyConfigError) as ctx:
            load_policy_config(path)
        assert ctx.value.code == "POLICY_DUPLICATE_KEY"

    def test_duplicate_json_keys_rejected(self, tmp_path):
        path = tmp_path / "duplicate.json"
        path.write_text(
            json.dumps(config_mapping())[:-1]
            + ', "profile": "licensed_only"}',
            encoding="utf-8")
        with pytest.raises(PolicyConfigError) as ctx:
            load_policy_config(path)
        assert ctx.value.code == "POLICY_DUPLICATE_KEY"

    def test_parse_errors_are_stable(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("profile: [unterminated\n", encoding="utf-8")
        with pytest.raises(PolicyConfigError) as ctx:
            load_policy_config(bad_yaml)
        assert ctx.value.code == "POLICY_PARSE_ERROR"
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        with pytest.raises(PolicyConfigError) as ctx:
            load_policy_config(bad_json)
        assert ctx.value.code == "POLICY_PARSE_ERROR"

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(PolicyConfigError) as ctx:
            load_policy_config(tmp_path / "absent.yaml")
        assert ctx.value.code == "POLICY_FILE_NOT_FOUND"

    def test_operator_decision_alias_is_normalised(self):
        mapping = config_mapping()
        reference = mapping.pop("operator_reference")
        mapping["operator_decision"] = reference
        alias = PolicyConfig.from_mapping(mapping)
        assert alias.operator_reference == reference
        assert alias.policy_hash == BASE_TEST_HASH

    def test_alias_and_canonical_field_together_rejected(self):
        mapping = config_mapping()
        mapping["operator_decision"] = "another-reference"
        with pytest.raises(PolicyConfigError) as ctx:
            PolicyConfig.from_mapping(mapping)
        assert ctx.value.code == "POLICY_DUPLICATE_FIELD"

    def test_config_research_policy_roundtrip(self):
        config = PolicyConfig.from_mapping(config_mapping())
        roundtrip = config.to_research_policy().to_config()
        assert roundtrip == config
        assert roundtrip.policy_hash == config.policy_hash

    def test_canonical_payload_is_documented_and_exact(self):
        config = PolicyConfig.from_mapping(config_mapping())
        assert config.canonical_payload() == {
            "policy_version": POLICY_VERSION,
            "profile": "operator_private_research",
            "scope": "local_nonpublic_only",
            "operator_reference": "unit-test-operator",
            "allowed_local_sources": ["selfplay_own", "tenhou_old"],
            "release_allowed": False,
            "network_upload_allowed": False,
        }
        assert len(config.policy_hash) == 64

    def test_parse_text_requires_known_format(self):
        with pytest.raises(PolicyConfigError) as ctx:
            parse_policy_config_text("{}", fmt="toml")
        assert ctx.value.code == "POLICY_PARSE_ERROR"

    def test_string_blank_lines_and_comments_are_allowed(self, tmp_path):
        path = tmp_path / "comments.yaml"
        path.write_text(
            "# comment\n\n"
            "policy_version: rf-local-policy/1\n"
            "profile: operator_private_research\n"
            "scope: local_nonpublic_only\n"
            "operator_reference: '  spaced reference  '\n"
            "allowed_local_sources:\n  - selfplay_own\n"
            "release_allowed: false\n"
            "network_upload_allowed: false\n",
            encoding="utf-8")
        config = load_policy_config(path)
        assert config.operator_reference == "spaced reference"


class TestPolicyConfigDecisionIntegration:
    def test_private_config_allows_only_allowlisted_private_training(self):
        config = PolicyConfig.from_mapping(config_mapping())
        routing = config.to_research_policy()
        assert decide(REG, routing, "tenhou_old", "private_train").operational_allow
        assert not decide(REG, routing, "fixture_licensed", "private_train").operational_allow
        assert not decide(REG, routing, "tenhou_old", "distribute").operational_allow

    def test_licensed_release_config_can_distribute_only_verified_rows(self):
        routing = PolicyConfig.from_mapping(licensed_mapping()).to_research_policy()
        allowed = decide(REG, routing, "fixture_licensed", "distribute")
        assert allowed.operational_allow
        assert allowed.basis == "verified_external"
        denied = decide(REG, routing, "selfplay_own", "distribute")
        assert not denied.operational_allow
        assert denied.external_license_status == "UNKNOWN"

    def test_registry_status_is_never_mutated_by_a_decision(self):
        before = json.dumps(REG, sort_keys=True)
        routing = PolicyConfig.from_mapping(config_mapping()).to_research_policy()
        for action in ACTIONS:
            decide(REG, routing, "tenhou_old", action)
        assert json.dumps(REG, sort_keys=True) == before
