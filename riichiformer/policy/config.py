"""Typed operator research-policy config (strict YAML/JSON, no secrets).

This module is the single source of truth for the *local research policy hash*
written into shards and checkpoints.  A policy config records an operator
choice -- which local sources they are willing to use for non-public research
-- and it is **not** a third-party licence, copyright permission, or a
redistribution grant.  The external facts continue to live in
``configs/riichiformer/source_rights.csv`` and must never be rewritten by a
policy config.

Design rules
------------
* Strict loading: the top-level mapping must contain exactly the known fields;
  unknown fields, duplicate keys, wrong types, and unknown enum values raise
  :class:`PolicyConfigError` with a stable ``code``.
* No secrets: the config is a small, committable-as-example preference file.
  The real file may stay local; only its canonical hash enters artifacts.
* Stable hash: :attr:`PolicyConfig.policy_hash` is the SHA-256 of the
  canonical JSON payload documented in :meth:`PolicyConfig.canonical_payload`.
  Key order in the source file and list order of the allowlist do not change
  the hash; every semantic field does.
* Mandatory separation: ``release_allowed`` and ``network_upload_allowed``
  must be ``false`` for ``operator_private_research``.  The private profile can
  never be turned into a publication/upload permission.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POLICY_VERSION = "rf-local-policy/1"
PROFILES = ("licensed_only", "operator_private_research")
SCOPES = ("local_nonpublic_only", "external_licensed_only")
PRIVATE_SCOPE = "local_nonpublic_only"

#: Canonical top-level fields.  ``operator_decision`` is accepted as a
#: documented alias for ``operator_reference`` (the PR-37 spec used that name);
#: it is normalised to one canonical key before hashing.
_REQUIRED_FIELDS = (
    "policy_version",
    "profile",
    "scope",
    "operator_reference",
    "allowed_local_sources",
)
_OPTIONAL_FIELDS = ("release_allowed", "network_upload_allowed")
_ALLOWED_FIELDS = frozenset(_REQUIRED_FIELDS + _OPTIONAL_FIELDS)
_FIELD_ALIASES = {"operator_decision": "operator_reference"}

_SOURCE_ID_RE = re.compile(r"^\S+$")


class PolicyConfigError(ValueError):
    """Malformed policy config; fail closed with a stable error code."""

    def __init__(self, message: str, *, code: str = "POLICY_CONFIG_INVALID"):
        super().__init__(message)
        self.code = code


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error(message: str, code: str) -> PolicyConfigError:
    return PolicyConfigError(message, code=code)


def _require_bool(mapping: Mapping[str, Any], name: str) -> bool:
    value = mapping.get(name, False)
    if not isinstance(value, bool):
        raise _error(
            f"policy config field {name!r} must be a JSON/YAML boolean, "
            f"got {type(value).__name__}", "POLICY_FIELD_TYPE")
    return value


def _require_text(mapping: Mapping[str, Any], name: str, *, code: str) -> str:
    value = mapping[name]
    if not isinstance(value, str):
        raise _error(
            f"policy config field {name!r} must be a string, got "
            f"{type(value).__name__}", "POLICY_FIELD_TYPE")
    text = value.strip()
    if not text:
        raise _error(f"policy config field {name!r} must be non-empty", code)
    if "\x00" in text or "\n" in text or "\r" in text:
        raise _error(
            f"policy config field {name!r} must be a single line without NUL",
            "POLICY_INVALID_OPERATOR_REFERENCE")
    return text


def _normalise_sources(value: Any) -> frozenset[str]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise _error(
            "policy config field 'allowed_local_sources' must be a list of "
            "source ids", "POLICY_INVALID_ALLOWLIST")
    seen: set[str] = set()
    ordered: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise _error(
                "every 'allowed_local_sources' entry must be a string",
                "POLICY_INVALID_ALLOWLIST")
        source = item.strip()
        if not source or not _SOURCE_ID_RE.match(source):
            raise _error(
                f"invalid source id in 'allowed_local_sources': {item!r}",
                "POLICY_INVALID_ALLOWLIST")
        if source in seen:
            raise _error(
                f"duplicate source id in 'allowed_local_sources': {source!r}",
                "POLICY_INVALID_ALLOWLIST")
        seen.add(source)
        ordered.append(source)
    return frozenset(ordered)


@dataclass(frozen=True)
class PolicyConfig:
    """A typed, hashable operator research-policy decision.

    Field order mirrors the semantic payload, not the file layout.  The
    constructor validates every field, so an in-memory config is exactly as
    strict as one loaded from disk.
    """

    profile: str
    operator_reference: str
    allowed_local_sources: frozenset[str]
    scope: str = PRIVATE_SCOPE
    release_allowed: bool = False
    network_upload_allowed: bool = False
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        if not isinstance(self.policy_version, str) \
                or self.policy_version != POLICY_VERSION:
            raise _error(
                f"policy_version must be {POLICY_VERSION!r}, got "
                f"{self.policy_version!r}", "POLICY_INVALID_VERSION")
        if not isinstance(self.profile, str) or self.profile not in PROFILES:
            raise _error(
                f"profile must be one of {PROFILES}, got {self.profile!r}",
                "POLICY_INVALID_PROFILE")
        if not isinstance(self.scope, str) or self.scope not in SCOPES:
            raise _error(
                f"scope must be one of {SCOPES}, got {self.scope!r}",
                "POLICY_INVALID_SCOPE")
        if not isinstance(self.operator_reference, str) \
                or not self.operator_reference.strip():
            raise _error(
                "operator_reference must be a non-empty string",
                "POLICY_INVALID_OPERATOR_REFERENCE")
        reference = self.operator_reference.strip()
        if "\n" in reference or "\r" in reference or "\x00" in reference:
            raise _error(
                "operator_reference must be a single line without NUL",
                "POLICY_INVALID_OPERATOR_REFERENCE")
        if not isinstance(self.release_allowed, bool):
            raise _error(
                "release_allowed must be boolean", "POLICY_FIELD_TYPE")
        if not isinstance(self.network_upload_allowed, bool):
            raise _error(
                "network_upload_allowed must be boolean", "POLICY_FIELD_TYPE")
        if isinstance(self.allowed_local_sources, str) \
                or not isinstance(self.allowed_local_sources,
                                  (frozenset, set, tuple, list)):
            raise _error(
                "allowed_local_sources must be a collection of source ids",
                "POLICY_INVALID_ALLOWLIST")
        sources = _normalise_sources(list(self.allowed_local_sources))
        if self.profile == "operator_private_research":
            if self.scope != PRIVATE_SCOPE:
                raise _error(
                    "operator_private_research requires scope="
                    f"{PRIVATE_SCOPE!r}", "POLICY_INVALID_SCOPE")
            if self.release_allowed:
                raise _error(
                    "operator_private_research cannot set release_allowed=true; "
                    "a local research profile never grants publication",
                    "POLICY_PRIVATE_RELEASE_FORBIDDEN")
            if self.network_upload_allowed:
                raise _error(
                    "operator_private_research cannot set "
                    "network_upload_allowed=true; uploads need a separate "
                    "authorisation", "POLICY_PRIVATE_UPLOAD_FORBIDDEN")
        # Store the canonical forms so equality/hash do not depend on input order.
        object.__setattr__(self, "operator_reference", reference)
        object.__setattr__(self, "allowed_local_sources", sources)

    @property
    def policy_hash(self) -> str:
        """SHA-256 of the canonical payload; stable and independently reproducible."""
        return hashlib.sha256(_canonical(self.canonical_payload()).encode("utf-8")).hexdigest()

    def canonical_payload(self) -> dict:
        """Exact fields covered by :attr:`policy_hash` (documented contract)."""
        return {
            "policy_version": self.policy_version,
            "profile": self.profile,
            "scope": self.scope,
            "operator_reference": self.operator_reference,
            "allowed_local_sources": sorted(self.allowed_local_sources),
            "release_allowed": self.release_allowed,
            "network_upload_allowed": self.network_upload_allowed,
        }

    def to_dict(self) -> dict:
        return self.canonical_payload()

    def to_research_policy(self):
        """Build the routing policy view (lazy import avoids an import cycle)."""
        from .research_policy import ResearchPolicy

        return ResearchPolicy.from_config(self)

    def covers(self, source_id: str) -> bool:
        return source_id in self.allowed_local_sources

    @classmethod
    def from_mapping(cls, value: Any) -> "PolicyConfig":
        if not isinstance(value, Mapping):
            raise _error(
                "policy config root must be a mapping/object, got "
                f"{type(value).__name__}", "POLICY_NOT_A_MAPPING")
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _error(
                    f"policy config keys must be strings, got {key!r}",
                    "POLICY_UNKNOWN_FIELD")
            name = _FIELD_ALIASES.get(key, key)
            if name in canonical:
                raise _error(
                    f"policy config field {name!r} specified more than once "
                    "(including its alias)", "POLICY_DUPLICATE_FIELD")
            canonical[name] = item
        unknown = sorted(set(canonical) - _ALLOWED_FIELDS)
        if unknown:
            raise _error(
                f"unknown policy config field(s): {unknown}; allowed fields are "
                f"{sorted(_ALLOWED_FIELDS)}", "POLICY_UNKNOWN_FIELD")
        missing = [name for name in _REQUIRED_FIELDS if name not in canonical]
        if missing:
            raise _error(
                f"missing required policy config field(s): {missing}",
                "POLICY_MISSING_FIELD")
        profile = _require_text(canonical, "profile", code="POLICY_INVALID_PROFILE")
        if profile not in PROFILES:
            raise _error(
                f"profile must be one of {PROFILES}, got {profile!r}",
                "POLICY_INVALID_PROFILE")
        scope = _require_text(canonical, "scope", code="POLICY_INVALID_SCOPE")
        if scope not in SCOPES:
            raise _error(
                f"scope must be one of {SCOPES}, got {scope!r}",
                "POLICY_INVALID_SCOPE")
        version = _require_text(
            canonical, "policy_version", code="POLICY_INVALID_VERSION")
        reference = _require_text(
            canonical, "operator_reference",
            code="POLICY_INVALID_OPERATOR_REFERENCE")
        sources = _normalise_sources(canonical["allowed_local_sources"])
        release = _require_bool(canonical, "release_allowed")
        upload = _require_bool(canonical, "network_upload_allowed")
        return cls(
            profile=profile,
            operator_reference=reference,
            allowed_local_sources=sources,
            scope=scope,
            release_allowed=release,
            network_upload_allowed=upload,
            policy_version=version,
        )


def _json_object(pairs):
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _error(
                f"duplicate JSON key in policy config: {key!r}",
                "POLICY_DUPLICATE_KEY")
        result[key] = value
    return result


def _yaml_loader():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment failure
        raise _error(
            "PyYAML is required to load YAML policy configs; install pyyaml or "
            "use a .json policy config", "POLICY_YAML_UNAVAILABLE") from exc

    class _StrictSafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise _error(
                    f"policy config keys must be strings, got {key!r}",
                    "POLICY_UNKNOWN_FIELD")
            if key in mapping:
                raise _error(
                    f"duplicate YAML key in policy config: {key!r}",
                    "POLICY_DUPLICATE_KEY")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _StrictSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return _StrictSafeLoader


def parse_policy_config_text(text: str, *, fmt: str = "yaml",
                             source: str = "<text>") -> PolicyConfig:
    """Parse one policy config document; rejects extra keys and duplicate keys."""
    if not isinstance(text, str):
        raise _error("policy config text must be a string", "POLICY_PARSE_ERROR")
    fmt = (fmt or "").lower()
    if fmt == "json":
        try:
            data = json.loads(text, object_pairs_hook=_json_object)
        except PolicyConfigError:
            raise
        except json.JSONDecodeError as exc:
            raise _error(
                f"cannot parse JSON policy config {source}: {exc}",
                "POLICY_PARSE_ERROR") from exc
    elif fmt in {"yaml", "yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment failure
            raise _error(
                "PyYAML is required to load YAML policy configs",
                "POLICY_YAML_UNAVAILABLE") from exc
        try:
            data = yaml.load(text, Loader=_yaml_loader())
        except PolicyConfigError:
            raise
        except yaml.YAMLError as exc:
            raise _error(
                f"cannot parse YAML policy config {source}: {exc}",
                "POLICY_PARSE_ERROR") from exc
    else:
        raise _error(
            f"unsupported policy config format {fmt!r}; use json or yaml",
            "POLICY_PARSE_ERROR")
    return PolicyConfig.from_mapping(data)


def load_policy_config(path: str | Path) -> PolicyConfig:
    """Load a strict YAML/JSON policy config from disk (read-only, no network)."""
    config_path = Path(path)
    if not config_path.is_file():
        raise _error(
            f"policy config not found: {config_path}", "POLICY_FILE_NOT_FOUND")
    try:
        text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _error(
            f"cannot read policy config {config_path}: {exc}",
            "POLICY_PARSE_ERROR") from exc
    fmt = "json" if config_path.suffix.lower() == ".json" else "yaml"
    return parse_policy_config_text(text, fmt=fmt, source=str(config_path))


__all__ = [
    "POLICY_VERSION", "PRIVATE_SCOPE", "PROFILES", "SCOPES",
    "PolicyConfig", "PolicyConfigError", "load_policy_config",
    "parse_policy_config_text",
]
