#!/usr/bin/env python3
"""PR-42 opponent registry: every arena seat must be a declared, labelled opponent.

The registry exists to make impersonation impossible:

* an unknown name is refused (no silently invented opponent);
* a local heuristic bot can never be registered as ``mortal``/``akochan``/a Tenhou
  engine, and cannot carry a checkpoint hash (it has no weights);
* a third-party engine/model entry requires a content hash **and** a licence caveat,
  and is refused outright unless the caller opts in explicitly;
* every entry names the ruleset it is eligible for.

No entry in this repository registers Mortal, akochan, a Tenhou engine or any external
checkpoint: this worktree has no approved third-party weight.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

from ..selfplay.provenance import (
    RULESET_METADATA,
    ProvenanceError,
    assert_known_ruleset,
)

__all__ = [
    "OpponentKind",
    "OpponentSpec",
    "OpponentRegistry",
    "OpponentError",
    "UnknownOpponent",
    "ImpersonationError",
    "ExternalOpponentRefused",
    "RegistryConflict",
    "default_registry",
    "build_lineup",
    "RESERVED_ENGINE_TOKENS",
]

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

#: Tokens that may only appear on an explicitly-labelled third-party engine entry.
RESERVED_ENGINE_TOKENS: tuple[str, ...] = (
    "mortal",
    "akochan",
    "tenhou",
    "天凤",
    "majsoul",
    "雀魂",
    "九段",
)


class OpponentKind(str, Enum):
    LOCAL_HEURISTIC = "local_heuristic"
    LOCAL_MODEL = "local_model"
    EXTERNAL_ENGINE = "external_engine"


_KIND_CHECKPOINT = {
    OpponentKind.LOCAL_HEURISTIC: False,   # must be None
    OpponentKind.LOCAL_MODEL: True,        # must be a sha256
    OpponentKind.EXTERNAL_ENGINE: False,   # may be None only with a caveat
}

_MIN_CAVEAT = 20


class OpponentError(ValueError):
    """Base class for registry contract violations (fail closed)."""


class UnknownOpponent(OpponentError):
    """The requested opponent is not registered."""


class ImpersonationError(OpponentError):
    """A local bot was labelled as a third-party engine or claimed its strength."""


class ExternalOpponentRefused(OpponentError):
    """A third-party engine/model entry was requested without explicit approval."""


class RegistryConflict(OpponentError):
    """The same opponent name was registered twice with different metadata."""


def _normalise_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpponentError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class OpponentSpec:
    """One registered opponent (identity + licence/rules caveat)."""

    name: str
    version: str
    kind: OpponentKind
    checkpoint_sha256: str | None
    license_caveat: str
    ruleset: str
    display_name: str = ""
    code_sha256: str | None = None
    external: bool = False
    notes: str = ""

    def __post_init__(self):
        name = _normalise_text(self.name, "name")
        if not _NAME_RE.match(name):
            raise OpponentError(
                f"opponent name {name!r} must match {_NAME_RE.pattern} (lowercase id)")
        version = _normalise_text(self.version, "version")
        if not isinstance(self.kind, OpponentKind):
            raise OpponentError(f"kind must be an OpponentKind, got {self.kind!r}")
        caveat = _normalise_text(self.license_caveat, "license_caveat")
        if len(caveat) < _MIN_CAVEAT:
            raise OpponentError(
                f"license_caveat must be at least {_MIN_CAVEAT} characters "
                f"(got {len(caveat)})")
        try:
            assert_known_ruleset(self.ruleset)
        except ProvenanceError as exc:
            raise OpponentError(str(exc)) from None
        if self.checkpoint_sha256 is not None:
            if not isinstance(self.checkpoint_sha256, str) \
                    or not _SHA_RE.match(self.checkpoint_sha256):
                raise OpponentError("checkpoint_sha256 must be 64 lowercase hex chars")
        if self.code_sha256 is not None and not _SHA_RE.match(self.code_sha256):
            raise OpponentError("code_sha256 must be 64 lowercase hex chars")
        if self.kind is OpponentKind.LOCAL_HEURISTIC:
            if self.checkpoint_sha256 is not None:
                raise ImpersonationError(
                    f"local heuristic opponent {name!r} cannot carry a checkpoint hash")
            if self.external:
                raise ImpersonationError(
                    f"local heuristic opponent {name!r} cannot be marked external")
        if self.kind is OpponentKind.LOCAL_MODEL and self.checkpoint_sha256 is None:
            raise OpponentError("local_model opponents require a checkpoint_sha256")
        if self.kind is OpponentKind.EXTERNAL_ENGINE:
            if not self.external:
                raise ExternalOpponentRefused(
                    f"external engine {name!r} must be marked external=True "
                    "(explicit third-party approval)")
            if self.checkpoint_sha256 is None and not self.notes:
                raise OpponentError(
                    "external_engine without checkpoint_sha256 must document the engine "
                    "build in notes")
        lowered = f"{name} {self.display_name} {version}".lower()
        for token in RESERVED_ENGINE_TOKENS:
            if token in lowered and self.kind is not OpponentKind.EXTERNAL_ENGINE:
                raise ImpersonationError(
                    f"opponent {name!r} uses reserved token {token!r}; local bots must "
                    "not be presented as a third-party engine")

    @property
    def is_local_heuristic(self) -> bool:
        return self.kind is OpponentKind.LOCAL_HEURISTIC

    def as_dict(self) -> dict:
        row = asdict(self)
        row["kind"] = self.kind.value
        row["official_rank"] = False
        return row


class OpponentRegistry:
    """A closed registry: only declared opponents can enter an arena lineup."""

    def __init__(self, *, allow_external: bool = False):
        self._entries: dict[str, OpponentSpec] = {}
        self.allow_external = bool(allow_external)

    # -- registration ------------------------------------------------------
    def register(self, spec: OpponentSpec) -> OpponentSpec:
        if not isinstance(spec, OpponentSpec):
            raise OpponentError("register() expects an OpponentSpec")
        if spec.kind is OpponentKind.EXTERNAL_ENGINE and not self.allow_external:
            raise ExternalOpponentRefused(
                f"external engine {spec.name!r} is not approved in this worktree "
                "(no third-party weight/engine permission has been recorded)")
        existing = self._entries.get(spec.name)
        if existing is not None:
            if existing == spec:
                return existing
            raise RegistryConflict(
                f"opponent {spec.name!r} is already registered with different metadata")
        self._entries[spec.name] = spec
        return spec

    # -- lookup ------------------------------------------------------------
    def get(self, name: str) -> OpponentSpec:
        key = str(name).strip().lower()
        try:
            return self._entries[key]
        except KeyError:
            raise UnknownOpponent(
                f"opponent {name!r} is not registered; known: {sorted(self._entries)}"
            ) from None

    def require(self, name: str, *, kind: OpponentKind | None = None) -> OpponentSpec:
        spec = self.get(name)
        if kind is not None and spec.kind is not kind:
            raise OpponentError(
                f"opponent {name!r} has kind {spec.kind.value}, expected {kind.value}")
        return spec

    def resolve(self, names: Iterable[str]) -> tuple[OpponentSpec, ...]:
        return tuple(self.get(name) for name in names)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def __contains__(self, name) -> bool:
        return str(name).strip().lower() in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # -- reporting ---------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "format": "rf-opponent-registry/1",
            "allow_external": self.allow_external,
            "opponents": {
                name: spec.as_dict() for name, spec in sorted(self._entries.items())
            },
            "rulesets": {
                ruleset: {
                    "parity_status": metadata["parity_status"],
                    "tenhou_equivalent": metadata["tenhou_equivalent"],
                }
                for ruleset, metadata in sorted(RULESET_METADATA.items())
            },
            "note": (
                "no Mortal/akochan/Tenhou engine or third-party checkpoint is registered; "
                "local heuristics carry no weights and make no strength claim"
            ),
        }


def _local_specs() -> list[OpponentSpec]:
    from ..selfplay.bots import bot_specs

    ruleset = "libriichi_tenhou_approx/1"
    specs = []
    for row in bot_specs():
        specs.append(OpponentSpec(
            name=row["name"],
            version=row["version"],
            kind=OpponentKind.LOCAL_HEURISTIC,
            checkpoint_sha256=None,
            license_caveat=row["license_caveat"],
            ruleset=ruleset,
            display_name=row["display_name"],
            notes="weight-free heuristic; no checkpoint and no external data",
        ))
    return specs


def default_registry(*, allow_external: bool = False) -> OpponentRegistry:
    registry = OpponentRegistry(allow_external=allow_external)
    for spec in _local_specs():
        registry.register(spec)
    return registry


def build_lineup(names: Iterable[str], *,
                 registry: OpponentRegistry | None = None) -> tuple[str, tuple[str, ...]]:
    """Resolve 2 names (1v3) or 4 names (free-for-all) into a four-seat lineup.

    Returns ``(mode, lineup)`` where ``lineup[0]`` is the challenger/champion slot and
    positions 1..3 fill the other seats.  A 2-name matchup becomes ``(a, b, b, b)``,
    i.e. one challenger against three copies of the champion, exactly the
    ``OneVsThree`` arrangement; a 4-name lineup is a free-for-all.
    """
    resolved = list(names)
    if len(resolved) not in (2, 4):
        raise OpponentError(
            f"--opponents needs 2 (1v3) or 4 (free-for-all) names, got {len(resolved)}")
    reg = registry if registry is not None else default_registry()
    for name in resolved:
        reg.get(name)
    if len(resolved) == 2:
        return "one_vs_three", (resolved[0], resolved[1], resolved[1], resolved[1])
    return "free_for_all", tuple(resolved)
