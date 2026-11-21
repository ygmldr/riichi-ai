#!/usr/bin/env python3
"""PR-45/PR-47 independent validation corpus and read-only diagnostics.

This package is **not** a training path.  It builds a self-play validation corpus
from a fresh wall window, proves that corpus is disjoint from the B0 training
sample and from every historical evaluation seed window, seals it into an
``rf-shard/3`` with ``purpose=diagnostic``, and reads it back through an explicit
read-only reader that can only run forwards (CE/top1/top3), never an optimizer
step.

Modules
-------
``disjoint``
    Seed-window / game-id / generator-hash intersection audit against the frozen
    B0 500-game sample and the historical evaluation windows.
``integrity``
    Re-hash every stored input/target row (the loader does not do this) and audit
    the POV visibility contract (``UNKNOWN`` only on opponents' hidden events).
``shard_build``
    ``selfplay engine log -> build_decisions -> make_game_record -> ShardWriter``
    with ``purpose=diagnostic``, ``source_kind=arena_selfplay``,
    ``ruleset_id=mortal_arena_pinned`` and R02/R25/R26/R29 provenance.
``corpus``
    Wall-window generation via the existing ``selfplay.rollout`` machinery, plus
    the corpus manifest.  Weak local heuristic bots only: no strong-teacher and
    no Tenhou-truth claim.
``reader``
    :class:`ValidationReader`: forward-only, hash-checked, no optimizer surface.
``metrics``
    CE/NLL, top1/top3, legality/non-finite rates, phase and scenario buckets,
    sequence-length/prefix coverage.
``prefix_parity``
    Same-weights prefix vs multi-decision logits parity on a small fixture.
``e2e``
    Native game -> four POV streams -> decisions -> shard -> loader -> 2-step
    six-layer CPU train -> checkpoint -> reload/inference roundtrip.
"""

from .corpus import (
    MAX_VALIDATION_GAMES,
    ValidationCorpusError,
    ValidationWindow,
    generate_seed_key,
    run_validation_selfplay,
)
from .disjoint import (
    B0_SAMPLE_SEED,
    HISTORICAL_SEED_WINDOWS,
    RESERVED_SEED_KEYS,
    audit_seed_window,
    build_intersection_report,
    load_b0_game_ids,
)
from .integrity import (
    IntegrityError,
    assert_visibility_contract,
    verify_shard_integrity,
)
from .reader import DecisionContext, ValidationBatch, ValidationError, ValidationReader
from .shard_build import (
    SELFPLAY_HASH_DERIVATION,
    ValidationShardSpec,
    build_validation_shard,
)

__all__ = [
    "B0_SAMPLE_SEED",
    "DecisionContext",
    "HISTORICAL_SEED_WINDOWS",
    "IntegrityError",
    "MAX_VALIDATION_GAMES",
    "RESERVED_SEED_KEYS",
    "SELFPLAY_HASH_DERIVATION",
    "ValidationBatch",
    "ValidationCorpusError",
    "ValidationError",
    "ValidationReader",
    "ValidationShardSpec",
    "ValidationWindow",
    "assert_visibility_contract",
    "audit_seed_window",
    "build_intersection_report",
    "build_validation_shard",
    "generate_seed_key",
    "load_b0_game_ids",
    "run_validation_selfplay",
    "verify_shard_integrity",
]
