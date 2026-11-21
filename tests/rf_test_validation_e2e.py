#!/usr/bin/env python3
"""PR-45/PR-47 end-to-end validation tests (native CPU smoke).

One 4-game native corpus is played once per session and sealed into a diagnostic
shard.  The tests then exercise the read-only reader, the metrics, the DB-level
negative controls, the POV visibility contract, the future-event causal control,
and the full PR-47 roundtrip (shard -> multi loader -> 2-step six-layer CPU train
-> checkpoint -> reload -> inference).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.encode import (  # noqa: E402
    StreamMeta,
    _state_snapshot,
    encode_multi_decision_batch,
)
from riichiformer.data.loader import (  # noqa: E402
    ShardLoadError,
    ShardReader,
    _decision_payload,
    _global,
    _player,
    _tile_state,
)
from riichiformer.data.multi_decision import DecisionMarker, MultiDecisionStream  # noqa: E402
from riichiformer.data.shard import ShardRightsError, read_manifest, unpack_json  # noqa: E402
from riichiformer.models.policy import Policy  # noqa: E402
from riichiformer.models.state_encoder import state_features_from_snapshot  # noqa: E402
from riichiformer.policy.config import load_policy_config  # noqa: E402
from riichiformer.schemas.action_v3 import ActionPhase  # noqa: E402
from riichiformer.schemas.event_v3 import event_from_dict  # noqa: E402
from riichiformer.train.model_config import resolve_mvp_config  # noqa: E402
from riichiformer.validation.corpus import (  # noqa: E402
    generate_seed_key,
    run_validation_selfplay,
)
from riichiformer.validation.disjoint import build_intersection_report  # noqa: E402
from riichiformer.validation.e2e import (  # noqa: E402
    run_roundtrip,
    tamper_cutoff,
    tamper_public_state,
    tamper_target_label,
)
from riichiformer.validation.integrity import (  # noqa: E402
    IntegrityError,
    VisibilityError,
    assert_visibility_contract,
    verify_shard_integrity,
)
from riichiformer.validation.reader import (  # noqa: E402
    ValidationError,
    ValidationReader,
)
from riichiformer.validation.shard_build import (  # noqa: E402
    ARENA_DEVIATIONS,
    ValidationShardSpec,
    build_validation_shard,
)

POLICY_CONFIG = REPO_ROOT / "configs/riichiformer/local_research_policy.example.yaml"


def _libriichi_available() -> bool:
    try:
        import libriichi  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    if not _libriichi_available():
        pytest.skip("pinned libriichi is required for native validation tests")
    root = tmp_path_factory.mktemp("pr45_validation")
    window_seed = 300200
    window_key = generate_seed_key()
    out = root / "corpus"
    from riichiformer.validation.corpus import ValidationWindow

    window = ValidationWindow(seed_start=window_seed, games=4, seed_key=window_key)
    corpus_report = run_validation_selfplay(out, window)
    policy = load_policy_config(POLICY_CONFIG)
    manifest = json.loads((out / "validation_corpus.json").read_text(encoding="utf-8"))
    spec = ValidationShardSpec(
        seed_start=window.seed_start, games=window.games, seed_key=window.seed_key,
        lineup=tuple(manifest["lineup"]),
        generator_sha256=manifest["generator_sha256"])
    shard = root / "validation.rfshard"
    build = build_validation_shard(
        games_dir=out / "games", out_shard=shard, spec=spec, policy=policy)
    return SimpleNamespace(root=root, out=out, corpus=corpus_report, manifest=manifest,
                           spec=spec, policy=policy, shard=shard, build=build,
                           window=window)


def test_corpus_and_shard_provenance(corpus):
    build = corpus.build
    metadata = build["metadata"]
    assert build["game_count"] == 4
    assert build["decision_count"] > 0
    assert metadata["purpose"] == "diagnostic"
    assert metadata["source_kind"] == "arena_selfplay"
    assert metadata["ruleset_id"] == "mortal_arena_pinned"
    assert metadata["train_allowed"] is False
    assert metadata["train_eligible_under_policy"] is False
    assert metadata["research_policy"] == "operator_private_research"
    assert len(metadata["research_policy_hash"]) == 64
    assert set(ARENA_DEVIATIONS) <= set(metadata["ruleset_deviations"])
    assert "R29" in metadata["ruleset_deviations"]
    assert build["integrity"]["ok"] is True
    assert build["visibility"]["violation_count"] == 0
    assert build["visibility"]["unknown_on_hidden_events"] > 0
    assert all(game["score_sum"] == 100_000 for game in build["games"])
    assert build["corpus_manifest_sha256"] == metadata["canonical_manifest_sha256"]
    assert build["canonical_index_sha256"] == metadata["canonical_index_sha256"]
    assert build["split_manifest_sha256"] == metadata["split_manifest_sha256"]
    manifest = read_manifest(corpus.shard)
    assert manifest["game_count"] == 4
    assert manifest["metadata"]["purpose"] == "diagnostic"


def test_training_loader_refuses_diagnostic_shard(corpus):
    with pytest.raises((ShardRightsError, ShardLoadError)):
        ShardReader(corpus.shard, purpose="train",
                    expected_policy_hash=corpus.policy.policy_hash)


def test_validation_reader_is_forward_only(corpus):
    reader = ValidationReader(corpus.shard,
                              expected_policy_hash=corpus.policy.policy_hash,
                              batch_size=4)
    try:
        describe = reader.describe()
        assert describe["purpose"] == "diagnostic"
        assert describe["train_eligible_under_policy"] is False
        assert describe["integrity_ok"] is True
        forward = next(reader.iter_forward_inputs(limit=8))
        assert not hasattr(forward, "targets")
        assert int(forward.decision_cutoff_idx.numel()) > 0
        batch = next(reader.iter_batches(limit=8))
        assert batch.decisions == len(batch.batch.targets)
        assert len(batch.contexts) == batch.decisions
        for target in batch.read_only_targets:
            with pytest.raises(TypeError):
                target["expert_action"] = 0
        with pytest.raises(ValidationError, match="read-only"):
            reader.optimizer_update_forbidden()
    finally:
        reader.close()


def test_validation_reader_evaluate_and_parity(corpus):
    resolved = resolve_mvp_config()
    model = Policy(resolved.config)
    reader = ValidationReader(corpus.shard,
                              expected_policy_hash=corpus.policy.policy_hash,
                              batch_size=4, max_positions=resolved.config.max_positions)
    try:
        model.train()
        result = reader.evaluate(model, limit=16, prefix_parity=True,
                                 parity_max_decisions=8)
        assert model.training is True  # mode restored
        assert all(parameter.grad is None for parameter in model.parameters())
        assert result["supervised"] > 0
        assert result["illegal_rate"] == 0.0
        assert result["nonfinite_rate"] == 0.0
        assert result["phase_partition_sum"] == result["supervised"]
        assert result["action_family_sum"] == result["supervised"]
        assert result["sequence_coverage"]["truncated"] is False
        assert result["reader"]["read_only"] is True
        assert result["reader"]["optimizer_steps"] == 0
        parity = result["prefix_multi_parity"]
        assert parity["ok"] is True
        assert parity["logit_max_abs"] <= 1e-5
    finally:
        reader.close()


def test_target_label_tamper_is_detected(corpus, tmp_path):
    tampered = tamper_target_label(corpus.shard, tmp_path / "label.rfshard")
    with pytest.raises(IntegrityError, match="target_hash"):
        ValidationReader(tampered, expected_policy_hash=corpus.policy.policy_hash,
                         verify_integrity=True)


def test_public_state_tamper_is_refused_by_loader(corpus, tmp_path):
    tampered = tamper_public_state(corpus.shard, tmp_path / "state.rfshard")
    reader = ValidationReader(tampered,
                              expected_policy_hash=corpus.policy.policy_hash,
                              batch_size=4)
    try:
        with pytest.raises(ShardLoadError):
            for _ in reader.iter_batches():
                pass
    finally:
        reader.close()


def test_cutoff_tamper_is_refused_by_loader(corpus, tmp_path):
    tampered = tamper_cutoff(corpus.shard, tmp_path / "cutoff.rfshard")
    reader = ValidationReader(tampered,
                              expected_policy_hash=corpus.policy.policy_hash,
                              batch_size=4)
    try:
        with pytest.raises(ShardLoadError):
            for _ in reader.iter_batches():
                pass
    finally:
        reader.close()


def test_visibility_contract_only_unknown_on_hidden_events(corpus):
    conn = sqlite3.connect(f"file:{corpus.shard}?mode=ro", uri=True)
    try:
        rows = list(conn.execute("SELECT game_id, pov_streams FROM games"))
    finally:
        conn.close()
    assert rows
    total_unknown = 0
    for game_id, packed in rows:
        streams = unpack_json(packed)
        report = assert_visibility_contract(streams, game_id=str(game_id))
        assert report["ok"] is True
        assert report["violation_count"] == 0
        assert report["unknown_on_hidden_events"] > 0
        assert report["unknown_tiles"] == (report["unknown_on_hidden_events"]
                                           + report["unknown_padding_slots"])
        total_unknown += report["unknown_tiles"]
    assert total_unknown > 0
    # Negative control: UNKNOWN on a public discard must be refused.
    fake = [[{"type": "dahai", "seq_idx": 0, "kyoku_idx": 0,
              "timing": "turn", "actor_rel": 1,
              "tile": {"kind": 34, "aka": None}, "tsumogiri": False}]] * 4
    with pytest.raises(VisibilityError):
        assert_visibility_contract(fake, game_id="fake")


def _load_stream(corpus, game_id: str, pov: int, *, max_markers: int = 4):
    conn = sqlite3.connect(f"file:{corpus.shard}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        game = conn.execute("SELECT * FROM games WHERE game_id=?",
                            (game_id,)).fetchone()
        streams = unpack_json(game["pov_streams"])
        events = tuple(event_from_dict(item) for item in streams[pov])
        rows = conn.execute(
            "SELECT * FROM inputs WHERE game_id=? AND pov=? "
            "ORDER BY decision_index LIMIT ?",
            (game_id, pov, max_markers)).fetchall()
        markers = []
        last_cutoff = 0
        for row in rows:
            payload = _decision_payload(row)
            cutoff = int(payload["before_event_idx"])
            tile_state = _tile_state(payload["tile_state"])
            players = tuple(_player(player) for player in payload["players_public"])
            public = _global(payload["global_public"])
            phase = ActionPhase(payload["phase"])
            legal = tuple(bool(value) for value in payload["legal_mask"])
            snapshot = SimpleNamespace(
                events=events[:cutoff], tile_state=tile_state,
                players_public=players, global_public=public, phase=phase,
                legal_mask=legal)
            features = state_features_from_snapshot(_state_snapshot(snapshot))
            markers.append(DecisionMarker(
                cutoff_idx=cutoff, phase=phase, legal_mask=legal,
                state_features=features))
            last_cutoff = cutoff
    finally:
        conn.close()
    metadata = read_manifest(corpus.shard)["metadata"]
    meta = StreamMeta(
        pov=pov, projection_version=metadata["projection_version"],
        schema_version=metadata["schema_version"], schema_hash=metadata["schema_hash"],
        ruleset_id=metadata["ruleset_id"], source_kind=metadata["source_kind"])
    prefix = events[:last_cutoff]
    return prefix, tuple(markers), meta


def test_future_event_does_not_change_earlier_decision(corpus):
    with sqlite3.connect(f"file:{corpus.shard}?mode=ro", uri=True) as conn:
        game_id = conn.execute("SELECT game_id FROM games ORDER BY game_id LIMIT 1"
                               ).fetchone()[0]
    prefix, markers, meta = _load_stream(corpus, game_id, 0, max_markers=4)
    cutoff = markers[0].cutoff_idx
    future_index = None
    for index in range(cutoff + 1, len(prefix)):
        event = prefix[index]
        if event.EVENT_TYPE == "dahai":
            future_index = index
            break
    if future_index is None:
        pytest.skip("no public future event after the first decision in this game")
    altered = list(prefix)
    original = altered[future_index]
    new_kind = 1 if int(original.tile.kind) != 1 else 2
    from dataclasses import replace

    from riichiformer.schemas.event_v3 import TileRef

    altered[future_index] = replace(original, tile=TileRef(new_kind))
    stream_a = MultiDecisionStream(events=prefix, decisions=markers, meta=meta)
    stream_b = MultiDecisionStream(events=tuple(altered), decisions=markers, meta=meta)
    batch_a = encode_multi_decision_batch([stream_a], max_events=4096,
                                          verify_prefix="exact")
    batch_b = encode_multi_decision_batch([stream_b], max_events=4096,
                                          verify_prefix="exact")
    # The marker before the altered event is causally identical.
    assert torch.equal(batch_a.state_features[0], batch_b.state_features[0])
    assert torch.equal(batch_a.legal_mask[0], batch_b.legal_mask[0])
    model = Policy(resolve_mvp_config().config)
    model.eval()
    with torch.no_grad():
        logits_a = model.forward_multi(batch_a).logits[0]
        logits_b = model.forward_multi(batch_b).logits[0]
    assert torch.equal(logits_a, logits_b)


def test_pr47_roundtrip(corpus):
    result = run_roundtrip(
        games_dir=corpus.out / "games",
        shard_path=corpus.root / "roundtrip.rfshard",
        workdir=corpus.root / "roundtrip_work",
        policy=corpus.policy, spec=corpus.spec, steps=2, batch_size=2)
    assert result["passed"] is True
    assert result["games"] == 4
    assert result["parameter_count"] == 5_217_582
    assert result["steps"] == 2 and len(result["step_losses"]) == 2
    assert all(loss == loss and abs(loss) != float("inf")
               for loss in result["step_losses"])
    assert result["reload_logit_max_abs"] == 0.0
    assert result["training_loader_refusal"] in ("ShardRightsError", "ShardLoadError")
    assert result["gpu_used"] is False
    assert Path(result["checkpoint"]).exists()
    build = result["build"]
    assert build["integrity"]["ok"] is True
    assert build["visibility"]["violation_count"] == 0


def test_intersection_against_local_b0_if_present(corpus):
    b0 = Path("/tmp/b0_sample500.parquet")
    if not b0.exists():
        pytest.skip("local B0 sample artifact is not present on this host")
    report = build_intersection_report(
        seed_start=corpus.window.seed_start, games=corpus.window.games,
        seed_key=corpus.window.seed_key,
        validation_game_ids=[game["game_id"] for game in corpus.build["games"]],
        generator_sha256=corpus.spec.generator_sha256, b0_artifact=b0)
    assert report["complete"] is True
    assert report["ok"] is True
    assert report["checks"]["b0_game_ids"]["game_id_intersection"] == 0
    assert report["checks"]["seed_window"]["ok"] is True
    assert report["checks"]["seed_key"]["ok"] is True


def test_integrity_summary_matches_builder(corpus):
    integrity = verify_shard_integrity(corpus.shard)
    assert integrity["ok"] is True
    assert integrity["inputs"] == corpus.build["decision_count"]
    assert integrity["targets"] == corpus.build["decision_count"]
