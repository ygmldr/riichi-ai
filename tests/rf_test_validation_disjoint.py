#!/usr/bin/env python3
"""PR-45 disjointness contracts: fresh window, fresh key, B0/history separation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichiformer.validation.corpus import (  # noqa: E402
    MAX_VALIDATION_GAMES,
    ValidationCorpusError,
    ValidationWindow,
    generate_seed_key,
    validation_game_ids,
)
from riichiformer.validation.disjoint import (  # noqa: E402
    B0_SAMPLE_SEED,
    HISTORICAL_SEED_WINDOWS,
    RESERVED_SEED_KEYS,
    audit_seed_window,
    build_intersection_report,
    game_id_intersection,
    load_b0_game_ids,
)

FAKE_KEY = 0x0F1E2D3C4B5A6978


def test_seed_key_is_fresh_and_not_reserved():
    first = generate_seed_key()
    second = generate_seed_key()
    assert isinstance(first, int) and 0 < first < 2 ** 64
    assert first != second
    assert first not in RESERVED_SEED_KEYS
    assert second not in RESERVED_SEED_KEYS
    assert first not in {0, B0_SAMPLE_SEED}


def test_seed_key_exclusion_is_respected():
    excluded = generate_seed_key()
    for _ in range(8):
        assert generate_seed_key(exclude=[excluded]) != excluded


def test_window_accepts_fresh_window():
    window = ValidationWindow(seed_start=300000, games=100, seed_key=FAKE_KEY)
    assert window.walls == 25
    assert window.window == "[300000,300025)"
    assert window.as_dict()["seed_key_hex"] == f"0x{FAKE_KEY:016x}"


@pytest.mark.parametrize("games", [2, 4.0, MAX_VALIDATION_GAMES + 4, 0])
def test_window_rejects_bad_game_counts(games):
    with pytest.raises(ValidationCorpusError):
        ValidationWindow(seed_start=300000, games=games, seed_key=FAKE_KEY)


def test_window_rejects_consumed_range():
    with pytest.raises(ValidationCorpusError, match="overlaps consumed"):
        ValidationWindow(seed_start=10000, games=4, seed_key=FAKE_KEY)
    with pytest.raises(ValidationCorpusError, match="overlaps consumed"):
        ValidationWindow(seed_start=119999, games=8, seed_key=FAKE_KEY)


def test_window_rejects_reserved_key():
    with pytest.raises(ValidationCorpusError, match="B0 sample seed"):
        ValidationWindow(seed_start=300000, games=4, seed_key=B0_SAMPLE_SEED)


def test_window_rejects_synthetic_ruleset():
    with pytest.raises(ValidationCorpusError):
        ValidationWindow(seed_start=300000, games=4, seed_key=FAKE_KEY,
                         ruleset_id="synthetic_abstract_hanchan/1")


def test_audit_lists_every_overlap():
    audit = audit_seed_window(9999, 8, FAKE_KEY)
    labels = {item["label"] for item in audit["window_overlaps"]}
    assert {"variance_pool_1", "gate_1v3"} <= labels
    assert audit["window_ok"] is False
    clean = audit_seed_window(300000, 100, FAKE_KEY)
    assert clean["window_ok"] is True and clean["seed_key_ok"] is True
    assert clean["window"] == "[300000,300025)"


def test_historical_windows_are_nonempty_and_ordered():
    assert HISTORICAL_SEED_WINDOWS
    for window in HISTORICAL_SEED_WINDOWS:
        assert window.start < window.stop
        assert window.role.startswith("consumed")


def test_b0_sample_seed_is_not_a_window():
    # The B0 sample seed is a sampling seed; using it as a wall key is refused.
    assert B0_SAMPLE_SEED in RESERVED_SEED_KEYS
    assert all(window.start <= B0_SAMPLE_SEED < window.stop
               for window in HISTORICAL_SEED_WINDOWS) is False


def _write_b0_fixture(path: Path, ids) -> Path:
    path.write_text(json.dumps({"game_ids": list(ids)}), encoding="utf-8")
    return path


def test_intersection_report_passes_with_disjoint_fixture(tmp_path):
    b0 = _write_b0_fixture(
        tmp_path / "b0.json",
        ["tenhou:2019091622gm-00a9-0000-316695ea",
         "tenhou:2010063003gm-00a9-0000-e380fe22"])
    game_ids = ["300000:123:0", "300000:123:1", "300000:123:2", "300000:123:3"]
    report = build_intersection_report(
        seed_start=300000, games=4, seed_key=FAKE_KEY,
        validation_game_ids=game_ids,
        generator_sha256="a" * 64, b0_artifact=b0)
    assert report["complete"] is True
    assert report["ok"] is True
    assert report["checks"]["b0_game_ids"]["game_id_intersection"] == 0
    assert report["checks"]["game_id_namespace"]["ok"] is True
    assert report["checks"]["seed_window"]["ok"] is True
    assert report["b0"]["loaded"] is True
    assert report["not_run"] == []


def test_intersection_report_fails_on_b0_collision(tmp_path):
    collision = "tenhou:2019091622gm-00a9-0000-316695ea"
    b0 = _write_b0_fixture(tmp_path / "b0.json", [collision])
    report = build_intersection_report(
        seed_start=300000, games=4, seed_key=FAKE_KEY,
        validation_game_ids=["300000:1:0", collision],
        generator_sha256="b" * 64, b0_artifact=b0)
    assert report["ok"] is False
    assert report["checks"]["b0_game_ids"]["game_id_intersection"] == 1


def test_intersection_report_is_not_run_without_b0(tmp_path):
    report = build_intersection_report(
        seed_start=300000, games=4, seed_key=FAKE_KEY,
        validation_game_ids=["300000:1:0"],
        generator_sha256="c" * 64,
        b0_artifact=tmp_path / "missing.parquet")
    assert report["complete"] is False
    assert report["ok"] is None
    assert "b0_game_ids" in report["not_run"]
    assert report["checks"]["b0_game_ids"]["status"] == "NOT_RUN"


def test_intersection_report_detects_namespace_overlap(tmp_path):
    # A validation corpus that started emitting tenhou: ids must be refused even
    # when the literal id sets do not overlap.
    b0 = _write_b0_fixture(tmp_path / "b0.json", ["tenhou:other"])
    report = build_intersection_report(
        seed_start=300000, games=4, seed_key=FAKE_KEY,
        validation_game_ids=["tenhou:looks-like-a-human-log"],
        generator_sha256="d" * 64, b0_artifact=b0)
    assert report["checks"]["game_id_namespace"]["ok"] is False
    assert report["ok"] is False


def test_game_id_intersection_counts():
    result = game_id_intersection(["a", "b", "c"], ["c", "d"])
    assert result["intersection_count"] == 1
    assert result["intersection_examples"] == ["c"]


def test_load_b0_game_ids_reports_missing(tmp_path):
    loaded = load_b0_game_ids(tmp_path / "nope.parquet")
    assert loaded["loaded"] is False
    assert loaded["ids"] == frozenset()


def test_validation_game_ids_from_engine_logs(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    for rotation in "abcd":
        (games / f"300000_{FAKE_KEY}_{rotation}.json.gz").write_bytes(b"")
    ids = validation_game_ids(games)
    assert ids == [f"300000:{FAKE_KEY}:{index}" for index in range(4)]
