#!/usr/bin/env python3
"""PR-12 shard format, rights, resume, loader and fixture tests."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rf_build_shards as rfbs  # noqa: E402
from riichiformer.data.decision import build_decisions  # noqa: E402
from riichiformer.data.encode import StreamMeta, encode_decision, encoded_hash  # noqa: E402
from riichiformer.data.loader import ShardLoadError, ShardReader  # noqa: E402
from riichiformer.data.shard import (  # noqa: E402
    ACTION_HASH, NORMALIZER_HASH, SCHEMA_HASH, ShardError,
    ShardMetadata,
    ShardRightsError,
    ShardWriter,
    _canonical,
    _sha,
    make_game_record,
    read_manifest,
    sample_hashes,
)
from riichiformer.policy.config import (  # noqa: E402
    POLICY_VERSION, PolicyConfig, PolicyConfigError, load_policy_config,
)
from riichiformer.policy.research_policy import ResearchPolicyError  # noqa: E402


HAND0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p"]
HAND1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N"]
HAND2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
         "7m", "8m", "9m", "1p"]
HAND3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s"]


def main_checkout_root() -> Path:
    """Resolve the main checkout from a linked worktree (never `parent.parent`)."""
    try:
        common = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--git-common-dir"],
            text=True, stderr=subprocess.DEVNULL).strip()
        root = Path(common)
        root = root if root.is_absolute() else (REPO_ROOT / root)
        root = root.resolve().parent
        if (root / "artifacts/riichiformer/h02").exists() or (
                root / "vendor/mortal/target/release").exists():
            return root
    except (OSError, subprocess.SubprocessError):
        pass
    return REPO_ROOT.parent.parent


def raw_four() -> list[dict]:
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1p",
         "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
         "scores": [25000] * 4,
         "tehais": [HAND0, HAND1, HAND2, HAND3]},
        {"type": "tsumo", "actor": 0, "pai": "5mr"},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "1s"},
        {"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "P"},
        {"type": "dahai", "actor": 2, "pai": "P", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "2p"},
        {"type": "dahai", "actor": 3, "pai": "2p", "tsumogiri": True},
        {"type": "end_kyoku"}, {"type": "end_game"},
    ]


def metadata(*, purpose="train", train_allowed=True, rights="ALLOWED",
             source="synthetic_allowed") -> ShardMetadata:
    return ShardMetadata(
        purpose=purpose, source_id=source, rights_status=rights,
        train_allowed=train_allowed,
        canonical_manifest_sha256="a" * 64,
        canonical_index_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        split_version="rf-split/2",
        schema_version="rf-schema-v3/2",
        schema_hash=SCHEMA_HASH,
        action_map_hash=ACTION_HASH,
        normalizer_hash=NORMALIZER_HASH,
    )


def game_record(game_id="g1", split="train", train_allowed=True,
                rights_status="ALLOWED", source="synthetic_allowed"):
    raw = raw_four()
    samples = build_decisions(raw)
    return make_game_record(
        raw, samples, game_id=game_id, component_id="comp-" + game_id,
        component_sha256="d" * 64, split=split, source_id=source,
        train_allowed=train_allowed, rights_status=rights_status)


class ShardRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "train.rfshard"

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, records, meta=None, path=None):
        writer = ShardWriter(path or self.path, meta or metadata())
        for record in records:
            writer.add_game(record)
        return writer.finalize()

    def test_roundtrip_four_povs_one_game_and_split(self):
        manifest = self._write([game_record()])
        self.assertEqual(manifest["game_count"], 1)
        self.assertEqual(manifest["decision_count"], 4)
        reader = ShardReader(self.path, purpose="train")
        games = list(reader.games())
        self.assertEqual(len(games), 1)
        self.assertEqual(len(games[0]["pov_streams"]), 4)
        self.assertEqual(games[0]["split"], "train")
        decisions = list(reader.iter_decisions())
        self.assertEqual({item[4] for item in decisions}, {0, 1, 2, 3})
        self.assertEqual({item[3] for item in decisions}, {"g1"})
        statements = {
            "inputs": [row[1] for row in sqlite3.connect(self.path).execute(
                "PRAGMA table_info(inputs)")],
            "targets": [row[1] for row in sqlite3.connect(self.path).execute(
                "PRAGMA table_info(targets)")],
        }
        self.assertNotIn("expert_action", statements["inputs"])
        self.assertNotIn("rank_final", statements["inputs"])
        self.assertNotIn("tile_state", statements["targets"])
        self.assertNotIn("legal_mask", statements["targets"])

    def test_decision_regenerated_from_stored_pov_stream(self):
        raw = raw_four()
        samples = build_decisions(raw)
        record = game_record()
        self._write([record])
        reader = ShardReader(self.path, purpose="train")
        loaded = list(reader.iter_decisions())
        self.assertEqual(len(loaded), len(samples))
        by_pov = {sample.header.pov: sample for sample in samples}
        for decision, _target, meta, _game, pov, _index in loaded:
            original = by_pov[pov]
            direct = encode_decision(original.input, StreamMeta.from_header(original.header))
            regenerated = encode_decision(decision, meta)
            self.assertEqual(encoded_hash(direct), encoded_hash(regenerated))

    def test_sample_hashes_1_10_100_percent_and_resume(self):
        records = [game_record(f"g{index:02d}") for index in range(20)]
        self._write(records[:5])
        hashes_a = sample_hashes(self.path)
        writer = ShardWriter(self.path, metadata(), resume=True)
        for record in records[5:]:
            writer.add_game(record)
        writer.finalize()
        hashes_b = sample_hashes(self.path)
        self.assertEqual(len(hashes_a), 3)
        # 1/10/100% hashes are deterministic and the 10/100% selections grow.
        self.assertEqual(set(hashes_a), {"0.01", "0.1", "1"})
        self.assertNotEqual(hashes_a["0.1"], hashes_b["0.1"])
        self.assertNotEqual(hashes_a["1"], hashes_b["1"])
        # Idempotent re-add of an existing game must not duplicate decisions.
        before = read_manifest(self.path)["decision_count"]
        writer = ShardWriter(self.path, metadata(), resume=True)
        writer.add_game(records[0])
        manifest = writer.finalize()
        self.assertEqual(manifest["decision_count"], before)

    def test_batches_modelbatch_and_separate_targets(self):
        self._write([game_record(f"g{index}") for index in range(3)])
        reader = ShardReader(self.path, purpose="train")
        batches = list(reader.iter_batches(batch_size=4, bucket_by_length=True))
        self.assertTrue(batches)
        total = 0
        for batch in batches:
            total += len(batch.targets)
            self.assertEqual(batch.batch.n_decisions, len(batch.targets))
            self.assertTrue(batch.batch.padding_mask.any())
            for target in batch.targets:
                self.assertNotIn("tile_state", target)
                self.assertNotIn("legal_mask", target)
        self.assertEqual(total, 12)
        self.assertEqual(sum(len(batch.targets) for batch in batches), 12)


class ShardRightsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "audit.rfshard"

    def tearDown(self):
        self.temp.cleanup()

    def test_blocked_training_metadata_refused(self):
        with self.assertRaises(ShardRightsError):
            metadata(purpose="train", train_allowed=False, rights="UNKNOWN")

    def test_audit_shard_is_explicit_and_train_loader_refuses(self):
        writer = ShardWriter(
            self.path, metadata(purpose="audit", train_allowed=False, rights="UNKNOWN"))
        writer.add_game(game_record())
        writer.finalize()
        reader = ShardReader(self.path, purpose="audit")
        self.assertEqual(len(list(reader.games())), 1)
        with self.assertRaises(ShardRightsError):
            ShardReader(self.path, purpose="train")

    def test_quarantine_never_enters_shard(self):
        record = game_record()
        with self.assertRaises(ShardError):
            dataclasses.replace(record, quarantine=True)

    def test_hash_and_source_mismatch_fail(self):
        writer = ShardWriter(self.path, metadata())
        writer.add_game(game_record())
        writer.finalize()
        with self.assertRaises(ShardLoadError):
            ShardReader(self.path, purpose="train",
                        expected_hashes={"schema_hash": "0" * 64,
                                         "action_map_hash": "1" * 64,
                                         "normalizer_hash": "2" * 64})
        with self.assertRaises(ShardLoadError):
            ShardReader(self.path, purpose="train",
                        expected_source_id="different_source")
        with self.assertRaises(ShardRightsError):
            ShardMetadata(
                purpose="train", source_id="blocked", rights_status="UNKNOWN",
                train_allowed=True,
                canonical_manifest_sha256="a" * 64,
                canonical_index_sha256="b" * 64,
                split_manifest_sha256="c" * 64,
                split_version="rf-split/2",
                schema_version="rf-schema-v3/2",
                schema_hash=SCHEMA_HASH,
                action_map_hash=ACTION_HASH,
                normalizer_hash=NORMALIZER_HASH)



    def test_metadata_requires_explicit_hashes(self):
        with self.assertRaises(TypeError):
            ShardMetadata(
                purpose="train", source_id="synthetic_allowed",
                rights_status="ALLOWED", train_allowed=True,
                canonical_manifest_sha256="a" * 64,
                canonical_index_sha256="b" * 64,
                split_manifest_sha256="c" * 64,
                split_version="rf-split/2")

    def test_synthetic_allowed_registry_training_gate(self):
        import csv
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rf_build_shards_test", REPO_ROOT / "scripts/rf_build_shards.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rights.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["source_id", "train_allowed", "status"])
                writer.writeheader()
                writer.writerow({"source_id": "synthetic_allowed",
                                 "train_allowed": "ALLOWED", "status": "ALLOWED"})
                writer.writerow({"source_id": "blocked", "train_allowed": "RESTRICTED",
                                 "status": "RESTRICTED"})
            rights = module.load_rights(path)
            self.assertTrue(module.source_rights_allowed(rights, "synthetic_allowed"))
            self.assertFalse(module.source_rights_allowed(rights, "blocked"))


def private_policy(sources=("tenhou_old",)) -> PolicyConfig:
    return PolicyConfig(
        profile="operator_private_research",
        operator_reference="unit-test-private-operator",
        allowed_local_sources=frozenset(sources),
        scope="local_nonpublic_only",
    )


def private_metadata(policy: PolicyConfig | None = None, *, purpose="train",
                     rights="RESTRICTED", source="tenhou_old",
                     train_allowed=False) -> ShardMetadata:
    policy = policy or private_policy((source,))
    base = metadata(purpose="audit", train_allowed=False, rights=rights, source=source)
    return dataclasses.replace(
        base, purpose=purpose, train_allowed=train_allowed,
        research_policy="operator_private_research",
        research_policy_hash=policy.policy_hash,
        policy_version=policy.policy_version,
        train_eligible_under_policy=purpose == "train")


class PrivateResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.policy = private_policy()

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, policy=None, *, rights="RESTRICTED", source="tenhou_old",
               name="private.rfshard", train_allowed=False):
        path = self.dir / name
        writer = ShardWriter(
            path, private_metadata(policy or self.policy, rights=rights, source=source))
        writer.add_game(game_record(source=source, train_allowed=train_allowed,
                                    rights_status=rights))
        writer.finalize()
        return path

    def test_private_train_preserves_original_restriction(self):
        path = self._write()
        reader = ShardReader(
            path, purpose="train",
            expected_policy_hash=self.policy.policy_hash,
            allowed_local_sources=self.policy.allowed_local_sources)
        rows = list(reader.games())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rights_status"], "RESTRICTED")
        self.assertFalse(rows[0]["train_allowed"])
        self.assertEqual(reader.metadata.research_policy,
                         "operator_private_research")
        self.assertEqual(reader.metadata.research_policy_hash,
                         self.policy.policy_hash)
        self.assertEqual(reader.metadata.external_license_status, "RESTRICTED")
        self.assertEqual(len(list(reader.iter_decisions())), 4)
        reader.close()

    def test_private_metadata_records_policy_version_and_eligibility(self):
        path = self._write()
        meta = read_manifest(path)["metadata"]
        self.assertEqual(meta["policy_version"], POLICY_VERSION)
        self.assertEqual(meta["research_policy_hash"], self.policy.policy_hash)
        self.assertEqual(meta["external_license_status"], "RESTRICTED")
        self.assertEqual(meta["rights_status"], "RESTRICTED")
        self.assertFalse(meta["train_allowed"])
        self.assertTrue(meta["train_eligible_under_policy"])

    def test_strict_gate_rejects_same_unlicensed_source(self):
        with self.assertRaises(ShardRightsError):
            metadata(purpose="train", train_allowed=False,
                     rights="RESTRICTED", source="tenhou_old")

    def test_private_shard_still_refuses_holdout(self):
        path = self.dir / "private.rfshard"
        writer = ShardWriter(path, private_metadata(self.policy))
        with self.assertRaises(ShardRightsError):
            writer.add_game(game_record(source="tenhou_old", split="val_holdout",
                                        train_allowed=False, rights_status="UNKNOWN"))
        writer.close()

    def test_policy_hash_required(self):
        base = metadata(purpose="audit", train_allowed=False,
                        rights="UNKNOWN", source="tenhou_old")
        with self.assertRaises(ShardError):
            dataclasses.replace(base, purpose="train",
                                research_policy="operator_private_research",
                                research_policy_hash="",
                                train_eligible_under_policy=True)

    def test_private_train_requires_expected_policy_hash(self):
        path = self._write()
        with self.assertRaises(ShardRightsError) as ctx:
            ShardReader(path, purpose="train")
        self.assertIn("expected policy hash", str(ctx.exception))

    def test_private_train_wrong_policy_hash_refused(self):
        path = self._write()
        with self.assertRaises(ShardRightsError) as ctx:
            ShardReader(path, purpose="train", expected_policy_hash="e" * 64)
        self.assertIn("research_policy_hash mismatch", str(ctx.exception))

    def test_private_train_source_outside_allowlist_refused(self):
        path = self._write()
        with self.assertRaises(ShardRightsError) as ctx:
            ShardReader(
                path, purpose="train",
                expected_policy_hash=self.policy.policy_hash,
                allowed_local_sources={"selfplay_own"})
        self.assertIn("outside the expected policy allowlist", str(ctx.exception))

    def test_private_train_eligibility_flag_required(self):
        path = self._write()
        manifest_path = Path(str(path) + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata"]["train_eligible_under_policy"] = False
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ShardRightsError):
            ShardReader(path, purpose="train",
                        expected_policy_hash=self.policy.policy_hash)

    def test_private_training_metadata_requires_eligibility(self):
        base = private_metadata(self.policy, purpose="audit")
        with self.assertRaises(ShardRightsError):
            dataclasses.replace(base, purpose="train",
                                train_eligible_under_policy=False)

    def test_external_license_status_cannot_be_rewritten(self):
        base = private_metadata(self.policy)
        with self.assertRaises(ShardError):
            dataclasses.replace(base, external_license_status="ALLOWED")


class PolicyBuildGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.canonical = self.dir / "canonical"
        self.canonical.mkdir()
        (self.canonical / "manifest.json").write_text(json.dumps({
            "artifacts": {"canonical_log_index.parquet": {"sha256": "a" * 64}}}
        ), encoding="utf-8")
        self.split = self.dir / "split_manifest.parquet"
        self.split.write_bytes(b"split")
        self.rows = [{
            "source_id": "tenhou_old", "rights_status": "RESTRICTED",
            "train_allowed": False, "split_version": "rf-split/2",
        }]

    def tearDown(self):
        self.temp.cleanup()

    def _config(self, *, profile="operator_private_research",
                sources=("tenhou_old",), reference="unit-test-operator"):
        path = self.dir / f"{profile}.json"
        path.write_text(json.dumps({
            "policy_version": POLICY_VERSION,
            "profile": profile,
            "scope": ("local_nonpublic_only"
                      if profile == "operator_private_research"
                      else "external_licensed_only"),
            "operator_reference": reference,
            "allowed_local_sources": list(sources),
            "release_allowed": False,
            "network_upload_allowed": False,
        }), encoding="utf-8")
        return path

    def test_parser_default_is_strict_licensed_only(self):
        args = rfbs.build_parser().parse_args(["--out", "unused"])
        self.assertEqual(args.research_policy, "licensed_only")
        self.assertIsNone(args.policy_config)

    def test_private_cli_requires_policy_config(self):
        code = rfbs.main(["--out", str(self.dir / "x.rfshard"),
                          "--purpose", "train",
                          "--research-policy", "operator_private_research"])
        self.assertEqual(code, 2)
        self.assertFalse((self.dir / "x.rfshard").exists())

    def test_private_build_metadata_without_config_fails(self):
        with self.assertRaises(ResearchPolicyError):
            rfbs.build_metadata(self.rows, self.canonical, self.split, "train",
                                research_policy="operator_private_research")

    def test_policy_allowlist_must_cover_selected_source(self):
        policy = load_policy_config(self._config(sources=("selfplay_own",)))
        with self.assertRaises(ShardRightsError) as ctx:
            rfbs.build_metadata(self.rows, self.canonical, self.split, "train",
                                research_policy="operator_private_research",
                                policy=policy)
        self.assertIn("does not cover", str(ctx.exception))
        self.assertIn("tenhou_old", str(ctx.exception))

    def test_private_build_metadata_records_config_hash(self):
        policy = load_policy_config(self._config())
        meta = rfbs.build_metadata(self.rows, self.canonical, self.split, "train",
                                   research_policy="operator_private_research",
                                   policy=policy)
        self.assertEqual(meta.research_policy_hash, policy.policy_hash)
        self.assertEqual(meta.policy_version, POLICY_VERSION)
        self.assertEqual(meta.rights_status, "RESTRICTED")
        self.assertEqual(meta.external_license_status, "RESTRICTED")
        self.assertFalse(meta.train_allowed)
        self.assertTrue(meta.train_eligible_under_policy)

    def test_cli_rejects_profile_config_mismatch(self):
        config = self._config(profile="licensed_only")
        code = rfbs.main(["--out", str(self.dir / "x.rfshard"),
                          "--purpose", "train",
                          "--research-policy", "operator_private_research",
                          "--policy-config", str(config)])
        self.assertEqual(code, 2)

    def test_cli_missing_policy_config_fails(self):
        code = rfbs.main(["--out", str(self.dir / "x.rfshard"),
                          "--purpose", "train",
                          "--research-policy", "operator_private_research",
                          "--policy-config", str(self.dir / "absent.yaml")])
        self.assertEqual(code, 2)

    def test_cli_private_with_covering_config_gets_past_policy_gate(self):
        # No real canonical DB here, so select_rows/build fails later with exit 1;
        # the important part is that the policy gate (exit 2) is no longer hit.
        config = self._config()
        code = rfbs.main(["--out", str(self.dir / "x.rfshard"),
                          "--purpose", "train", "--limit", "1",
                          "--split-manifest", str(self.dir / "absent.parquet"),
                          "--research-policy", "operator_private_research",
                          "--policy-config", str(config)])
        self.assertEqual(code, 1)


class FixtureTests(unittest.TestCase):
    def test_100k_split_fixture_dry_run_and_audit_build(self):
        fixture_root = main_checkout_root() / "artifacts/riichiformer/h02"
        canonical = fixture_root / "bench_100k"
        split = fixture_root / "split_100k/split_manifest_L_v2.parquet"
        if not canonical.is_dir() or not split.is_file():
            self.skipTest("no 100k H-02 fixture")
        manifest = json.loads((canonical / "manifest.json").read_text())
        self.assertEqual(manifest["representatives"], 100000)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "audit.rfshard"
            dry = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts/rf_build_shards.py"),
                 "--out", str(out), "--purpose", "diagnostic", "--limit", "3",
                 "--canonical-dir", str(canonical),
                 "--split-manifest", str(split), "--dry-run", "--json"],
                env={**os_environ(), "PYTHONPATH": str(vendor_path())},
                capture_output=True, text=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertFalse(out.exists())
            build = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts/rf_build_shards.py"),
                 "--out", str(out), "--purpose", "diagnostic", "--limit", "3",
                 "--canonical-dir", str(canonical),
                 "--split-manifest", str(split), "--write"],
                env={**os_environ(), "PYTHONPATH": str(vendor_path())},
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            self.assertTrue(out.exists())
            reader = ShardReader(out, purpose="audit")
            meta = reader.metadata
            self.assertEqual(meta.purpose, "diagnostic")
            self.assertFalse(meta.train_allowed)
            games = list(reader.games())
            self.assertGreaterEqual(len(games), 1)
            self.assertTrue(all(game["split"] in {"train", "val_holdout",
                                                  "external_diagnostic"}
                                for game in games))
            # Training path must refuse this blocked fixture.
            refused = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts/rf_build_shards.py"),
                 "--out", str(Path(temp) / "train.rfshard"), "--purpose", "train",
                 "--limit", "1", "--canonical-dir", str(canonical),
                 "--split-manifest", str(split), "--research-policy", "licensed_only"],
                env={**os_environ(), "PYTHONPATH": str(vendor_path())},
                capture_output=True, text=True)
            self.assertEqual(refused.returncode, 2, refused.stderr)


class BuildSelectionTests(unittest.TestCase):
    def _row(self, game_id, *, split="train", component_size=1,
             is_representative=True, component_id=None, source="synthetic_allowed",
             train_allowed=True, rights="ALLOWED"):
        return {
            "protocol": "L", "is_representative": is_representative,
            "quarantine": False, "component_size": component_size,
            "train_allowed": train_allowed, "rights_status": rights,
            "source_id": source, "game_id": game_id,
            "component_id": component_id or ("comp-" + game_id),
            "canonical_sha256": "a" * 64, "root": "/tmp", "relpath": game_id,
            "split": split, "split_version": "rf-split/2",
        }

    def _manifest(self, tmp: Path, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = tmp / "split.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def test_write_root_guard_refuses_datasets_before_writer(self):
        for name in ("datasets", "runs", "checkpoints"):
            with self.assertRaises(ShardRightsError):
                rfbs.assert_safe_shard_out(REPO_ROOT / name / "__review", REPO_ROOT)
        main = main_checkout_root()
        with self.assertRaises(ShardRightsError):
            rfbs.assert_safe_shard_out(
                main / "artifacts/riichiformer/split", REPO_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            rfbs.assert_safe_shard_out(Path(tmp) / "ok.rfshard", REPO_ROOT)
        target = REPO_ROOT / "datasets" / "__review_guard"
        code = rfbs.main(["--out", str(target)])
        self.assertEqual(code, 2)
        self.assertFalse(target.exists())

    def test_two_file_component_representative_selected_once(self):
        rows = [
            self._row("g_rep", component_size=2, component_id="c1"),
            self._row("g_dup", component_size=2, component_id="c1",
                      is_representative=False),
            self._row("g_single"),
        ]
        rights = {"synthetic_allowed": {
            "source_id": "synthetic_allowed", "train_allowed": "ALLOWED",
            "status": "ALLOWED"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(Path(tmp), rows)
            diagnostic = rfbs.select_rows(path, "L", 10, "diagnostic", rights)
            self.assertEqual([row["game_id"] for row in diagnostic],
                             ["g_rep", "g_single"])
            rep = next(row for row in diagnostic if row["game_id"] == "g_rep")
            self.assertEqual(rep["component_size"], 2)
            self.assertTrue(rep["is_representative"])
            train = rfbs.select_rows(path, "L", 10, "train", rights)
            self.assertEqual([row["game_id"] for row in train],
                             ["g_rep", "g_single"])

    def test_train_purpose_rejects_val_holdout_row(self):
        rights = {"synthetic_allowed": {
            "source_id": "synthetic_allowed", "train_allowed": "ALLOWED",
            "status": "ALLOWED"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(Path(tmp), [
                self._row("g_train", split="train"),
                self._row("g_val", split="val_holdout"),
            ])
            with self.assertRaises(ShardRightsError):
                rfbs.select_rows(path, "L", 10, "train", rights)
            # Diagnostic purpose may still audit both partitions.
            rows = rfbs.select_rows(path, "L", 10, "diagnostic", rights)
            self.assertEqual([row["game_id"] for row in rows], ["g_train", "g_val"])


class P1ShardGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _train_shard(self, name="train.rfshard"):
        path = self.dir / name
        writer = ShardWriter(path, metadata())
        writer.add_game(game_record())
        writer.finalize()
        return path

    def test_writer_rejects_val_split_under_train_metadata(self):
        path = self.dir / "reject.rfshard"
        writer = ShardWriter(path, metadata())
        with self.assertRaises(ShardRightsError):
            writer.add_game(game_record("valgame", split="val_holdout"))
        writer.close()
        self.assertFalse(path.exists())

    def test_writer_rejects_per_game_blocked_rights(self):
        path = self.dir / "reject2.rfshard"
        writer = ShardWriter(path, metadata())
        with self.assertRaises(ShardRightsError):
            writer.add_game(game_record("blockedgame", train_allowed=False,
                                        rights_status="UNKNOWN"))
        writer.close()
        self.assertFalse(path.exists())

    def test_loader_rejects_tampered_val_split(self):
        path = self._train_shard()
        conn = sqlite3.connect(path)
        conn.execute("UPDATE games SET split='val_holdout'")
        conn.commit()
        conn.close()
        reader = ShardReader(path, purpose="train", verify_file_hash=False)
        with self.assertRaises(ShardRightsError):
            list(reader.games())

    def test_loader_rejects_tampered_per_game_rights(self):
        path = self._train_shard("rights.rfshard")
        conn = sqlite3.connect(path)
        conn.execute("UPDATE games SET train_allowed=0, rights_status='UNKNOWN'")
        conn.commit()
        conn.close()
        reader = ShardReader(path, purpose="train", verify_file_hash=False)
        with self.assertRaises(ShardRightsError):
            list(reader.games())

    def test_loader_rejects_tampered_source(self):
        path = self._train_shard("source.rfshard")
        conn = sqlite3.connect(path)
        conn.execute("UPDATE games SET source_id='tenhou_old'")
        conn.commit()
        conn.close()
        reader = ShardReader(path, purpose="train", verify_file_hash=False)
        with self.assertRaises(ShardRightsError):
            list(reader.games())

    def test_loader_hashes_without_read_bytes(self):
        path = self._train_shard()
        with mock.patch.object(Path, "read_bytes",
                               side_effect=AssertionError("read_bytes used")):
            reader = ShardReader(path, purpose="train")
            self.assertEqual(len(list(reader.games())), 1)

    def test_loader_pins_canonical_split_and_component_hashes(self):
        path = self._train_shard()
        metadata_dict = read_manifest(path)["metadata"]
        expected = {
            "schema_hash": SCHEMA_HASH,
            "action_map_hash": ACTION_HASH,
            "normalizer_hash": NORMALIZER_HASH,
            "canonical_manifest_sha256": metadata_dict["canonical_manifest_sha256"],
            "canonical_index_sha256": metadata_dict["canonical_index_sha256"],
            "split_manifest_sha256": metadata_dict["split_manifest_sha256"],
        }
        ShardReader(path, purpose="train", expected_hashes=expected,
                    expected_component_sha256={"g1": "d" * 64})
        bad = dict(expected, split_manifest_sha256="0" * 64)
        with self.assertRaises(ShardLoadError):
            ShardReader(path, purpose="train", expected_hashes=bad)
        with self.assertRaises(ShardLoadError):
            reader = ShardReader(path, purpose="train",
                                 expected_component_sha256={"g1": "0" * 64})
            list(reader.games())

    def test_content_sha_is_reproducible_across_created_utc(self):
        meta_a = metadata()
        meta_b = dataclasses.replace(meta_a, created_utc="2020-01-01T00:00:00Z")
        path_a = self.dir / "a.rfshard"
        path_b = self.dir / "b.rfshard"
        for path, meta in ((path_a, meta_a), (path_b, meta_b)):
            writer = ShardWriter(path, meta)
            writer.add_game(game_record())
            writer.finalize()
        manifest_a = read_manifest(path_a)
        manifest_b = read_manifest(path_b)
        self.assertEqual(manifest_a["content_sha256"], manifest_b["content_sha256"])
        self.assertNotEqual(manifest_a["shard_sha256"], manifest_b["shard_sha256"])

    def test_sample_hashes_streaming_matches_reference(self):
        records = [game_record(f"g{i}") for i in range(3)]
        path = self.dir / "samples.rfshard"
        writer = ShardWriter(path, metadata())
        for record in records:
            writer.add_game(record)
        writer.finalize()
        conn = sqlite3.connect(path)
        rows = conn.execute(
            "SELECT i.decision_id, i.input_sha256, t.target_sha256 "
            "FROM inputs i JOIN targets t USING(decision_id) "
            "ORDER BY i.decision_id").fetchall()
        conn.close()
        actual = sample_hashes(path)
        for fraction in (0.01, 0.10, 1.0):
            n = max(1, int(round(len(rows) * fraction)))
            expected = _sha(_canonical([
                {"decision_id": row[0], "input_sha256": row[1],
                 "target_sha256": row[2]} for row in rows[:n]
            ]))
            self.assertEqual(actual[f"{fraction:g}"], expected)


def os_environ():
    import os
    return os.environ.copy()


def vendor_path() -> Path:
    return main_checkout_root() / "vendor/mortal/target/release"


if __name__ == "__main__":
    unittest.main(verbosity=2)
