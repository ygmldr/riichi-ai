#!/usr/bin/env python3
"""t11: `own_state` provider 的有效性 + 覆盖率(冻结夹具逐位相等 + report_sub 样本)。

判定口径**独立于 provider 实现**,直接读 obs 切片(`layout.span` 查表):可读 = 块非全零;
布局 A(`can_discard`)= +68 指向的弃牌候选的 +t/+34+t 平面;布局 B(`!can_discard`)= +69
单候选平面(obs_layout.json 的 "+68" 为文档差一);全零块 ⇒ 必须 unavailable + 非空 reason。
产物 `runs/exp/EX1/own_state_validity.json`:覆盖率 = 可读数/总决策数 + PREREG sha256。
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest
from datetime import datetime, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from exp_ex1_audit import PREREG_PATH, assert_prereg_unchanged, load_prereg  # noqa: E402

OUT_PATH = REPO_ROOT / "runs/exp/EX1/own_state_validity.json"
CORPUS_INDEX = REPO_ROOT / "runs/exp/EX1/corpus_index.json"
LAYOUT_PATH = REPO_ROOT / "runs/analysis/obs_layout.json"
FIXTURE_REL = "runs/1v3/10219_579411421376059130_d.json.gz"
REPORT_SAMPLE_FILES = 8
TRIAGE_BAR = 0.90          # PREREG §6.2 F1 triage: coverage >= 0.90
DELIVERY_BAR = 0.95        # PREREG §6.2 F1 delivery: coverage >= 0.95
TILE_KINDS = 34
SHANTEN_CAP = 6
MAX_REQUIRED_TILE_PLANE = 68
DRAW_REQUIRED_TILE_PLANE = 69


def _fixture_decisions():
    from jiangpu.record import read_record
    from jiangpu.replay import iter_decisions
    return iter_decisions(read_record(str(REPO_ROOT / FIXTURE_REL)))


def _report_sample_groups():
    """report_sub 前 N 份的 subject 席位决策,**按文件分组**:`iter_decisions` 的 `index`
    只在单份牌谱内唯一,跨文件拼接会重号覆盖 ⇒ 逐份 compute + 逐份审计。"""
    if not CORPUS_INDEX.exists():
        return [], [], "corpus_index.json 缺失:跳过 report_sub 样本(仅冻结夹具)"
    from eval_report import SEAT_OF_SPLIT
    from jiangpu.record import read_record
    from jiangpu.replay import iter_decisions

    idx = json.loads(CORPUS_INDEX.read_text(encoding="utf-8"))
    files = sorted(REPO_ROOT / f["target_path"] for f in idx["subsets"]["report_sub"]["files"])
    files = files[:REPORT_SAMPLE_FILES]
    groups = []
    for path in files:
        record = read_record(str(path))
        groups.append((str(path.relative_to(REPO_ROOT)),
                       iter_decisions(record, pov=SEAT_OF_SPLIT[record.split])))
    return groups, [rel for rel, _ in groups], None


def _expected_from_obs(dec, spans):
    """独立重算(只读 obs):返回 (期望字段, None) 或 (None, unavailable 类别)。"""
    sh_off, sh_count = spans["shanten"]
    rt_off, rt_count = spans["sp_required_tiles"]
    sh_lit = [i for i in range(sh_count) if float(dec.obs[sh_off + i].max()) > 0.0]
    if len(sh_lit) != 1:
        return None, "shanten_not_one_hot"
    rt = dec.obs[rt_off:rt_off + rt_count]
    if float(rt.max()) == 0.0:
        return None, "all_zero"
    lit = [i for i in range(rt_count) if float(rt[i].max()) > 0.0]
    if lit == [DRAW_REQUIRED_TILE_PLANE]:
        plane = rt[DRAW_REQUIRED_TILE_PLANE]
        return {"layout": "draw", "subject_tile": None, "shanten": sh_lit[0],
                "ukeire_tiles": [i for i in range(TILE_KINDS) if float(plane[i]) > 0.0]}, None
    if MAX_REQUIRED_TILE_PLANE in lit:
        subject = max(range(TILE_KINDS), key=lambda i: float(rt[MAX_REQUIRED_TILE_PLANE][i]))
        for plane in (rt[subject], rt[TILE_KINDS + subject]):
            tiles = [i for i in range(TILE_KINDS) if float(plane[i]) > 0.0]
            if tiles:
                return {"layout": "discard", "subject_tile": subject, "shanten": sh_lit[0],
                        "ukeire_tiles": tiles}, None
        return None, "candidate_plane_all_zero"
    return None, "unknown_layout"


def _audit(decisions, payloads, spans):
    """逐决策比对 payload 与独立重算,汇总覆盖率/布局/未编码理由。"""
    stats = _empty_stats()
    for dec in decisions:
        stats["decisions"] += 1
        payload = payloads.get(dec.index)
        if payload is None:
            stats["missing_payloads"] += 1
            continue
        expected, category = _expected_from_obs(dec, spans)
        if expected is None:
            stats["unavailable"] += 1
            if payload.get("unavailable") is not True:
                stats["mismatches"] += 1
                continue
            reason = str(payload.get("reason", ""))
            if not reason:
                stats["unavailable_without_reason"] += 1
            entry = stats["unavailable_reasons"].setdefault(
                category, {"count": 0, "sample_reason": reason})
            entry["count"] += 1
            continue
        stats["readable"] += 1
        stats["layouts"][expected["layout"]] += 1
        if payload.get("unavailable"):
            stats["mismatches"] += 1
            continue
        same = (payload.get("layout") == expected["layout"]
                and payload.get("subject_tile") == expected["subject_tile"]
                and payload.get("shanten") == expected["shanten"]
                and payload.get("ukeire_tiles") == expected["ukeire_tiles"]
                and payload.get("ukeire_kinds") == len(expected["ukeire_tiles"]))
        stats["mismatches"] += int(not same)
        if payload.get("ukeire_kinds", 0) <= 0 or not payload.get("ukeire_tiles"):
            stats["readable_with_empty_ukeire"] += 1
    return stats


def _merge_stats(into, part):
    """把逐份牌谱的审计结果合并进总计(样本按文件分组审计,见 `_report_sample_groups`)。"""
    for key in ("decisions", "readable", "unavailable", "missing_payloads", "mismatches",
                "readable_with_empty_ukeire", "unavailable_without_reason"):
        into[key] += part[key]
    for kind in ("discard", "draw"):
        into["layouts"][kind] += part["layouts"][kind]
    for category, entry in part["unavailable_reasons"].items():
        target = into["unavailable_reasons"].setdefault(
            category, {"count": 0, "sample_reason": entry["sample_reason"]})
        target["count"] += entry["count"]
    return into


def _empty_stats():
    return {"files": [], "decisions": 0, "readable": 0, "unavailable": 0,
            "missing_payloads": 0, "mismatches": 0, "readable_with_empty_ukeire": 0,
            "unavailable_without_reason": 0, "layouts": {"discard": 0, "draw": 0},
            "unavailable_reasons": {}}


def build_artifact(populations, prereg_sha, spans):
    total = sum(p["decisions"] for p in populations.values())
    readable = sum(p["readable"] for p in populations.values())
    unavailable = sum(p["unavailable"] for p in populations.values())
    ratio = readable / total if total else None
    return {
        "schema": "ex1-own-state-validity/v1",
        "todo": "t11",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": {"name": "own_state", "provider_version": 1, "unit": "count",
                     "unit_by_field": {"shanten": "shanten_level", "ukeire_kinds": "count",
                                       "ukeire_tiles": "tile_index"},
                     "basis": "self_hand", "no_hidden_info": True, "checkpoint_sha256": None},
        "spans": {name: list(span) for name, span in spans.items()},
        "layout_semantics": {
            "discard": "+68 = max-required-tiles discard candidate; read its +t / +34+t plane",
            "draw": "!can_discard: +69 = single draw candidate's required tiles ('+68' doc off by one)",
            "unavailable": "all-zero block: SP tables not encoded (tiles_left<4 / no tsumo / agari hand)",
        },
        "populations": populations,
        "coverage": {"decisions": total, "readable": readable, "unavailable": unavailable,
                     "ratio": ratio, "triage_bar": TRIAGE_BAR, "delivery_bar": DELIVERY_BAR,
                     "triage_met": ratio is not None and ratio >= TRIAGE_BAR,
                     "delivery_met": ratio is not None and ratio >= DELIVERY_BAR},
        "total_mismatches": sum(p["mismatches"] for p in populations.values()),
        "unavailable_all_have_reason": all(p["unavailable_without_reason"] == 0
                                           for p in populations.values()),
        "report_sub_sample_note": populations["report_sub_sample"].get("note"),
        "prereg": {"path": "scripts/exp_ex1_prereg.md", "sha256": prereg_sha},
    }


def _minimal_doc(payload):
    return {
        "schema_version": 2, "engine_version": 1,
        "checkpoint": {"path": "x.pth", "sha256": "a" * 64, "steps": 396000},
        "record": {"path": "p.json.gz", "source": "frozen", "names": ["a"] * 4,
                   "seed": 1, "key": 2, "split": "a"},
        "decisions": [{"index": 0, "player_id": 0, "at_kyoku": 0, "at_turn": 1, "shanten": 3,
                       "logged_action": 5, "legal_actions": [5, 6],
                       "metrics": {"own_state": payload}}],
        "kyoku_summaries": [], "game_summary": {}, "key_moments": [], "unavailable": [],
    }


class OwnStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prereg_sha = load_prereg(PREREG_PATH)
        missing = [p for p in (LAYOUT_PATH, REPO_ROOT / FIXTURE_REL) if not p.exists()]
        if missing:
            raise unittest.SkipTest("own_state fixtures missing: "
                                    + ", ".join(str(p) for p in missing))
        from jiangpu import layout
        from jiangpu.providers import own_state

        cls.own_state = own_state
        cls.top = layout.load_layout()
        cls.spans = {name: layout.span(cls.top, name) for name in
                     ("shanten", "sp_required_tiles", "discard_candidates", "tiles_left")}
        cls.fixture_decisions = _fixture_decisions()
        cls.fixture_payloads = own_state.compute(cls.fixture_decisions)
        cls.sample_groups, cls.sample_files, cls.sample_note = _report_sample_groups()
        sample_stats = _empty_stats()
        sample_stats["files"] = cls.sample_files
        sample_stats["note"] = cls.sample_note
        for _, decs in cls.sample_groups:
            _merge_stats(sample_stats, _audit(decs, own_state.compute(decs), cls.spans))
        fixture_stats = _audit(cls.fixture_decisions, cls.fixture_payloads, cls.spans)
        fixture_stats["files"] = [FIXTURE_REL]
        populations = {"frozen_fixture": fixture_stats, "report_sub_sample": sample_stats}
        cls.artifact = build_artifact(populations, cls.prereg_sha, cls.spans)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(cls.artifact, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        assert_prereg_unchanged(PREREG_PATH, cls.prereg_sha)

    def test_fixture_payloads_match_obs_bit_exact(self):
        stats = self.artifact["populations"]["frozen_fixture"]
        self.assertGreater(stats["readable"], 0)
        self.assertEqual(stats["mismatches"], 0)
        self.assertEqual(stats["missing_payloads"], 0)

    def test_report_sample_payloads_match_obs_bit_exact(self):
        stats = self.artifact["populations"]["report_sub_sample"]
        if stats["decisions"] == 0:
            self.skipTest(self.sample_note or "report_sub 样本不可用")
        self.assertEqual(stats["mismatches"], 0)
        self.assertEqual(stats["missing_payloads"], 0)

    def test_unavailable_explicit_reasoned_never_zero_filled(self):
        for name, stats in self.artifact["populations"].items():
            self.assertEqual(stats["unavailable_without_reason"], 0, name)
            self.assertEqual(stats["readable_with_empty_ukeire"], 0, name)

    def test_fixture_exercises_both_layouts_and_unavailable(self):
        stats = self.artifact["populations"]["frozen_fixture"]
        self.assertGreater(stats["layouts"]["discard"], 0)
        self.assertGreater(stats["layouts"]["draw"], 0)
        self.assertGreater(stats["unavailable"], 0)

    def test_payload_contract_meta_and_validation(self):
        payload = next(p for p in self.fixture_payloads.values() if not p.get("unavailable"))
        self.assertEqual(payload["provider"], "own_state")
        self.assertIsNone(payload["checkpoint_sha256"])
        self.assertEqual(payload["unit"], "count")
        self.assertIs(payload["no_hidden_info"], True)
        self.assertEqual(payload["basis"], "self_hand")
        from jiangpu.contract import validate
        self.assertEqual(validate(_minimal_doc(payload)), [])

    def test_compute_uses_layout_span_not_hardcoded_offsets(self):
        for block in ("shanten", "sp_required_tiles"):
            tampered = dict(self.top)
            del tampered[block]
            with self.assertRaises(KeyError):
                self.own_state.compute(self.fixture_decisions[:1], top=tampered)

    def test_coverage_recorded_and_bars_met(self):
        cov = self.artifact["coverage"]
        self.assertEqual(cov["decisions"], cov["readable"] + cov["unavailable"])
        self.assertAlmostEqual(cov["ratio"], cov["readable"] / cov["decisions"], places=12)
        self.assertTrue(cov["triage_met"])
        self.assertTrue(cov["delivery_met"])

    def test_artifact_round_trips_with_prereg_hash(self):
        loaded = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(loaded["schema"], "ex1-own-state-validity/v1")
        self.assertEqual(loaded["prereg"]["sha256"], self.prereg_sha)
        self.assertEqual(loaded["total_mismatches"], 0)
        self.assertEqual(loaded["coverage"], self.artifact["coverage"])
        self.assertTrue(loaded["unavailable_all_have_reason"])

    def test_empty_input_returns_empty(self):
        self.assertEqual(self.own_state.compute([]), {})


if __name__ == "__main__":
    unittest.main()
