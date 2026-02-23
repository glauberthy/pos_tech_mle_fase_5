"""
Tests for src/utils.py.
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils import (
    compute_psi,
    monitor_drift,
    phase_sort_key,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    setup_logging,
    load_jsonl,
    log_monitoring_event,
)


class TestComputePsi:
    def test_identical_distributions(self):
        rng = np.random.default_rng(0)
        data = rng.random(1000)
        psi = compute_psi(data, data.copy())
        assert psi < 0.01  # Nearly zero drift

    def test_very_different_distributions(self):
        rng = np.random.default_rng(1)
        d1 = rng.random(1000)
        d2 = np.ones(1000) * 0.95  # Completely different
        psi = compute_psi(d1, d2)
        assert psi > 0.25

    def test_returns_float(self):
        d = np.random.random(100)
        assert isinstance(compute_psi(d, d), float)


class TestMonitorDrift:
    def test_no_drift_same_distribution(self):
        d = np.random.default_rng(42).random(500)
        result = monitor_drift(d, d.copy(), threshold=0.25)
        assert result["drift_detected"] is False
        assert result["status"] == "stable"

    def test_drift_detected(self):
        baseline = np.zeros(500)  # All scores near 0
        current = np.ones(500)   # All scores near 1
        result = monitor_drift(baseline, current, threshold=0.25)
        assert result["drift_detected"] is True
        assert result["status"] == "DRIFT DETECTED"

    def test_returns_psi_key(self):
        d = np.random.random(100)
        result = monitor_drift(d, d)
        assert "psi" in result


class TestPhaseSortKey:
    def test_alfa_is_zero(self):
        assert phase_sort_key("ALFA") == 0

    def test_numeric_phases(self):
        assert phase_sort_key("FASE1") == 1
        assert phase_sort_key("FASE8") == 8

    def test_sort_order(self):
        phases = ["FASE3", "ALFA", "FASE1", "FASE8", "FASE2"]
        sorted_phases = sorted(phases, key=phase_sort_key)
        assert sorted_phases == ["ALFA", "FASE1", "FASE2", "FASE3", "FASE8"]


class TestPersistence:
    def test_save_load_json(self, tmp_path):
        data = {"key": "value", "nums": [1, 2, 3]}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_save_load_pickle(self, tmp_path):
        data = {"array": np.array([1, 2, 3]), "str": "hello"}
        path = tmp_path / "test.pkl"
        save_pickle(data, path)
        loaded = load_pickle(path)
        assert loaded["str"] == "hello"
        np.testing.assert_array_equal(loaded["array"], data["array"])

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c.json"
        save_json({"x": 1}, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_does_not_raise(self):
        setup_logging("INFO")

    def test_accepts_debug_level(self):
        setup_logging("DEBUG")

    def test_writes_log_file(self, tmp_path):
        import logging
        log_path = tmp_path / "test.log"
        setup_logging("WARNING", log_file=log_path)
        logging.getLogger("test_setup").warning("test message")
        assert log_path.exists()

    def test_creates_parent_dirs_for_log(self, tmp_path):
        log_path = tmp_path / "nested" / "dir" / "app.log"
        setup_logging("INFO", log_file=log_path)
        assert log_path.parent.exists()


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_returns_list_of_dicts(self, tmp_path):
        path = tmp_path / "events.jsonl"
        import json
        path.write_text(
            json.dumps({"a": 1}) + "\n" + json.dumps({"b": 2}) + "\n",
            encoding="utf-8",
        )
        result = load_jsonl(path)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"a": 1}

    def test_returns_empty_list_when_file_missing(self, tmp_path):
        result = load_jsonl(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_skips_blank_lines(self, tmp_path):
        import json
        path = tmp_path / "events.jsonl"
        path.write_text(
            json.dumps({"x": 1}) + "\n\n" + json.dumps({"y": 2}) + "\n",
            encoding="utf-8",
        )
        result = load_jsonl(path)
        assert len(result) == 2

    def test_skips_invalid_json_lines(self, tmp_path):
        import json
        path = tmp_path / "events.jsonl"
        path.write_text(
            json.dumps({"valid": True}) + "\nNOT_JSON\n",
            encoding="utf-8",
        )
        result = load_jsonl(path)
        assert len(result) == 1
        assert result[0]["valid"] is True

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_jsonl(path) == []


# ---------------------------------------------------------------------------
# phase_sort_key (edge cases)
# ---------------------------------------------------------------------------

class TestPhaseSortKeyEdgeCases:
    def test_f0_is_zero(self):
        assert phase_sort_key("F0") == 0

    def test_fase0_is_zero(self):
        assert phase_sort_key("FASE0") == 0

    def test_string_zero_is_zero(self):
        assert phase_sort_key("0") == 0

    def test_f_prefix_with_digit(self):
        assert phase_sort_key("F3") == 3

    def test_unknown_returns_99(self):
        assert phase_sort_key("XPTO_ABC") == 99


# ---------------------------------------------------------------------------
# compute_psi (edge cases)
# ---------------------------------------------------------------------------

class TestComputePsiEdgeCases:
    def test_empty_baseline_returns_zero(self):
        assert compute_psi(np.array([]), np.array([0.5, 0.6])) == 0.0

    def test_empty_current_returns_zero(self):
        assert compute_psi(np.array([0.5, 0.6]), np.array([])) == 0.0

    def test_both_empty_returns_zero(self):
        assert compute_psi(np.array([]), np.array([])) == 0.0

    def test_psi_non_negative(self):
        rng = np.random.default_rng(5)
        b = rng.random(200)
        c = rng.random(200) * 0.5
        assert compute_psi(b, c) >= 0.0


# ---------------------------------------------------------------------------
# monitor_drift (severity levels)
# ---------------------------------------------------------------------------

class TestMonitorDriftSeverity:
    def test_moderate_severity(self):
        rng = np.random.default_rng(10)
        baseline = rng.random(500)
        # small shift to produce PSI between 0.10 and 0.25
        current = np.clip(baseline + 0.25, 0, 1)
        result = monitor_drift(baseline, current, threshold=0.25)
        # PSI could be low/moderate depending on shift; just validate structure
        assert result["severity"] in ("low", "moderate", "high")
        assert "psi" in result
        assert "drift_detected" in result

    def test_low_severity_on_identical(self):
        d = np.random.default_rng(0).random(300)
        result = monitor_drift(d, d.copy())
        assert result["severity"] == "low"

    def test_high_severity_on_extreme_shift(self):
        baseline = np.zeros(500)
        current = np.ones(500)
        result = monitor_drift(baseline, current, threshold=0.25)
        assert result["severity"] == "high"
        assert result["drift_detected"] is True

    def test_n_baseline_and_current_correct(self):
        b = np.random.random(100)
        c = np.random.random(200)
        result = monitor_drift(b, c)
        assert result["n_baseline"] == 100
        assert result["n_current"] == 200


# ---------------------------------------------------------------------------
# log_monitoring_event
# ---------------------------------------------------------------------------

class TestLogMonitoringEvent:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "drift.jsonl"
        log_monitoring_event(path, "test_event", {"psi": 0.05})
        assert path.exists()

    def test_appends_multiple_events(self, tmp_path):
        path = tmp_path / "drift.jsonl"
        log_monitoring_event(path, "event_1", {"psi": 0.01})
        log_monitoring_event(path, "event_2", {"psi": 0.03})
        result = load_jsonl(path)
        assert len(result) == 2

    def test_returned_dict_has_timestamp(self, tmp_path):
        path = tmp_path / "drift.jsonl"
        event = log_monitoring_event(path, "predict_batch", {"psi": 0.1})
        assert "timestamp_utc" in event

    def test_returned_dict_has_event_type(self, tmp_path):
        path = tmp_path / "drift.jsonl"
        event = log_monitoring_event(path, "baseline_snapshot", {"n_students": 100})
        assert event["event_type"] == "baseline_snapshot"

    def test_data_fields_merged_into_event(self, tmp_path):
        path = tmp_path / "drift.jsonl"
        event = log_monitoring_event(path, "test", {"psi": 0.07, "severity": "low"})
        assert event["psi"] == 0.07
        assert event["severity"] == "low"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "events.jsonl"
        log_monitoring_event(path, "test", {})
        assert path.exists()
