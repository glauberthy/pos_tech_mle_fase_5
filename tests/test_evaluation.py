"""
Tests for src/evaluation.py (métricas independentes da definição do target).
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    recall_at_topk,
    precision_at_topk,
    lift_at_topk,
    roc_auc,
    stratified_topk_alert,
    evaluate,
)


class TestRecallAtTopK:
    def test_perfect_model(self):
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        s = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # Top 20% = 2 students, both are positives
        r = recall_at_topk(y, s, k_pct=20)
        assert r == pytest.approx(1.0)

    def test_random_model(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=1000)
        s = rng.random(1000)
        r = recall_at_topk(y, s, k_pct=20)
        # Random model should capture ~20% of positives
        assert 0.1 < r < 0.35

    def test_no_positives(self):
        y = np.zeros(10)
        s = np.ones(10)
        assert recall_at_topk(y, s, k_pct=20) == 0.0

    def test_topk_100_captures_all(self):
        y = np.array([1, 0, 1, 0, 1])
        s = np.random.random(5)
        assert recall_at_topk(y, s, k_pct=100) == pytest.approx(1.0)


class TestPrecisionAtTopK:
    def test_perfect_model(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.3, 0.1])
        p = precision_at_topk(y, s, k_pct=50)
        assert p == pytest.approx(1.0)

    def test_worst_model(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        p = precision_at_topk(y, s, k_pct=50)
        assert p == pytest.approx(0.0)


class TestLiftAtTopK:
    def test_lift_greater_than_one_for_good_model(self):
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        s = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        lift = lift_at_topk(y, s, k_pct=20)
        assert lift > 1.0

    def test_lift_one_for_random_model(self):
        # Uniform scores = random model → lift ≈ 1 (approximately)
        rng = np.random.default_rng(123)
        y = rng.integers(0, 2, 10000)
        s = np.full(10000, 0.5)
        lift = lift_at_topk(y, s, k_pct=20)
        assert 0.8 < lift < 1.2


class TestRocAuc:
    def test_perfect_classifier(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        assert roc_auc(y, s) == pytest.approx(1.0)

    def test_random_classifier(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 10000)
        s = rng.random(10000)
        auc = roc_auc(y, s)
        assert 0.45 < auc < 0.55

    def test_auc_at_least_half(self):
        y = np.array([1, 0, 1, 0])
        s = np.array([0.7, 0.6, 0.4, 0.3])
        auc = roc_auc(y, s)
        assert auc >= 0.5


class TestStratifiedTopkAlert:
    def test_alerts_per_phase(self):
        df = pd.DataFrame({
            "ra": list(range(20)),
            "fase": ["FASE1"] * 10 + ["FASE2"] * 10,
            "score": list(range(20)),
        })
        result = stratified_topk_alert(df, score_col="score", fase_col="fase", k_pct=20)
        # Top 20% of each phase (2 each)
        assert result[result["fase"] == "FASE1"]["alerta"].sum() == 2
        assert result[result["fase"] == "FASE2"]["alerta"].sum() == 2

    def test_total_alerts_approximately_k_pct(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "ra": range(100),
            "fase": ["FASE1"] * 50 + ["FASE2"] * 50,
            "score": rng.random(100) * 100,
        })
        result = stratified_topk_alert(df, k_pct=20)
        # Each phase contributes ~10 alerts → ~20 total
        assert 18 <= result["alerta"].sum() <= 22


class TestEvaluate:
    def _make_scored_df(self, n=100, seed=42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        # Good model: positives get higher scores
        s = np.where(y == 1, rng.uniform(60, 100, n), rng.uniform(0, 60, n))
        return pd.DataFrame({
            "target": y,
            "score": s,
            "fase": (["FASE1", "FASE2"] * (n // 2 + 1))[:n],
        })

    def test_evaluate_returns_auc(self):
        df = self._make_scored_df()
        results = evaluate(df, k_pcts=[15])
        assert "auc" in results
        assert 0.5 < results["auc"] <= 1.0

    def test_evaluate_topk_metrics_present(self):
        df = self._make_scored_df()
        results = evaluate(df, k_pcts=[10, 15])
        assert "recall_top10" in results
        assert "precision_top15" in results
        assert "lift_top15" in results

    def test_per_fase_breakdown(self):
        df = self._make_scored_df()
        results = evaluate(df, k_pcts=[15])
        assert "per_fase" in results
        assert len(results["per_fase"]) >= 1


# ---------------------------------------------------------------------------
# Notebook-aligned helpers
# ---------------------------------------------------------------------------

from src.evaluation import (
    print_report,
    operational_metrics_topk,
    operational_metrics_topk_by_fase,
)


class TestPrintReport:
    def _results(self):
        return {
            "n_total": 100,
            "n_positive": 20,
            "base_rate": 0.2,
            "auc": 0.75,
            "recall_top10": 0.4,
            "precision_top10": 0.8,
            "lift_top10": 4.0,
            "recall_top15": 0.5,
            "precision_top15": 0.67,
            "lift_top15": 3.35,
        }

    def test_returns_string(self):
        report = print_report(self._results())
        assert isinstance(report, str)

    def test_contains_auc(self):
        report = print_report(self._results())
        assert "0.75" in report

    def test_contains_k_percentages(self):
        report = print_report(self._results())
        assert "10" in report
        assert "15" in report

    def test_empty_results_no_error(self):
        report = print_report({})
        assert isinstance(report, str)


class TestOperationalMetricsTopk:
    def _make_df(self, n=100, seed=42):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        s = np.where(y == 1, rng.uniform(60, 100, n), rng.uniform(0, 60, n))
        fases = ([1, 2, 3, 4] * (n // 4 + 1))[:n]
        return pd.DataFrame({"target": y, "score": s, "fase": fases})

    def test_returns_dict(self):
        df = self._make_df()
        res = operational_metrics_topk(df, k_pct=15.0)
        assert isinstance(res, dict)

    def test_contains_expected_keys(self):
        df = self._make_df()
        res = operational_metrics_topk(df, k_pct=15.0)
        for key in ["recall@k", "precision@k", "lift@k", "tp", "fp", "fn", "df_with_alerts"]:
            assert key in res, f"missing key: {key}"

    def test_recall_between_0_and_1(self):
        df = self._make_df()
        res = operational_metrics_topk(df, k_pct=15.0)
        assert 0.0 <= res["recall@k"] <= 1.0

    def test_n_alert_matches_expected(self):
        df = self._make_df(n=100)
        res = operational_metrics_topk(df, k_pct=15.0)
        # Estratificado por 4 fases: cada fase 25 alunos → ceil(25*0.15)=4 alertas
        assert res["n_alert"] == 16

    def test_df_with_alerts_has_alert_column(self):
        df = self._make_df()
        res = operational_metrics_topk(df)
        assert "alerta" in res["df_with_alerts"].columns

    def test_custom_alert_col(self):
        df = self._make_df()
        res = operational_metrics_topk(df, alert_col="my_alert")
        assert "my_alert" in res["df_with_alerts"].columns

    def test_lift_is_none_when_no_positives(self):
        df = pd.DataFrame({"target": [0] * 20, "score": list(range(20)), "fase": [1] * 20})
        res = operational_metrics_topk(df, k_pct=10.0)
        assert res["lift@k"] is None


class TestOperationalMetricsTopkByFase:
    def _make_df(self, n=80, seed=7):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        s = np.where(y == 1, rng.uniform(60, 100, n), rng.uniform(0, 60, n))
        fases = ([1, 2, 3, 4] * (n // 4 + 1))[:n]
        return pd.DataFrame({"target": y, "score": s, "fase": fases})

    def test_returns_dataframe(self):
        df = self._make_df()
        out = operational_metrics_topk_by_fase(df, k_pct=15.0)
        assert isinstance(out, pd.DataFrame)

    def test_one_row_per_fase(self):
        df = self._make_df()
        out = operational_metrics_topk_by_fase(df, k_pct=15.0)
        assert len(out) == 4

    def test_contains_expected_columns(self):
        df = self._make_df()
        out = operational_metrics_topk_by_fase(df)
        for col in ["fase", "n", "n_pos", "recall@k", "precision@k", "lift@k"]:
            assert col in out.columns, f"missing column: {col}"

    def test_sorted_by_fase(self):
        df = self._make_df()
        out = operational_metrics_topk_by_fase(df)
        assert list(out["fase"]) == sorted(out["fase"].tolist())

    def test_recall_between_0_and_1(self):
        df = self._make_df()
        out = operational_metrics_topk_by_fase(df)
        assert (out["recall@k"].dropna().between(0, 1)).all()
