"""
Tests for src/inference.py.
Uses mocks to avoid requiring a trained CatBoost model on disk.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.inference import alert_list, explain_student


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scored_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    fases = ["FASE1" if i % 2 == 0 else "FASE2" for i in range(n)]
    return pd.DataFrame(
        {
            "ra": [f"RA-{i:03d}" for i in range(n)],
            "fase": fases,
            "turma": ["A"] * n,
            "score": rng.uniform(10, 90, n),
            "alerta": [False] * n,
            "top3_factors": [["ieg", "matem", "ips"]] * n,
            "top3_values": [[0.3, -0.2, 0.1]] * n,
            "matem": rng.uniform(4, 10, n),
            "ieg": rng.uniform(4, 10, n),
            "ips": rng.uniform(4, 10, n),
        }
    )


# ---------------------------------------------------------------------------
# alert_list
# ---------------------------------------------------------------------------

class TestAlertList:
    def test_returns_dataframe(self):
        df = _scored_df()
        result = alert_list(df, k_pct=15.0)
        assert isinstance(result, pd.DataFrame)

    def test_alerta_column_exists(self):
        df = _scored_df()
        result = alert_list(df, k_pct=15.0)
        assert "alerta" in result.columns

    def test_top_k_pct_per_fase(self):
        df = _scored_df(n=20)
        # 15% do top de cada fase de 10 alunos = ceil(10*0.15) = 2 alertas por fase
        result = alert_list(df, k_pct=15.0)
        n_alerta = result["alerta"].sum()
        assert 2 <= n_alerta <= 6  # margem razoável

    def test_sorted_by_fase_and_score_desc(self):
        df = _scored_df(n=20)
        result = alert_list(df, k_pct=15.0)
        for fase, grp in result.groupby("fase"):
            scores = grp["score"].tolist()
            assert scores == sorted(scores, reverse=True), f"Fase {fase} não está ordenada"

    def test_k_pct_100_marks_all(self):
        df = _scored_df(n=10)
        result = alert_list(df, k_pct=100.0)
        assert result["alerta"].all()

    def test_fase_col_param(self):
        df = _scored_df(n=10)
        df = df.rename(columns={"fase": "phase"})
        result = alert_list(df, k_pct=50.0, fase_col="phase")
        assert "alerta" in result.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["ra", "fase", "score", "turma"])
        result = alert_list(df, k_pct=15.0)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# explain_student
# ---------------------------------------------------------------------------

class TestExplainStudent:
    def test_found_student(self):
        df = _scored_df(n=10)
        ra = df["ra"].iloc[3]
        result = explain_student(ra, df)
        assert result["ra"] == ra
        assert "score" in result
        assert "alerta" in result
        assert "fase" in result
        assert "top_factors" in result

    def test_not_found_returns_error(self):
        df = _scored_df(n=5)
        result = explain_student("RA-NONEXISTENT", df)
        assert "error" in result
        assert "not found" in result["error"]

    def test_top_factors_list(self):
        df = _scored_df(n=5)
        ra = df["ra"].iloc[0]
        result = explain_student(ra, df)
        assert isinstance(result["top_factors"], list)

    def test_top_factors_contain_shap_value(self):
        df = _scored_df(n=5)
        ra = df["ra"].iloc[0]
        result = explain_student(ra, df)
        for item in result["top_factors"]:
            assert "feature" in item
            assert "shap_value" in item

    def test_score_is_float(self):
        df = _scored_df(n=5)
        ra = df["ra"].iloc[0]
        result = explain_student(ra, df)
        assert isinstance(result["score"], float)

    def test_alerta_is_bool(self):
        df = _scored_df(n=5)
        ra = df["ra"].iloc[0]
        result = explain_student(ra, df)
        assert isinstance(result["alerta"], bool)

    def test_uses_latest_record_when_multiple(self):
        df = pd.DataFrame(
            {
                "ra": ["RA-001", "RA-001"],
                "fase": ["FASE1", "FASE2"],
                "turma": ["A", "B"],
                "score": [30.0, 80.0],
                "alerta": [False, True],
                "top3_factors": [["ieg"], ["matem"]],
                "top3_values": [[0.1], [0.5]],
                "matem": [7.0, 8.0],
                "ieg": [6.0, 9.0],
                "ips": [5.0, 7.0],
            }
        )
        result = explain_student("RA-001", df)
        # retorna o primeiro match (iloc[0]) — score 30.0
        assert result["score"] == 30.0

    def test_student_value_is_none_when_feature_not_in_row(self):
        df = pd.DataFrame(
            {
                "ra": ["RA-001"],
                "fase": ["FASE1"],
                "turma": ["A"],
                "score": [50.0],
                "alerta": [False],
                "top3_factors": [["feature_nao_existe"]],
                "top3_values": [[0.2]],
            }
        )
        result = explain_student("RA-001", df)
        assert result["top_factors"][0]["student_value"] is None


# ---------------------------------------------------------------------------
# score_students (mocked)
# ---------------------------------------------------------------------------

class TestScoreStudents:
    def _make_fe_df(self, n=5):
        return pd.DataFrame(
            {
                "ra": [f"RA-{i:03d}" for i in range(n)],
                "fase": ["FASE1"] * n,
                "turma": ["A"] * n,
                "matem": [7.0] * n,
                "ieg": [6.0] * n,
                "ips": [5.0] * n,
                "target": [0] * n,
            }
        )

    def test_score_column_0_to_100(self):
        from src.inference import score_students
        import src.inference as inf_module

        fake_proba = np.array([0.1, 0.5, 0.9, 0.2, 0.7])
        mock_model = MagicMock()
        mock_meta = {
            "num_features": ["matem", "ieg", "ips"],
            "cat_features": ["fase", "turma"],
        }

        original = inf_module._SHAP_AVAILABLE
        try:
            inf_module._SHAP_AVAILABLE = False
            with patch("src.inference.load_model", return_value=(mock_model, mock_meta)):
                with patch("src.inference.predict_proba", return_value=fake_proba):
                    df = self._make_fe_df(n=5)
                    result = score_students(df, model_dir="models")
        finally:
            inf_module._SHAP_AVAILABLE = original

        assert "score" in result.columns
        assert (result["score"] >= 0).all()
        assert (result["score"] <= 100).all()

    def test_prob_column_added(self):
        from src.inference import score_students
        import src.inference as inf_module

        fake_proba = np.array([0.3, 0.6])
        mock_meta = {"num_features": ["matem"], "cat_features": ["fase"]}

        original = inf_module._SHAP_AVAILABLE
        try:
            inf_module._SHAP_AVAILABLE = False
            with patch("src.inference.load_model", return_value=(MagicMock(), mock_meta)):
                with patch("src.inference.predict_proba", return_value=fake_proba):
                    df = self._make_fe_df(n=2)
                    result = score_students(df, model_dir="models")
        finally:
            inf_module._SHAP_AVAILABLE = original

        assert "prob" in result.columns
        np.testing.assert_allclose(result["prob"].values, fake_proba)

    def test_shap_fallback_when_unavailable(self):
        """When SHAP is unavailable, top3_factors and top3_values are empty lists."""
        from src.inference import score_students
        import src.inference as inf_module

        fake_proba = np.array([0.4, 0.8])
        mock_meta = {"num_features": ["matem"], "cat_features": ["fase"]}

        original = inf_module._SHAP_AVAILABLE
        try:
            inf_module._SHAP_AVAILABLE = False
            with patch("src.inference.load_model", return_value=(MagicMock(), mock_meta)):
                with patch("src.inference.predict_proba", return_value=fake_proba):
                    df = self._make_fe_df(n=2)
                    result = score_students(df, model_dir="models")
            assert result["top3_factors"].iloc[0] == []
            assert result["top3_values"].iloc[0] == []
        finally:
            inf_module._SHAP_AVAILABLE = original

    def test_shap_available_populates_fields(self):
        """If shap present, _add_shap_explanations should fill top3 lists."""
        from src.inference import score_students
        import src.inference as inf_module

        fake_proba = np.array([0.2, 0.8])
        mock_meta = {"num_features": ["x", "y"], "cat_features": []}
        # create fake shap values: 2 examples, 2 features
        fake_shap = np.array([[0.1, -0.3], [0.4, 0.2]])

        class DummyExplainer:
            def __init__(self, model):
                pass

            def shap_values(self, x):
                return fake_shap

        original = inf_module._SHAP_AVAILABLE
        try:
            inf_module._SHAP_AVAILABLE = True
            # patch shap.TreeExplainer
            monkey = patch("shap.TreeExplainer", DummyExplainer)
            with monkey:
                with patch("src.inference.load_model", return_value=(MagicMock(), mock_meta)):
                    with patch("src.inference.predict_proba", return_value=fake_proba):
                        df = self._make_fe_df(n=2)
                        result = score_students(df, model_dir="models")
            # check that lists are populated
            assert result["top3_factors"].iloc[0] != []
            assert result["top3_values"].iloc[0] != []
        finally:
            inf_module._SHAP_AVAILABLE = original

    def test_add_shap_explanations_handles_exception(self):
        """_add_shap_explanations should catch exceptions from explainer."""
        from src.inference import score_students
        import src.inference as inf_module

        fake_proba = np.array([0.5])
        mock_meta = {"num_features": ["a"], "cat_features": []}

        class ExplainerExcept:
            def __init__(self, model):
                pass
            def shap_values(self, x):
                raise RuntimeError("boom")

        original = inf_module._SHAP_AVAILABLE
        try:
            inf_module._SHAP_AVAILABLE = True
            with patch("shap.TreeExplainer", ExplainerExcept):
                with patch("src.inference.load_model", return_value=(MagicMock(), mock_meta)):
                    with patch("src.inference.predict_proba", return_value=fake_proba):
                        df = self._make_fe_df(n=1)
                        result = score_students(df, model_dir="models")
            # should not crash and top lists empty
            assert result["top3_factors"].iloc[0] == []
            assert result["top3_values"].iloc[0] == []
        finally:
            inf_module._SHAP_AVAILABLE = original

    def test_shap_import_failure_sets_flag(self, monkeypatch):
        # reload module without shap installed to hit ImportError branch
        import sys, importlib
        if "src.inference" in sys.modules:
            del sys.modules["src.inference"]
        # temporarily remove shap
        orig_shap = sys.modules.pop("shap", None)
        try:
            # force import to raise ImportError by monkeypatching __import__
            import builtins
            real_import = builtins.__import__
            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "shap":
                    raise ImportError("no shap")
                return real_import(name, globals, locals, fromlist, level)
            builtins.__import__ = fake_import

            import src.inference as inf_mod
            importlib.reload(inf_mod)
            assert inf_mod._SHAP_AVAILABLE is False
        finally:
            builtins.__import__ = real_import
            if orig_shap is not None:
                sys.modules["shap"] = orig_shap
