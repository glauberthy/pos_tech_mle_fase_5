"""
Tests for src/model_training.py.
Heavy CatBoost operations (train/load) are mocked to keep tests fast.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.model_training import _prepare_xy, DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ra": [f"RA-{i}" for i in range(n)],
            "fase": ["FASE1" if i % 2 == 0 else "FASE2" for i in range(n)],
            "turma": ["A"] * n,
            "matem": rng.uniform(4, 10, n),
            "ieg": rng.uniform(3, 10, n),
            "ips": rng.uniform(3, 10, n),
            "target": rng.integers(0, 2, n),
        }
    )


# ---------------------------------------------------------------------------
# _prepare_xy
# ---------------------------------------------------------------------------

class TestPrepareXy:
    NUM_FEATS = ["matem", "ieg", "ips"]
    CAT_FEATS = ["fase", "turma"]

    def test_returns_x_and_y(self):
        df = _make_df()
        x, y = _prepare_xy(df, self.NUM_FEATS + self.CAT_FEATS, self.CAT_FEATS)
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, np.ndarray)

    def test_x_has_correct_columns(self):
        df = _make_df()
        feat_cols = self.NUM_FEATS + self.CAT_FEATS
        x, _ = _prepare_xy(df, feat_cols, self.CAT_FEATS)
        assert list(x.columns) == feat_cols

    def test_y_is_integer_array(self):
        df = _make_df()
        _, y = _prepare_xy(df, self.NUM_FEATS + self.CAT_FEATS, self.CAT_FEATS)
        assert y.dtype in (np.int32, np.int64, int)

    def test_y_values_are_0_or_1(self):
        df = _make_df()
        _, y = _prepare_xy(df, self.NUM_FEATS + self.CAT_FEATS, self.CAT_FEATS)
        assert set(y).issubset({0, 1})

    def test_cat_features_become_category_dtype(self):
        df = _make_df()
        x, _ = _prepare_xy(df, self.NUM_FEATS + self.CAT_FEATS, self.CAT_FEATS)
        for col in self.CAT_FEATS:
            assert str(x[col].dtype) == "category", f"{col} should be category"

    def test_num_features_are_numeric(self):
        df = _make_df()
        x, _ = _prepare_xy(df, self.NUM_FEATS + self.CAT_FEATS, self.CAT_FEATS)
        for col in self.NUM_FEATS:
            assert pd.api.types.is_numeric_dtype(x[col]), f"{col} should be numeric"

    def test_missing_column_filled_with_nan(self):
        df = _make_df()
        # 'ingles' not in df — should be added as NaN
        feats = self.NUM_FEATS + ["ingles"] + self.CAT_FEATS
        x, _ = _prepare_xy(df, feats, self.CAT_FEATS)
        assert "ingles" in x.columns
        assert x["ingles"].isna().all()

    def test_nan_target_filled_with_zero(self):
        df = _make_df()
        df.loc[0, "target"] = float("nan")
        _, y = _prepare_xy(df, self.NUM_FEATS + self.CAT_FEATS, self.CAT_FEATS)
        assert y[0] == 0

    def test_cat_nan_filled_with_missing_category(self):
        df = _make_df()
        df.loc[0, "fase"] = None
        x, _ = _prepare_xy(df, self.NUM_FEATS + self.CAT_FEATS, self.CAT_FEATS)
        assert "MISSING_CATEGORY" in x["fase"].cat.categories

    def test_string_target_coerced(self):
        df = _make_df()
        df["target"] = df["target"].astype(str)
        _, y = _prepare_xy(df, self.NUM_FEATS + self.CAT_FEATS, self.CAT_FEATS)
        assert y.dtype in (np.int32, np.int64, int)

    def test_length_matches_input(self):
        df = _make_df(n=15)
        x, y = _prepare_xy(df, self.NUM_FEATS + self.CAT_FEATS, self.CAT_FEATS)
        assert len(x) == 15
        assert len(y) == 15


# ---------------------------------------------------------------------------
# DEFAULT_PARAMS
# ---------------------------------------------------------------------------

class TestDefaultParams:
    def test_has_required_keys(self):
        required = ["iterations", "learning_rate", "depth", "loss_function", "eval_metric", "random_seed"]
        for key in required:
            assert key in DEFAULT_PARAMS, f"Missing key: {key}"

    def test_loss_function_is_logloss(self):
        assert DEFAULT_PARAMS["loss_function"] == "Logloss"

    def test_eval_metric_is_auc(self):
        assert DEFAULT_PARAMS["eval_metric"] == "AUC"

    def test_no_file_writing(self):
        assert DEFAULT_PARAMS["allow_writing_files"] is False


# ---------------------------------------------------------------------------
# load_model (mocked filesystem)
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_loads_model_and_meta(self, tmp_path):
        meta = {
            "num_features": ["matem"],
            "cat_features": ["fase"],
            "feature_cols": ["matem", "fase"],
        }
        meta_path = tmp_path / "model_meta.json"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        (tmp_path / "catboost_model.cbm").write_bytes(b"fake")

        mock_model = MagicMock()
        mock_cls = MagicMock(return_value=mock_model)

        with patch("src.model_training.CatBoostClassifier", mock_cls):
            from src.model_training import load_model
            model, loaded_meta = load_model(tmp_path)

        mock_model.load_model.assert_called_once()
        assert loaded_meta["num_features"] == ["matem"]
        assert loaded_meta["cat_features"] == ["fase"]

    def test_returns_tuple(self, tmp_path):
        meta = {"num_features": [], "cat_features": [], "feature_cols": []}
        (tmp_path / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (tmp_path / "catboost_model.cbm").write_bytes(b"fake")

        with patch("src.model_training.CatBoostClassifier", MagicMock(return_value=MagicMock())):
            from src.model_training import load_model
            result = load_model(tmp_path)

        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# predict_proba (mocked model)
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_returns_array_of_correct_shape(self):
        df = _make_df(n=8)
        num_feats = ["matem", "ieg", "ips"]
        cat_feats = ["fase", "turma"]

        fake_proba = np.array([[0.6, 0.4]] * 8)
        mock_pool_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = fake_proba

        with patch("src.model_training.Pool", mock_pool_cls):
            from src.model_training import predict_proba
            result = predict_proba(mock_model, df, num_feats, cat_feats)

        assert result.shape == (8,)

    def test_returns_second_column_proba(self):
        df = _make_df(n=4)
        num_feats = ["matem"]
        cat_feats = ["fase"]

        expected = np.array([0.1, 0.9, 0.3, 0.7])
        fake_proba = np.column_stack([1 - expected, expected])
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = fake_proba

        with patch("src.model_training.Pool", MagicMock()):
            from src.model_training import predict_proba
            result = predict_proba(mock_model, df, num_feats, cat_feats)

        np.testing.assert_allclose(result, expected)
