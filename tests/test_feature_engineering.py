"""
Tests for src/feature_engineering.py (estratégia de piora de defasagem).
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    _compute_base_scores,
    _compute_context,
    build_features,
    get_feature_columns,
)


def _make_minimal_df(n=10, year=2022) -> pd.DataFrame:
    """Minimal DataFrame aligned with the new preprocessing output schema."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "ra": [f"RA-{i}" for i in range(n)],
        "ano_base": year,
        "fase": ([1, 2] * (n // 2 + 1))[:n],
        "turma": (["A", "B"] * (n // 2 + 1))[:n],
        "genero": (["F", "M"] * (n // 2 + 1))[:n],
        "instituicao": ["PUBLICA"] * n,
        "ano_ingresso": [2019] * n,
        "ieg": rng.uniform(4, 10, n),
        "iaa": rng.uniform(5, 10, n),
        "ips": rng.uniform(4, 9, n),
        "ipp": rng.uniform(3, 8, n),
        "matem": rng.uniform(4, 10, n),
        "portug": rng.uniform(4, 10, n),
        "ingles": np.where(rng.random(n) > 0.3, rng.uniform(4, 10, n), np.nan),
        "defasagem_t": ([0, 1] * (n // 2 + 1))[:n],
        "defasagem_t1": ([0, -1] * (n // 2 + 1))[:n],
        "target": ([0, 1] * (n // 2 + 1))[:n],
    })


class TestComputeBaseScores:
    def test_fez_ingles_flag(self):
        df = _make_minimal_df()
        df_out = _compute_base_scores(df)
        assert "fez_ingles" in df_out.columns
        assert df_out["fez_ingles"].isin([0, 1]).all()
        assert df_out.loc[df_out["ingles"].isna(), "fez_ingles"].eq(0).all()

    def test_media_provas_computed(self):
        df = _make_minimal_df()
        df["ingles"] = np.nan
        df_out = _compute_base_scores(df)
        # media_provas should equal mean of matem and portug when ingles is NaN
        expected = df[["matem", "portug"]].mean(axis=1)
        pd.testing.assert_series_equal(
            df_out["media_provas"].round(6),
            expected.round(6),
            check_names=False,
        )

    def test_disp_provas_zero_when_one_exam(self):
        df = _make_minimal_df()
        df["ingles"] = np.nan
        df["portug"] = np.nan
        df_out = _compute_base_scores(df)
        # std of a single value is 0
        assert (df_out["disp_provas"] == 0).all()


class TestComputeContext:
    def test_tempo_casa(self):
        df = _make_minimal_df(year=2023)
        df["ano_ingresso"] = 2019
        df_out = _compute_context(df)
        assert (df_out["tempo_casa"] == 4).all()

    def test_iaa_participou_flag(self):
        df = _make_minimal_df()
        df.loc[:2, "iaa"] = np.nan
        df_out = _compute_context(df)
        assert "iaa_participou" in df_out.columns
        assert df_out.loc[:2, "iaa_participou"].eq(0).all()
        assert df_out.loc[3:, "iaa_participou"].eq(1).all()

    def test_iaa_filled_zero_when_nan(self):
        df = _make_minimal_df()
        df["iaa"] = np.nan
        df_out = _compute_context(df)
        assert (df_out["iaa"] == 0).all()


class TestBuildFeatures:
    def test_returns_lookup_tables_in_train(self):
        df = _make_minimal_df(n=20)
        fe_df, tables = build_features(df, is_train=True)
        assert isinstance(tables, dict)
        assert len(tables) > 0

    def test_group_stats_columns_created(self):
        df = _make_minimal_df(n=20)
        fe_df, _ = build_features(df, is_train=True)
        assert "turma_mean_ieg" in fe_df.columns
        assert "z_turma_ieg" in fe_df.columns
        assert "delta_turma_ieg" in fe_df.columns

    def test_flags_created(self):
        df = _make_minimal_df(n=20)
        fe_df, _ = build_features(df, is_train=True)
        assert "abaixo_p25_turma_media_provas" in fe_df.columns
        assert "abaixo_p25_turma_matem" in fe_df.columns
        assert "abaixo_p25_turma_ieg" in fe_df.columns

    def test_no_nan_in_numeric_features_after_train(self):
        df = _make_minimal_df(n=20)
        fe_df, _ = build_features(df, is_train=True)
        num_cols = fe_df.select_dtypes(include=[np.number]).columns
        # Exclude defasagem columns from check
        check_cols = [c for c in num_cols if "defasagem" not in c and c != "target"]
        assert fe_df[check_cols].isna().sum().sum() == 0

    def test_valid_uses_train_lookup(self):
        train_df = _make_minimal_df(n=20, year=2022)
        valid_df = _make_minimal_df(n=10, year=2023)
        _, tables = build_features(train_df, is_train=True)
        valid_fe, _ = build_features(valid_df, lookup_tables=tables, is_train=False)
        assert "turma_mean_ieg" in valid_fe.columns


class TestGetFeatureColumns:
    def test_excludes_identifiers(self):
        df = _make_minimal_df(n=10)
        fe_df, _ = build_features(df, is_train=True)
        num_feats, cat_feats = get_feature_columns(fe_df)
        assert "ra" not in num_feats
        assert "ra" not in cat_feats
        assert "target" not in num_feats

    def test_cat_features_present(self):
        df = _make_minimal_df(n=10)
        fe_df, _ = build_features(df, is_train=True)
        _, cat_feats = get_feature_columns(fe_df)
        assert "fase" in cat_feats
        assert "turma" in cat_feats
