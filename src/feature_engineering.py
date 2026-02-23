"""
Feature engineering aligned with the notebook strategy.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-6
PHASES_ENGLISH_NOT_REQUIRED_INT = {0, 1, 2, 8}
CAT_FEATURES = ["fase", "turma", "genero", "instituicao"]
GROUP_VARS = ["media_provas", "matem", "portug", "ieg", "ips", "iaa"]


def _phase_as_int_series(phase: pd.Series) -> pd.Series:
    s = phase.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    mapped = (
        s.astype(str)
        .str.upper()
        .str.extract(r"(\d+)", expand=False)
    )
    alfa_mask = s.astype(str).str.upper().str.strip().eq("ALFA")
    result = pd.to_numeric(mapped, errors="coerce").astype("Int64")
    result = result.where(~alfa_mask, 0)
    return result


def _p25(x: pd.Series):
    return np.nanquantile(x, 0.25)


def _p75(x: pd.Series):
    return np.nanquantile(x, 0.75)


def _compute_base_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fez_ingles"] = df["ingles"].notna().astype(int)
    provas_cols = [c for c in ["matem", "portug", "ingles"] if c in df.columns]
    df["media_provas"] = df[provas_cols].mean(axis=1, skipna=True)
    df["disp_provas"] = df[provas_cols].std(axis=1, ddof=0, skipna=True).fillna(0)
    return df


def _compute_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ano_ingresso"] = pd.to_numeric(df.get("ano_ingresso"), errors="coerce")
    df["tempo_casa"] = pd.to_numeric(df.get("ano_base"), errors="coerce") - df["ano_ingresso"]
    df["iaa_participou"] = df["iaa"].notna().astype(int) if "iaa" in df.columns else 0
    if "iaa" in df.columns:
        df["iaa"] = pd.to_numeric(df["iaa"], errors="coerce").fillna(0)
    if "ipp" not in df.columns:
        df["ipp"] = np.nan
    return df


def make_features(df: pd.DataFrame, lookup: Optional[dict] = None, is_train: bool = True) -> tuple[pd.DataFrame, dict]:
    """Notebook-aligned main feature builder."""
    df = df.copy()
    df = _compute_base_scores(df)
    df = _compute_context(df)

    if "fase" in df.columns:
        phase_int = _phase_as_int_series(df["fase"])
        df["fase"] = phase_int
        mask_no_eng = phase_int.isin(PHASES_ENGLISH_NOT_REQUIRED_INT)
        df.loc[mask_no_eng, "ingles"] = np.nan
        df.loc[mask_no_eng, "fez_ingles"] = 0

    if "defasagem_t" in df.columns:
        d = pd.to_numeric(df["defasagem_t"], errors="coerce").fillna(0)
    else:
        d = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    df["defasagem_t"] = d
    df["defasagem_t_is0"] = (d == 0).astype(int)
    df["defasagem_t_bin_1"] = (d == 1).astype(int)
    df["defasagem_t_bin_2plus"] = (d >= 2).astype(int)

    if lookup is None:
        lookup = {}

    if is_train:
        turma_stats = (
            df.groupby(["ano_base", "turma"], dropna=False)[GROUP_VARS]
            .agg(["mean", "std", _p25, _p75])
        )
        turma_stats.columns = [f"turma_{stat}_{col}" for col, stat in turma_stats.columns]
        turma_stats = turma_stats.reset_index()

        fase_stats = (
            df.groupby(["ano_base", "fase"], dropna=False)[GROUP_VARS]
            .agg(["mean", "std", _p25, _p75])
        )
        fase_stats.columns = [f"fase_{stat}_{col}" for col, stat in fase_stats.columns]
        fase_stats = fase_stats.reset_index()

        global_stats = (
            df.groupby(["ano_base"], dropna=False)[GROUP_VARS]
            .agg(["mean", "std", _p25, _p75])
        )
        global_stats.columns = [f"global_{stat}_{col}" for col, stat in global_stats.columns]
        global_stats = global_stats.reset_index()

        lookup = {"turma": turma_stats, "fase": fase_stats, "global": global_stats}

    df = df.merge(lookup["turma"], on=["ano_base", "turma"], how="left")
    df = df.merge(lookup["fase"], on=["ano_base", "fase"], how="left")
    df = df.merge(lookup["global"], on=["ano_base"], how="left")

    for col in GROUP_VARS:
        for stat in ["mean", "std", "_p25", "_p75"]:
            c_turma = f"turma_{stat}_{col}"
            c_fase = f"fase_{stat}_{col}"
            c_global = f"global_{stat}_{col}"
            if c_turma in df.columns:
                df[c_turma] = df[c_turma].fillna(df[c_fase]).fillna(df[c_global])

    for col in GROUP_VARS:
        tm = f"turma_mean_{col}"
        ts = f"turma_std_{col}"
        df[f"delta_turma_{col}"] = df[col] - df[tm]
        denom_t = df[ts].where(df[ts] > EPS)
        df[f"z_turma_{col}"] = ((df[col] - df[tm]) / denom_t).fillna(0.0)

        fm = f"fase_mean_{col}"
        fs = f"fase_std_{col}"
        df[f"delta_fase_{col}"] = df[col] - df[fm]
        denom_f = df[fs].where(df[fs] > EPS)
        df[f"z_fase_{col}"] = ((df[col] - df[fm]) / denom_f).fillna(0.0)

    for col in ["media_provas", "matem", "ieg"]:
        p25 = f"turma__p25_{col}"
        if p25 in df.columns:
            df[f"abaixo_p25_turma_{col}"] = (df[col] < df[p25]).astype("Int64")
        else:
            df[f"abaixo_p25_turma_{col}"] = 0

    id_cols = ["ra", "ano_base", "pair_label", "fase", "turma", "target"]
    base_cols = [
        "genero", "instituicao", "escola", "ano_ingresso", "tempo_casa",
        "ano_nasc", "idade",
        "defasagem_t", "defasagem_t_is0", "defasagem_t_bin_1", "defasagem_t_bin_2plus",
        "matem", "portug", "ingles", "fez_ingles", "media_provas", "disp_provas",
        "ieg", "iaa", "ips", "ipp",
    ]
    derived_cols = [c for c in df.columns if c.startswith(("turma_", "fase_", "global_", "delta_", "z_", "abaixo_p25_"))]
    keep = [c for c in id_cols if c in df.columns] + [c for c in base_cols if c in df.columns] + derived_cols
    keep = [c for c in keep if c != "defasagem_t1"]

    out = df[keep].copy()

    # drop columns that are 100% NaN in train and mirror this in valid
    if is_train:
        all_nan_cols = out.columns[out.isna().all()].tolist()
        lookup["_all_nan_cols"] = all_nan_cols
    for c in lookup.get("_all_nan_cols", []):
        if c in out.columns:
            out = out.drop(columns=[c])

    # final numeric fill (except target)
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if col == "target":
            continue
        if is_train:
            fill_val = out[col].median()
            fill_val = 0.0 if pd.isna(fill_val) else float(fill_val)
        else:
            fill_val = 0.0
        out[col] = out[col].fillna(fill_val)

    for col in CAT_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype(str).fillna("MISSING_CATEGORY")

    logger.info("Features built: n=%d rows, %d cols", len(out), len(out.columns))
    return out, lookup


def build_features(
    df: pd.DataFrame,
    lookup_tables: Optional[dict] = None,
    is_train: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Compatibility wrapper keeping old function name/signature."""
    return make_features(df, lookup=lookup_tables, is_train=is_train)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {"ra", "ano_base", "pair_label", "target", "defasagem_t1"}
    cat_feats = [c for c in CAT_FEATURES if c in df.columns]
    num_feats = [
        c for c in df.columns
        if c not in exclude and c not in cat_feats and pd.api.types.is_numeric_dtype(df[c])
    ]
    return num_feats, cat_feats
