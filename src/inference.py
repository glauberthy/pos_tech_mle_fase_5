"""
Inference helpers for API/dashboard workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation import stratified_topk_alert
from src.model_training import load_model, predict_proba

logger = logging.getLogger(__name__)

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def score_students(
    df: pd.DataFrame,
    model_dir: str | Path = "models",
    lookup_tables: Optional[dict] = None,
) -> pd.DataFrame:
    model, meta = load_model(model_dir)
    num_features = meta.get("num_features", [])
    cat_features = meta.get("cat_features", [])

    proba = predict_proba(model, df, num_features, cat_features)
    out = df.copy()
    out["prob"] = proba
    out["score"] = (proba * 100).round(4)

    if _SHAP_AVAILABLE:
        out = _add_shap_explanations(out, model, num_features, cat_features)
    else:
        out["top3_factors"] = [[] for _ in range(len(out))]
        out["top3_values"] = [[] for _ in range(len(out))]
    return out


def alert_list(scored_df: pd.DataFrame, k_pct: float = 15.0, fase_col: str = "fase") -> pd.DataFrame:
    out = stratified_topk_alert(scored_df, score_col="score", fase_col=fase_col, k_pct=k_pct)
    return out.sort_values([fase_col, "score"], ascending=[True, False])


def explain_student(student_ra: str, scored_df: pd.DataFrame) -> dict:
    row = scored_df[scored_df["ra"].astype(str) == str(student_ra)]
    if row.empty:
        return {"error": f"RA {student_ra} not found"}
    r = row.iloc[0]
    top = []
    for feat, val in zip(r.get("top3_factors", []), r.get("top3_values", [])):
        top.append(
            {
                "feature": feat,
                "shap_value": float(val),
                "student_value": float(r[feat]) if feat in r.index and pd.notna(r[feat]) else None,
            }
        )
    return {
        "ra": str(r.get("ra")),
        "score": float(r.get("score", 0)),
        "alerta": bool(r.get("alerta", False)),
        "fase": str(r.get("fase", "")),
        "top_factors": top,
    }


def _add_shap_explanations(
    df: pd.DataFrame,
    model,
    num_features: list[str],
    cat_features: list[str],
) -> pd.DataFrame:
    feats = num_features + cat_features
    work = df.copy()
    for col in feats:
        if col not in work.columns:
            work[col] = np.nan
    x = work[feats].copy()
    for col in cat_features:
        x[col] = x[col].astype(str)

    explainer = shap.TreeExplainer(model)
    try:
        shap_values = explainer.shap_values(x)
        sv = np.array(shap_values[1] if isinstance(shap_values, list) else shap_values)
    except Exception as exc:
        logger.warning("SHAP failed: %s", exc)
        df["top3_factors"] = [[] for _ in range(len(df))]
        df["top3_values"] = [[] for _ in range(len(df))]
        return df

    names = np.array(feats)
    top3_factors = []
    top3_values = []
    for i in range(len(df)):
        idx = np.argsort(np.abs(sv[i]))[::-1][:3]
        top3_factors.append(names[idx].tolist())
        top3_values.append(sv[i][idx].tolist())
    df["top3_factors"] = top3_factors
    df["top3_values"] = top3_values
    return df
