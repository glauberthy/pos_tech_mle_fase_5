from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation import stratified_topk_alert
from src.model_training import load_model, predict_proba

logger = logging.getLogger(__name__)

_SHAP_AVAILABLE: Optional[bool] = None


def _get_shap():
    global _SHAP_AVAILABLE
    try:
        import shap  # type: ignore
        _SHAP_AVAILABLE = True
        return shap
    except Exception as exc:
        _SHAP_AVAILABLE = False
        logger.warning("SHAP indisponível: %s", exc)
        return None


def _empty_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["top3_factors"] = [[] for _ in range(len(out))]
    out["top3_values"] = [[] for _ in range(len(out))]
    return out


def score_students(
    df: pd.DataFrame,
    model=None,
    meta: Optional[dict] = None,
    model_dir: str | Path = "models",
    lookup_tables: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Gera score dos alunos.

    Preferência:
    - se `model` e `meta` vierem preenchidos, reutiliza os objetos já carregados
    - caso contrário, carrega do `model_dir`
    """
    if model is None or meta is None:
        model, meta = load_model(model_dir)

    num_features = meta.get("num_features", [])
    cat_features = meta.get("cat_features", [])

    proba = predict_proba(model, df, num_features, cat_features)
    out = df.copy()
    out["prob"] = proba
    out["score"] = (proba * 100).round(4)

    explain_enabled = os.getenv("ENABLE_SHAP", "0").strip() == "1"
    if not explain_enabled:
        return _empty_explanations(out)

    return _add_shap_explanations(out, model, num_features, cat_features)


def alert_list(
    scored_df: pd.DataFrame,
    k_pct: float = 15.0,
    fase_col: str = "fase",
) -> pd.DataFrame:
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
    shap = _get_shap()
    if shap is None:
        return _empty_explanations(df)

    feats = num_features + cat_features
    work = df.copy()

    for col in feats:
        if col not in work.columns:
            work[col] = np.nan

    x = work[feats].copy()
    for col in cat_features:
        x[col] = x[col].astype(str)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
        sv = np.array(shap_values[1] if isinstance(shap_values, list) else shap_values)
    except Exception as exc:
        logger.warning("SHAP falhou: %s", exc)
        return _empty_explanations(df)

    names = np.array(feats)
    top3_factors: list[list[str]] = []
    top3_values: list[list[float]] = []

    for i in range(len(df)):
        idx = np.argsort(np.abs(sv[i]))[::-1][:3]
        top3_factors.append(names[idx].tolist())
        top3_values.append([float(v) for v in sv[i][idx].tolist()])

    out = df.copy()
    out["top3_factors"] = top3_factors
    out["top3_values"] = top3_values
    return out