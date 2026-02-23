"""
Evaluation metrics and operational Top-K policy.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import topk_mask as _topk_mask

logger = logging.getLogger(__name__)


def recall_at_topk(y_true: np.ndarray, scores: np.ndarray, k_pct: float) -> float:
    mask = _topk_mask(scores, k_pct)
    tp = int((mask & (y_true == 1)).sum())
    pos = int((y_true == 1).sum())
    return float(tp / pos) if pos > 0 else 0.0


def precision_at_topk(y_true: np.ndarray, scores: np.ndarray, k_pct: float) -> float:
    mask = _topk_mask(scores, k_pct)
    tp = int((mask & (y_true == 1)).sum())
    k = int(mask.sum())
    return float(tp / k) if k > 0 else 0.0


def lift_at_topk(y_true: np.ndarray, scores: np.ndarray, k_pct: float) -> float:
    base = float(np.mean(y_true)) if len(y_true) else 0.0
    if base <= 0:
        return 0.0
    return float(precision_at_topk(y_true, scores, k_pct) / base)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg

    if hasattr(np, "trapezoid"):
        auc = np.trapezoid(tpr, fpr)
    else:
        auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5)
    return float(abs(auc))



def stratified_topk_alert(
    df: pd.DataFrame,
    score_col: str = "score",
    fase_col: str = "fase",
    k_pct: float = 15.0,
    alert_col: str = "alerta",
) -> pd.DataFrame:
    """Marca Top-K% por score dentro de cada fase (estratificado).

    O resultado é armazenado em ``alert_col`` como inteiro (1 = alerta, 0 = sem alerta).
    """
    out = df.copy()
    out[alert_col] = 0
    for fase, grp in out.groupby(fase_col, dropna=False):
        n = len(grp)
        if n == 0:
            continue
        k = max(1, int(np.ceil(n * k_pct / 100)))
        idx = grp.sort_values(score_col, ascending=False).head(k).index
        out.loc[idx, alert_col] = 1
    return out


def evaluate(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "target",
    fase_col: str = "fase",
    k_pcts: Optional[list[float]] = None,
) -> dict:
    if k_pcts is None:
        k_pcts = [10, 15, 20, 25]

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values
    s = pd.to_numeric(df[score_col], errors="coerce").fillna(0).values / 100.0

    results: dict = {
        "auc": roc_auc(y, s),
        "n_total": int(len(df)),
        "n_positive": int(y.sum()),
        "base_rate": float(np.mean(y)) if len(y) else 0.0,
    }

    for k in k_pcts:
        results[f"recall_top{k}"] = recall_at_topk(y, s, k)
        results[f"precision_top{k}"] = precision_at_topk(y, s, k)
        results[f"lift_top{k}"] = lift_at_topk(y, s, k)

    per_fase: dict = {}
    if fase_col in df.columns:
        for fase, grp in df.groupby(fase_col):
            yf = pd.to_numeric(grp[target_col], errors="coerce").fillna(0).astype(int).values
            sf = pd.to_numeric(grp[score_col], errors="coerce").fillna(0).values / 100.0
            per_fase[str(fase)] = {
                "n": int(len(grp)),
                "n_pos": int(yf.sum()),
                "auc": roc_auc(yf, sf),
                "recall_top15": recall_at_topk(yf, sf, 15),
                "precision_top15": precision_at_topk(yf, sf, 15),
            }
    results["per_fase"] = per_fase
    return results


def print_report(results: dict) -> str:
    lines = [
        "=" * 60,
        "AVALIACAO DO MODELO",
        "=" * 60,
        f"Total de alunos     : {results.get('n_total', 0)}",
        f"Positivos (target=1): {results.get('n_positive', 0)} ({100*results.get('base_rate', 0):.1f}%)",
        f"AUC                 : {results.get('auc', 0):.4f}",
        "",
        f"{'K%':>6} | {'Recall':>8} | {'Precision':>10} | {'Lift':>6}",
        "-" * 40,
    ]
    for k in [10, 15, 20, 25]:
        r = results.get(f"recall_top{k}")
        p = results.get(f"precision_top{k}")
        l = results.get(f"lift_top{k}")
        if r is None or p is None or l is None:
            continue
        lines.append(f"{k:>5}% | {r:>8.3f} | {p:>10.3f} | {l:>6.2f}")
    lines.append("=" * 60)
    report = "\n".join(lines)
    print(report)
    return report


# ============================================================
# Notebook-aligned helpers (evaluate.ipynb)
# ============================================================


def operational_metrics_topk(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "target",
    fase_col: str = "fase",
    k_pct: float = 15.0,
    alert_col: str = "alerta",
) -> dict:
    """Métricas globais com política estratificada por fase.

    Retorna dict com recall@k, precision@k, lift@k, TP/FP/FN e
    ``df_with_alerts`` (DataFrame com coluna de alerta marcada).
    """
    tmp = stratified_topk_alert(df, score_col=score_col, fase_col=fase_col, k_pct=k_pct, alert_col=alert_col)

    y = tmp[target_col].astype(int).values
    a = tmp[alert_col].astype(int).values

    tp = int(((a == 1) & (y == 1)).sum())
    fp = int(((a == 1) & (y == 0)).sum())
    fn = int(((a == 0) & (y == 1)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    base_rate = float(y.mean()) if len(y) else 0.0
    lift = (precision / base_rate) if base_rate > 0 else float("nan")

    return {
        "k_pct": float(k_pct),
        "n_total": int(len(y)),
        "n_alert": int(a.sum()),
        "n_pos": int(y.sum()),
        "base_rate": base_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall@k": float(recall),
        "precision@k": float(precision),
        "lift@k": float(lift) if np.isfinite(lift) else None,
        "df_with_alerts": tmp,
    }


def operational_metrics_topk_by_fase(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "target",
    fase_col: str = "fase",
    k_pct: float = 15.0,
    alert_col: str = "alerta",
) -> pd.DataFrame:
    """Métricas Top-K estratificadas por fase para diagnóstico operacional."""
    tmp = stratified_topk_alert(df, score_col=score_col, fase_col=fase_col, k_pct=k_pct, alert_col=alert_col)

    rows = []
    for fase, g in tmp.groupby(fase_col, dropna=False):
        y = g[target_col].astype(int).values
        a = g[alert_col].astype(int).values

        tp = int(((a == 1) & (y == 1)).sum())
        fp = int(((a == 1) & (y == 0)).sum())
        fn = int(((a == 0) & (y == 1)).sum())

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        base_rate = float(y.mean()) if len(y) else 0.0
        lift = (precision / base_rate) if base_rate > 0 else float("nan")

        rows.append({
            fase_col: fase,
            "n": int(len(g)),
            "n_alert": int(a.sum()),
            "n_pos": int(y.sum()),
            "base_rate": base_rate,
            "recall@k": float(recall),
            "precision@k": float(precision),
            "lift@k": float(lift) if np.isfinite(lift) else None,
        })

    return pd.DataFrame(rows).sort_values(by=fase_col).reset_index(drop=True)


def feature_importance_by_fase(  # pragma: no cover
    model,
    df: pd.DataFrame,
    feature_cols: list,
    cat_features: list,
    target_col: str = "target",
    fase_col: str = "fase",
    importance_type: str = "LossFunctionChange",
    min_n: int = 30,
    top_n: int = 20,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Calcula importância de features por fase usando CatBoost.

    Retorna:
      - long_df: importância em formato longo (fase, feature, importance)
      - top_df : top-N por fase, já ordenado
    """
    try:
        from catboost import Pool as _Pool
    except ImportError:
        raise ImportError("catboost must be installed: pip install catboost")

    long_rows = []
    fases = sorted(df[fase_col].dropna().unique().tolist())

    for fase in fases:
        g = df[df[fase_col] == fase].copy()
        n = len(g)

        X = g[feature_cols].copy()
        y = g[target_col].astype(int).values

        for c in cat_features:
            if c in X.columns:
                X[c] = X[c].astype(str)

        cat_idx = [X.columns.get_loc(c) for c in cat_features if c in X.columns]
        pool = _Pool(X, y, cat_features=cat_idx)

        imps = model.get_feature_importance(pool, type=importance_type)
        for feat, imp in zip(feature_cols, imps):
            long_rows.append({
                "fase": fase,
                "n": n,
                "feature": feat,
                "importance": float(imp),
                "importance_type": importance_type,
                "unstable": bool(n < min_n),
            })

    long_df = pd.DataFrame(long_rows)
    top_df = (
        long_df.sort_values(["fase", "importance"], ascending=[True, False])
        .groupby("fase", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return long_df, top_df


def shap_mean_by_fase(  # pragma: no cover
    model,
    df: pd.DataFrame,
    feature_cols: list,
    cat_features: list,
    target_col: str = "target",
    fase_col: str = "fase",
    sample_per_fase: "int | None" = 500,
    random_state: int = 42,
    top_n: int = 20,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Calcula SHAP médio por fase (com direção do efeito).

    Retorna:
      - long_df: (fase, n, feature, mean_shap, mean_abs_shap, pct_positive)
      - top_df : top-N por fase ordenado por mean_abs_shap
    """
    try:
        from catboost import Pool as _Pool
    except ImportError:
        raise ImportError("catboost must be installed: pip install catboost")

    rng = np.random.default_rng(random_state)
    rows = []
    fases = sorted(df[fase_col].dropna().unique().tolist())

    for fase in fases:
        g = df[df[fase_col] == fase].copy()
        n0 = len(g)
        if n0 == 0:
            continue

        if sample_per_fase is not None and n0 > sample_per_fase:
            idx = rng.choice(g.index.to_numpy(), size=sample_per_fase, replace=False)
            g = g.loc[idx].copy()

        n = len(g)
        X = g[feature_cols].copy()

        for c in cat_features:
            if c in X.columns:
                X[c] = X[c].astype(str)

        cat_idx = [X.columns.get_loc(c) for c in cat_features if c in X.columns]
        y = g[target_col].astype(int).values if target_col in g.columns else None
        pool = _Pool(X, label=y, cat_features=cat_idx)

        shap_matrix = model.get_feature_importance(pool, type="ShapValues")
        shap_vals = shap_matrix[:, :-1]  # (n, m) — última coluna é base value

        mean_shap = shap_vals.mean(axis=0)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        pct_pos = (shap_vals > 0).mean(axis=0)

        for feat, ms, ma, pp in zip(feature_cols, mean_shap, mean_abs, pct_pos):
            rows.append({
                "fase": fase,
                "n": n,
                "feature": feat,
                "mean_shap": float(ms),
                "mean_abs_shap": float(ma),
                "pct_positive": float(pp),
            })

    long_df = pd.DataFrame(rows)
    top_df = (
        long_df.sort_values(["fase", "mean_abs_shap"], ascending=[True, False])
        .groupby("fase", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return long_df, top_df
