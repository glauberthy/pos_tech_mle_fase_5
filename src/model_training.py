"""
CatBoost training and inference helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier, Pool

    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False
    logger.warning("catboost not installed - training disabled")


DEFAULT_PARAMS = dict(
    iterations=2500,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    auto_class_weights="Balanced",
    bootstrap_type="Bernoulli",
    subsample=0.8,
    rsm=0.8,
    l2_leaf_reg=6.0,
    early_stopping_rounds=200,
    verbose=200,
    allow_writing_files=False,
)


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    work = df.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = np.nan
    x = work[feature_cols].copy()
    y = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int).values

    num_features = [c for c in feature_cols if c not in cat_features]
    for col in cat_features:
        x[col] = x[col].astype("object").fillna("MISSING_CATEGORY").astype("category")
    for col in num_features:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x, y


def train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    params: Optional[dict] = None,
    model_dir: str | Path = "models",
):
    if not _CATBOOST_AVAILABLE:
        raise ImportError("catboost must be installed: pip install catboost")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = num_features + cat_features
    x_train, y_train = _prepare_xy(train_df, feature_cols, cat_features)
    x_valid, y_valid = _prepare_xy(valid_df, feature_cols, cat_features)

    cat_idx = [x_train.columns.get_loc(c) for c in cat_features]
    train_pool = Pool(x_train, y_train, cat_features=cat_idx, feature_names=feature_cols)
    valid_pool = Pool(x_valid, y_valid, cat_features=cat_idx, feature_names=feature_cols)

    hp = {**DEFAULT_PARAMS, **(params or {})}
    model = CatBoostClassifier(**hp)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    model_path = model_dir / "catboost_model.cbm"
    model.save_model(str(model_path))
    logger.info("Model saved -> %s", model_path)

    meta = {
        "feature_cols": feature_cols,
        "num_features": num_features,
        "cat_features": cat_features,
        "best_iteration": model.get_best_iteration(),
        "best_score": model.get_best_score(),
        "params": hp,
    }
    with open(model_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return model, train_pool, valid_pool


def load_model(model_dir: str | Path = "models"):
    if not _CATBOOST_AVAILABLE:
        raise ImportError("catboost must be installed: pip install catboost")

    model_dir = Path(model_dir)
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "catboost_model.cbm"))

    with open(model_dir / "model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def predict_proba(
    model,
    df: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
) -> np.ndarray:
    if not _CATBOOST_AVAILABLE:
        raise ImportError("catboost must be installed: pip install catboost")

    feature_cols = num_features + cat_features
    x, _ = _prepare_xy(df.assign(target=0), feature_cols, cat_features)
    cat_idx = [x.columns.get_loc(c) for c in cat_features]
    pool = Pool(x, cat_features=cat_idx, feature_names=feature_cols)
    return model.predict_proba(pool)[:, 1]
