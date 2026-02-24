"""
train.py – Main training script.

Usage:
    python train.py [--xls PATH] [--model-dir DIR] [--log-level LEVEL]

Runs the full pipeline:
  1. Load & preprocess all three PEDE sheets
  2. Build longitudinal train/validation sets
  3. Feature engineering
  4. Train CatBoost (temporal split, no shuffle)
  5. Evaluate on validation set (2023→2024)
  6. Save model, lookup tables, evaluation report
"""

import argparse
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils import setup_logging, save_pickle, save_json
from src.preprocessing import load_all_years, build_longitudinal_dataset
from src.feature_engineering import build_features, get_feature_columns
from src.model_training import train
from src.evaluation import evaluate, print_report

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train defasagem risk model")
    p.add_argument(
        "--xls",
        default="BASE DE DADOS PEDE 2024 - DATATHON.xlsx",
        help="Path to PEDE XLS file",
    )
    p.add_argument("--model-dir", default="models", help="Directory to save model artefacts")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    xls_path = Path(args.xls)
    model_dir = Path(args.model_dir)
    model_subdir     = model_dir / "model"
    eval_subdir      = model_dir / "evaluation"
    monitoring_subdir = model_dir / "monitoring"
    for d in (model_subdir, eval_subdir, monitoring_subdir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading data from %s", xls_path)
    data = load_all_years(xls_path)
    for yr, df in data.items():
        logger.info("  %d: %d students", yr, len(df))

    # ------------------------------------------------------------------
    # 2. Build longitudinal dataset
    # ------------------------------------------------------------------
    logger.info("Building longitudinal train/validation sets")
    train_raw, valid_raw = build_longitudinal_dataset(data)
    logger.info("  Train (temporal, pares anteriores): %d rows", len(train_raw))
    logger.info("  Valid (par temporal mais recente): %d rows", len(valid_raw))

    # Cohort transparency summary by year (for dashboard explainability)
    pair_rows = pd.concat(
        [
            train_raw[[c for c in ["pair_label", "target"] if c in train_raw.columns]],
            valid_raw[[c for c in ["pair_label", "target"] if c in valid_raw.columns]],
        ],
        ignore_index=True,
        sort=False,
    )
    years = sorted(data.keys())
    cohort_summary: list[dict] = []
    for i in range(len(years) - 1):
        y = years[i]
        y_next = years[i + 1]
        pair_label = f"{y}->{y_next}"
        if "pair_label" in pair_rows.columns:
            pair_df = pair_rows[pair_rows["pair_label"] == pair_label]
        else:
            pair_df = pair_rows.iloc[0:0]
        raw_total = int(len(data[y]))
        eligible_ontrack = int((pd.to_numeric(data[y].get("defasagem"), errors="coerce") == 0).sum())
        matched = int(len(pair_df))
        n_target_1 = int(pd.to_numeric(pair_df.get("target"), errors="coerce").fillna(0).sum()) if "target" in pair_df.columns else 0
        cohort_summary.append(
            {
                "ano_base": int(y),
                "ano_previsto": int(y_next),
                "raw_total": raw_total,
                "eligible_ontrack": eligible_ontrack,
                "matched_next_year": matched,
                "used_in_model": matched,
                "n_target_1": n_target_1,
            }
        )

    save_json(cohort_summary, model_subdir / "cohort_summary.json")
    logger.info("Cohort summary saved -> %s", model_subdir / "cohort_summary.json")

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    logger.info("Building features (train)")
    train_fe, lookup_tables = build_features(train_raw, is_train=True)

    logger.info("Building features (validation)")
    valid_fe, _ = build_features(valid_raw, lookup_tables=lookup_tables, is_train=False)

    # Save lookup tables for inference
    save_pickle(lookup_tables, model_subdir / "lookup_tables.pkl")
    logger.info("Lookup tables saved -> %s", model_subdir / "lookup_tables.pkl")

    # ------------------------------------------------------------------
    # 4. Identify feature columns
    # ------------------------------------------------------------------
    num_features, cat_features = get_feature_columns(train_fe)
    logger.info(
        "Features: %d numeric + %d categorical = %d total",
        len(num_features), len(cat_features), len(num_features) + len(cat_features),
    )
    logger.info("Numeric  : %s", num_features)
    logger.info("Categoric: %s", cat_features)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    logger.info("Training CatBoostClassifier (temporal split, no shuffle)")
    model, train_pool, valid_pool = train(
        train_fe, valid_fe,
        num_features=num_features,
        cat_features=cat_features,
        model_dir=model_subdir,
    )

    # ------------------------------------------------------------------
    # 6. Evaluate on validation set
    # ------------------------------------------------------------------
    from src.model_training import predict_proba

    proba_train = predict_proba(model, train_fe, num_features, cat_features)
    proba = predict_proba(model, valid_fe, num_features, cat_features)
    train_scored = train_fe.copy()
    train_scored["score"] = proba_train * 100
    train_scored["dataset_split"] = "train"
    valid_fe = valid_fe.copy()
    valid_fe["score"] = proba * 100
    valid_fe["dataset_split"] = "valid"

    results = evaluate(valid_fe, score_col="score", target_col="target", fase_col="fase")
    report = print_report(results)

    save_json(results, eval_subdir / "evaluation_results.json")
    (eval_subdir / "evaluation_report.txt").write_text(report, encoding="utf-8")
    logger.info("Evaluation results saved -> %s", eval_subdir / "evaluation_results.json")

    # Also save scored validation set for dashboard
    cols_to_save = (
        ["ra", "ano_base", "fase", "turma", "score", "target"]
        + [c for c in valid_fe.columns if c not in
           ["ra", "ano_base", "fase", "turma", "score", "target"]]
    )
    cols_to_save = [c for c in cols_to_save if c in valid_fe.columns]
    valid_fe[cols_to_save].to_csv(eval_subdir / "valid_scored.csv", index=False)
    logger.info("Scored validation set saved -> %s", eval_subdir / "valid_scored.csv")

    # Save multi-year scored history for dashboard slicing by ano_base
    history_cols = (
        ["ra", "ano_base", "fase", "turma", "score", "target", "dataset_split"]
        + [c for c in valid_fe.columns if c not in
           ["ra", "ano_base", "fase", "turma", "score", "target", "dataset_split"]]
    )
    history_cols = [c for c in history_cols if c in valid_fe.columns or c in train_scored.columns]
    scored_history = pd.concat(
        [
            train_scored[[c for c in history_cols if c in train_scored.columns]],
            valid_fe[[c for c in history_cols if c in valid_fe.columns]],
        ],
        ignore_index=True,
        sort=False,
    )
    scored_history.to_csv(eval_subdir / "scored_history.csv", index=False)
    logger.info("Scored history saved -> %s", eval_subdir / "scored_history.csv")

    # ------------------------------------------------------------------
    # 7. Write monitoring bootstrap files
    # ------------------------------------------------------------------
    # retrain_metadata.json – mesma estrutura usada pelo dashboard no retreino
    xls_sha256 = hashlib.sha256(xls_path.read_bytes()).hexdigest() if xls_path.exists() else ""
    retrain_meta = {
        "last_file_name": xls_path.name,
        "last_file_sha256": xls_sha256,
        "last_trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "auc": round(results.get("auc", 0.0), 4),
        "n_students_valid": int(len(valid_fe)),
        "source": "train.py",
    }
    save_json(retrain_meta, monitoring_subdir / "retrain_metadata.json")
    logger.info("Retrain metadata saved -> %s", monitoring_subdir / "retrain_metadata.json")

    # drift_history.jsonl – inicializa vazio se não existir (a API vai preenchendo)
    drift_path = monitoring_subdir / "drift_history.jsonl"
    if not drift_path.exists():
        drift_path.touch()
        logger.info("drift_history.jsonl initialised (empty) -> %s", drift_path)

    logger.info("Training pipeline complete.")
    return results


if __name__ == "__main__":
    main()
