"""Modular routes and shared state for the FastAPI service.

This module exports ``router`` (an :class:`APIRouter`) and a
``load_state`` helper which is executed on startup by ``api.main``.
All of the endpoint handlers, global constants and monitoring logic
were factored out of ``api/main.py`` so that the application
initialisation remains thin.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

# lazy imports from src (available after pip install -e .)
from src.feature_engineering import build_features
from src.model_training import load_model
from src.inference import score_students, alert_list, explain_student
from src.evaluation import stratified_topk_alert
from src.utils import (
    load_pickle,
    monitor_drift,
    setup_logging,
    log_monitoring_event,
    load_jsonl,
)

from .schemas import (
    StudentInput,
    PredictRequest,
    StudentScore,
    PredictResponse,
)

router = APIRouter()

# filesystem layout constants used by startup and endpoints
MODEL_DIR = Path("models")
MODEL_SUBDIR = MODEL_DIR / "model"
EVAL_SUBDIR = MODEL_DIR / "evaluation"
MONITORING_SUBDIR = MODEL_DIR / "monitoring"
MONITORING_LOG = MONITORING_SUBDIR / "monitoring.log"
DRIFT_HISTORY_JSONL = MONITORING_SUBDIR / "drift_history.jsonl"

# make sure the directories exist as soon as the module is imported
for _d in (MODEL_SUBDIR, EVAL_SUBDIR, MONITORING_SUBDIR):
    _d.mkdir(parents=True, exist_ok=True)

# logging is configured once per process
setup_logging(log_file=MONITORING_LOG)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# state variables that live for the lifetime of the service
# ---------------------------------------------------------------------------

_model = None
_meta = None
_lookup_tables = None
_baseline_scores: Optional[np.ndarray] = None
_baseline_scores_by_phase: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# helpers used by multiple endpoints
# ---------------------------------------------------------------------------

def _compute_phase_drift(scored_df: pd.DataFrame) -> list[dict]:
    """Compute PSI drift per phase for phases with sufficient baseline and current data."""
    if "fase" not in scored_df.columns or "score" not in scored_df.columns:
        return []

    phase_results: list[dict] = []
    for fase, grp in scored_df.groupby("fase"):
        phase_key = str(fase)
        baseline = _baseline_scores_by_phase.get(phase_key)
        if baseline is None:
            continue

        current = grp["score"].values / 100.0
        if len(current) < 5 or len(baseline) < 5:
            continue

        drift = monitor_drift(baseline, current)
        phase_results.append(
            {
                "fase": phase_key,
                "n_current": int(len(current)),
                "n_baseline": int(len(baseline)),
                **drift,
            }
        )

    phase_results.sort(key=lambda x: x.get("psi", 0), reverse=True)
    return phase_results


def _compute_baseline_phase_snapshot(vdf: pd.DataFrame) -> list[dict]:
    """Create a zero-drift per-phase snapshot from baseline itself."""
    if "fase" not in vdf.columns or "score" not in vdf.columns:
        return []
    rows: list[dict] = []
    for fase, grp in vdf.groupby("fase"):
        scores = grp["score"].values / 100.0
        if len(scores) < 5:
            continue
        drift = monitor_drift(scores, scores)
        rows.append(
            {
                "fase": str(fase),
                "n_current": int(len(scores)),
                "n_baseline": int(len(scores)),
                **drift,
            }
        )
    rows.sort(key=lambda x: x.get("fase", ""))
    return rows


# ---------------------------------------------------------------------------
# startup helper
# ---------------------------------------------------------------------------

def load_state() -> None:
    """Load model artefacts and baseline scores during application startup."""
    global _model, _meta, _lookup_tables, _baseline_scores, _baseline_scores_by_phase

    try:
        _model, _meta = load_model(MODEL_SUBDIR)
        logger.info("Model loaded from %s", MODEL_SUBDIR)
    except Exception as e:
        logger.error("Failed to load model: %s", e)

    lookup_path = MODEL_SUBDIR / "lookup_tables.pkl"
    if lookup_path.exists():
        _lookup_tables = load_pickle(lookup_path)
        logger.info("Lookup tables loaded")

    # Baseline scores for drift monitoring
    scored_csv = EVAL_SUBDIR / "valid_scored.csv"
    if scored_csv.exists():
        vdf = pd.read_csv(scored_csv)
        if "score" in vdf.columns:
            _baseline_scores = vdf["score"].values / 100.0
        if "score" in vdf.columns and "fase" in vdf.columns:
            _baseline_scores_by_phase = {
                str(fase): grp["score"].values / 100.0
                for fase, grp in vdf.groupby("fase")
                if len(grp) >= 5
            }

        # Seed a first monitoring event so dashboard is not empty.
        if _baseline_scores is not None and not DRIFT_HISTORY_JSONL.exists():
            seed_drift = monitor_drift(_baseline_scores, _baseline_scores)
            seed_event = log_monitoring_event(
                DRIFT_HISTORY_JSONL,
                event_type="baseline_snapshot",
                data={
                    "n_students": int(len(vdf)),
                    "k_pct": None,
                    "n_alerta": None,
                    "phase_drift": _compute_baseline_phase_snapshot(vdf),
                    **seed_drift,
                },
            )
            logger.info(
                "Monitoring baseline snapshot created | psi=%.4f | n=%d",
                seed_event["psi"],
                seed_event["n_students"],
            )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "lookup_tables_loaded": _lookup_tables is not None,
    }


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    # Convert input to DataFrame
    rows = [s.model_dump() for s in request.students]
    df = pd.DataFrame(rows)

    # Add columns required by feature engineering
    df["defasagem"] = 0  # Inference: we don't know next year's defasagem

    # Feature engineering
    fe_df, _ = build_features(df, lookup_tables=_lookup_tables, is_train=False)

    # Score
    scored = score_students(fe_df, model_dir=MODEL_SUBDIR, lookup_tables=_lookup_tables)

    # Alert
    scored = alert_list(scored, k_pct=request.k_pct, fase_col="fase")

    if _baseline_scores is not None and "score" in scored.columns:
        current_scores = scored["score"].values / 100.0
        drift = monitor_drift(_baseline_scores, current_scores)
        phase_drift = _compute_phase_drift(scored)
        drift_event = log_monitoring_event(
            DRIFT_HISTORY_JSONL,
            event_type="predict_batch",
            data={
                "n_students": len(scored),
                "k_pct": request.k_pct,
                "n_alerta": int(scored["alerta"].sum()) if "alerta" in scored.columns else 0,
                "phase_drift": phase_drift,
                **drift,
            },
        )
        logger.info(
            "Monitoring event saved | type=%s | psi=%.4f | severity=%s | n=%d",
            drift_event["event_type"],
            drift_event["psi"],
            drift_event["severity"],
            drift_event["n_students"],
        )

    # Build response
    students_out = []
    for _, row in scored.iterrows():
        students_out.append(StudentScore(
            ra=str(row["ra"]),
            score=float(row["score"]),
            fase=str(row.get("fase", "")),
            turma=str(row.get("turma", "")),
            alerta=bool(row.get("alerta", False)),
            top3_factors=list(row.get("top3_factors", [])),
            top3_values=[float(v) for v in row.get("top3_values", [])],
        ))

    return PredictResponse(
        n_students=len(students_out),
        n_alerta=sum(1 for s in students_out if s.alerta),
        k_pct=request.k_pct,
        students=students_out,
    )


@router.get("/alert")
def get_alert(k_pct: float = Query(default=15.0, ge=5.0, le=50.0)):
    """Return stratified alert list from validation set (demo mode)."""
    scored_csv = EVAL_SUBDIR / "valid_scored.csv"
    if not scored_csv.exists():
        raise HTTPException(status_code=404, detail="Scored validation CSV not found. Run train.py first.")

    df = pd.read_csv(scored_csv)
    if "fase" not in df.columns or "score" not in df.columns:
        raise HTTPException(status_code=500, detail="Invalid scored CSV format")

    alerted = stratified_topk_alert(df, score_col="score", fase_col="fase", k_pct=k_pct)
    alerted_only = alerted[alerted["alerta"]].sort_values(["fase", "score"], ascending=[True, False])

    return {
        "k_pct": k_pct,
        "n_alerta": int(alerted["alerta"].sum()),
        "n_total": len(alerted),
        "students": alerted_only[["ra", "fase", "turma", "score"]].to_dict(orient="records"),
    }


@router.get("/explain/{ra}")
def get_explanation(ra: str):
    """SHAP explanation for a specific student from validation set."""
    scored_csv = EVAL_SUBDIR / "valid_scored.csv"
    if not scored_csv.exists():
        raise HTTPException(status_code=404, detail="Scored CSV not found. Run train.py first.")

    df = pd.read_csv(scored_csv)
    result = explain_student(ra, df)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/metrics/drift")
def drift_report():
    """PSI drift report comparing current scores to baseline (validation set)."""
    if _baseline_scores is None:
        raise HTTPException(status_code=404, detail="No baseline scores. Run train.py first.")

    scored_csv = EVAL_SUBDIR / "valid_scored.csv"
    if not scored_csv.exists():
        return {"status": "no_current_data"}

    df = pd.read_csv(scored_csv)
    current_scores = df["score"].values / 100.0
    result = monitor_drift(_baseline_scores, current_scores)
    phase_drift = _compute_phase_drift(df)
    return {
        **result,
        "phase_drift": phase_drift,
        "history_file": str(DRIFT_HISTORY_JSONL),
    }


@router.get("/metrics/drift/history")
def drift_history(limit: int = Query(default=100, ge=1, le=1000)):
    """Return latest monitoring drift events."""
    events = load_jsonl(DRIFT_HISTORY_JSONL)
    if not events:
        return {"n_events": 0, "events": []}
    return {
        "n_events": len(events),
        "events": events[-limit:],
    }
