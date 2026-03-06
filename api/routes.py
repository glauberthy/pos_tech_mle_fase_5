"""Modular routes and shared state for the FastAPI service.

This module exports ``router`` (an :class:`APIRouter`) and a
``load_state`` helper which is executed on startup by ``api.main``.
All of the endpoint handlers, global constants and monitoring logic
were factored out of ``api/main.py`` so that the application
initialisation remains thin.
"""

import logging
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

# lazy imports from src (available after pip install -e .)
from src.feature_engineering import build_features
from src.model_training import load_model
from src.inference import score_students, alert_list, explain_student
from src.evaluation import stratified_topk_alert
from src.monitoring.logging import log_event
from src.utils import (
    load_pickle,
    load_json,
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
BASELINES_SUBDIR = MODEL_DIR / "baselines"
PRODUCTION_SNAPSHOTS_DIR = MODEL_DIR / "production_snapshots"
MONITORING_LOG = MONITORING_SUBDIR / "monitoring.log"
DRIFT_HISTORY_JSONL = MONITORING_SUBDIR / "drift_history.jsonl"
CURRENT_BASELINE_JSON = BASELINES_SUBDIR / "current_baseline.json"

# make sure the directories exist as soon as the module is imported
for _d in (MODEL_SUBDIR, EVAL_SUBDIR, MONITORING_SUBDIR, BASELINES_SUBDIR, PRODUCTION_SNAPSHOTS_DIR):
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
_baseline_id: Optional[str] = None
_model_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# helpers used by multiple endpoints
# ---------------------------------------------------------------------------

def _compute_phase_drift(
    scored_df: pd.DataFrame,
    baseline_by_phase: Optional[dict[str, np.ndarray]] = None,
) -> list[dict]:
    """Compute PSI drift per phase for phases with sufficient baseline and current data."""
    if "fase" not in scored_df.columns or "score" not in scored_df.columns:
        return []

    phase_results: list[dict] = []
    for fase, grp in scored_df.groupby("fase"):
        phase_key = str(fase)
        baseline_map = baseline_by_phase or _baseline_scores_by_phase
        baseline = baseline_map.get(phase_key)
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _parse_window_days(window: str) -> int:
    text = str(window).strip().lower()
    if text.endswith("d") and text[:-1].isdigit():
        days = int(text[:-1])
        if 1 <= days <= 365:
            return days
    raise HTTPException(status_code=400, detail="Invalid window. Use format '<n>d', e.g. '30d'.")


def _load_production_scores(window_days: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=window_days - 1)
    rows: list[dict] = []
    for partition in sorted(PRODUCTION_SNAPSHOTS_DIR.glob("dt=*")):
        if not partition.is_dir():
            continue
        date_txt = partition.name.split("=", 1)[-1]
        try:
            dt = datetime.strptime(date_txt, "%Y-%m-%d").date()
        except ValueError:
            continue
        if dt < cutoff:
            continue

        jsonl_path = partition / "predict_events.jsonl"
        for row in load_jsonl(jsonl_path):
            if isinstance(row, dict):
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    return df


def _persist_production_snapshots(
    input_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    request_id: str,
    k_pct: float,
) -> None:
    """Persist request/score rows partitioned by date (dt=YYYY-MM-DD)."""
    ts = datetime.now(timezone.utc)
    dt = ts.strftime("%Y-%m-%d")
    partition = PRODUCTION_SNAPSHOTS_DIR / f"dt={dt}"
    target = partition / "predict_events.jsonl"

    left = input_df.reset_index(drop=True).copy()
    right_cols = [c for c in ("score", "alerta") if c in scored_df.columns]
    right = scored_df[right_cols].reset_index(drop=True).copy()
    merged = pd.concat([left, right], axis=1)

    common_meta = {
        "timestamp_utc": ts.isoformat(),
        "dt": dt,
        "request_id": request_id,
        "baseline_id": _baseline_id,
        "model_version": _model_version,
        "k_pct": float(k_pct),
        "source": "api_predict",
    }
    for row in merged.to_dict(orient="records"):
        payload = {**common_meta, **row}
        _append_jsonl(target, payload)


# ---------------------------------------------------------------------------
# startup helper
# ---------------------------------------------------------------------------

def load_state() -> None:
    """Load model artefacts and baseline scores during application startup."""
    global _model, _meta, _lookup_tables, _baseline_scores, _baseline_scores_by_phase, _baseline_id, _model_version

    model_hash = ""
    try:
        _model, _meta = load_model(MODEL_SUBDIR)
        if isinstance(_meta, dict):
            _model_version = str(_meta.get("model_version", "1.0.0"))
        model_path = MODEL_SUBDIR / "catboost_model.cbm"
        if not model_path.exists():
            model_path = next(MODEL_SUBDIR.glob("*.cbm"), None) or next(MODEL_SUBDIR.glob("*.pkl"), None)
        if model_path and Path(model_path).exists():
            model_hash = _sha256_file(Path(model_path))[:16]
    except Exception as e:
        log_event(
            logger, logging.ERROR, "Failed to load model",
            event_type="error",
            error_type=type(e).__name__,
            status_code=503,
            message=str(e),
        )

    lookup_path = MODEL_SUBDIR / "lookup_tables.pkl"
    if lookup_path.exists():
        _lookup_tables = load_pickle(lookup_path)

    # Baseline scores for drift monitoring (prefer immutable baseline pointer)
    vdf: Optional[pd.DataFrame] = None
    if CURRENT_BASELINE_JSON.exists():
        try:
            ptr = load_json(CURRENT_BASELINE_JSON)
            ptr_baseline_id = str(ptr.get("baseline_id", "")).strip()
            if ptr_baseline_id:
                baseline_dir = BASELINES_SUBDIR / ptr_baseline_id
                baseline_csv = baseline_dir / "baseline.csv"
                baseline_manifest = baseline_dir / "baseline_manifest.json"
                if baseline_csv.exists() and baseline_manifest.exists():
                    manifest = load_json(baseline_manifest)
                    expected_hash = str(manifest.get("baseline_file_sha256", "")).strip()
                    if expected_hash:
                        actual_hash = _sha256_file(baseline_csv)
                        if actual_hash != expected_hash:
                            log_event(
                                logger, logging.ERROR, "Baseline hash mismatch",
                                event_type="error",
                                error_type="HashMismatch",
                                status_code=500,
                                message=f"expected={expected_hash[:16]} actual={actual_hash[:16]}",
                            )
                        else:
                            vdf = pd.read_csv(baseline_csv)
                            _baseline_id = ptr_baseline_id
                    else:
                        log_event(
                            logger, logging.ERROR, "Baseline manifest missing baseline_file_sha256",
                            event_type="error",
                            error_type="ConfigError",
                            status_code=500,
                            message=str(baseline_manifest),
                        )
                else:
                    log_event(
                        logger, logging.ERROR, "Baseline pointer invalid",
                        event_type="error",
                        error_type="ConfigError",
                        status_code=500,
                        message=str(baseline_dir),
                    )
        except Exception as e:
            log_event(
                logger, logging.ERROR, "Failed to load immutable baseline pointer",
                event_type="error",
                error_type=type(e).__name__,
                status_code=500,
                message=str(e),
            )

    # Backward-compatible fallback.
    if vdf is None:
        scored_csv = EVAL_SUBDIR / "valid_scored.csv"
        if scored_csv.exists():
            vdf = pd.read_csv(scored_csv)
            _baseline_id = "legacy_valid_scored"

    if vdf is not None and "score" in vdf.columns:
        _baseline_scores = vdf["score"].values / 100.0
        if "fase" in vdf.columns:
            _baseline_scores_by_phase = {
                str(fase): grp["score"].values / 100.0
                for fase, grp in vdf.groupby("fase")
                if len(grp) >= 5
            }

        # Seed a first monitoring event so dashboard is not empty.
        if not DRIFT_HISTORY_JSONL.exists():
            seed_drift = monitor_drift(_baseline_scores, _baseline_scores)
            seed_event = log_monitoring_event(
                DRIFT_HISTORY_JSONL,
                event_type="baseline_snapshot",
                data={
                    "baseline_id": _baseline_id,
                    "n_students": int(len(vdf)),
                    "k_pct": None,
                    "n_alerta": None,
                    "phase_drift": _compute_baseline_phase_snapshot(vdf),
                    **seed_drift,
                },
            )
            log_event(
                logger, logging.INFO, "Monitoring baseline snapshot created",
                event_type="drift",
                psi=seed_event["psi"],
                psi_severity=seed_event["severity"],
                baseline_id=_baseline_id,
                window="seed",
                n_records=seed_event["n_students"],
            )

    # Startup completo
    n_tables = len(_lookup_tables) if isinstance(_lookup_tables, dict) and _lookup_tables else 0
    log_event(
        logger, logging.INFO, "Startup complete",
        event_type="startup",
        model_version=_model_version,
        model_hash=model_hash or "n.a.",
        baseline_id=_baseline_id or "n.a.",
        tables=n_tables,
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
        "baseline_id": _baseline_id,
    }


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    request_id = uuid4().hex[:8]
    t0 = time.perf_counter()

    # Convert input to DataFrame
    rows = [s.model_dump() for s in request.students]
    df = pd.DataFrame(rows)

    # Add columns required by feature engineering
    df["defasagem"] = 0  # Inference: we don't know next year's defasagem

    # Feature engineering
    t_fe = time.perf_counter()
    fe_df, _ = build_features(df, lookup_tables=_lookup_tables, is_train=False, request_id=request_id)
    fe_ms = int((time.perf_counter() - t_fe) * 1000)

    # Score
    t_infer = time.perf_counter()
    scored = score_students(fe_df, model_dir=MODEL_SUBDIR, lookup_tables=_lookup_tables)
    infer_ms = int((time.perf_counter() - t_infer) * 1000)

    # Alert
    scored = alert_list(scored, k_pct=request.k_pct, fase_col="fase")

    latency_ms = int((time.perf_counter() - t0) * 1000)
    req_id_full = f"req_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{request_id}"

    # Score stats
    scores = pd.to_numeric(scored["score"], errors="coerce").dropna()
    score_mean = float(scores.mean()) if len(scores) else 0.0
    score_p90 = float(scores.quantile(0.9)) if len(scores) else 0.0
    score_p99 = float(scores.quantile(0.99)) if len(scores) else 0.0

    try:
        _persist_production_snapshots(df, scored, request_id=req_id_full, k_pct=request.k_pct)
    except Exception as e:
        log_event(
            logger, logging.ERROR, "Failed to persist production snapshots",
            event_type="error",
            request_id=request_id,
            error_type=type(e).__name__,
            status_code=500,
            message=str(e),
        )

    log_event(
        logger, logging.INFO, "Predict batch complete",
        event_type="predict_batch",
        request_id=request_id,
        n_records=len(scored),
        latency_ms=latency_ms,
        fe_ms=fe_ms,
        infer_ms=infer_ms,
        score_mean=round(score_mean, 4),
        score_p90=round(score_p90, 4),
        score_p99=round(score_p99, 4),
        model_version=_model_version,
        status_code=200,
    )

    if _baseline_scores is not None and "score" in scored.columns:
        current_scores = scored["score"].values / 100.0
        drift = monitor_drift(_baseline_scores, current_scores)
        phase_drift = _compute_phase_drift(scored)
        drift_event = log_monitoring_event(
            DRIFT_HISTORY_JSONL,
            event_type="predict_batch",
            data={
                "request_id": req_id_full,
                "baseline_id": _baseline_id,
                "model_version": _model_version,
                "n_students": len(scored),
                "k_pct": request.k_pct,
                "n_alerta": int(scored["alerta"].sum()) if "alerta" in scored.columns else 0,
                "phase_drift": phase_drift,
                **drift,
            },
        )
        psi_note = "low_sample" if len(scored) < 100 else None
        log_event(
            logger, logging.INFO, "Drift computed",
            event_type="drift",
            request_id=request_id,
            psi=drift_event["psi"],
            psi_severity=drift_event["severity"],
            baseline_id=_baseline_id,
            window="batch",
            n_records=len(scored),
            psi_note=psi_note,
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
    request_id = uuid4().hex[:8]
    scored_csv = EVAL_SUBDIR / "valid_scored.csv"
    if not scored_csv.exists():
        raise HTTPException(status_code=404, detail="Scored validation CSV not found. Run train.py first.")

    df = pd.read_csv(scored_csv)
    if "fase" not in df.columns or "score" not in df.columns:
        raise HTTPException(status_code=500, detail="Invalid scored CSV format")

    alerted = stratified_topk_alert(df, score_col="score", fase_col="fase", k_pct=k_pct)
    # boolean mask may be int dtype; convert to bool and use .loc to avoid
    # pandas treating the array as column selector (which raises KeyError).
    mask = alerted["alerta"].astype(bool)
    alerted_only = alerted.loc[mask].sort_values(["fase", "score"], ascending=[True, False])
    n_alerta = int(alerted["alerta"].sum())
    score_threshold = float(alerted_only["score"].min()) if not alerted_only.empty else 0.0

    log_event(
        logger, logging.INFO, "Alert list generated",
        event_type="alert",
        request_id=request_id,
        top_k=k_pct,
        alerts_generated=n_alerta,
        score_threshold=round(score_threshold, 4),
    )

    return {
        "k_pct": k_pct,
        "n_alerta": n_alerta,
        "n_total": len(alerted),
        "students": alerted_only[["ra", "fase", "turma", "score"]].to_dict(orient="records"),
    }


@router.get("/explain/{ra}")
def get_explanation(ra: str):
    """SHAP explanation for a specific student from validation set."""
    request_id = uuid4().hex[:8]
    scored_csv = EVAL_SUBDIR / "valid_scored.csv"
    if not scored_csv.exists():
        raise HTTPException(status_code=404, detail="Scored CSV not found. Run train.py first.")

    df = pd.read_csv(scored_csv)
    result = explain_student(ra, df)
    if "error" in result:
        log_event(
            logger, logging.WARNING, "Explain not found",
            event_type="explain",
            request_id=request_id,
            ra=ra,
            status_code=404,
        )
        raise HTTPException(status_code=404, detail=result["error"])
    log_event(
        logger, logging.INFO, "Explain generated",
        event_type="explain",
        request_id=request_id,
        ra=ra,
        status_code=200,
    )
    return result


@router.get("/metrics/drift")
def drift_report(
    mode: str = Query(default="internal", pattern="^(internal|oot|prod)$"),
    window: str = Query(default="30d"),
):
    """PSI drift report for internal (validation), oot (train vs validation) or prod (recent snapshots)."""
    source = "validation"
    effective_window = None

    if mode == "internal":
        scored_csv = EVAL_SUBDIR / "valid_scored.csv"
        if not scored_csv.exists():
            return {"status": "no_current_data", "mode": mode}
        df = pd.read_csv(scored_csv)
        baseline_scores = pd.to_numeric(df["score"], errors="coerce").dropna().values / 100.0
        current_scores = baseline_scores.copy()
        if "fase" in df.columns:
            baseline_by_phase = {
                str(fase): grp["score"].values / 100.0
                for fase, grp in df.groupby("fase")
                if len(grp) >= 5
            }
        else:
            baseline_by_phase = {}
    elif mode == "oot":
        source = "oot"
        train_csv = EVAL_SUBDIR / "train_scored.csv"
        valid_csv = EVAL_SUBDIR / "valid_scored.csv"
        if not train_csv.exists():
            raise HTTPException(status_code=400, detail="train_scored.csv not found. Run train.py first.")
        if not valid_csv.exists():
            return {"status": "no_current_data", "mode": mode}
        train_df = pd.read_csv(train_csv)
        df = pd.read_csv(valid_csv)
        baseline_scores = pd.to_numeric(train_df["score"], errors="coerce").dropna().values / 100.0
        current_scores = pd.to_numeric(df["score"], errors="coerce").dropna().values / 100.0
        if "fase" in train_df.columns:
            baseline_by_phase = {
                str(fase): grp["score"].values / 100.0
                for fase, grp in train_df.groupby("fase")
                if len(grp) >= 5
            }
        else:
            baseline_by_phase = {}
    else:
        if _baseline_scores is None:
            raise HTTPException(status_code=404, detail="No baseline scores. Run train.py first.")
        window_days = _parse_window_days(window)
        effective_window = f"{window_days}d"
        source = "production_snapshots"
        df = _load_production_scores(window_days=window_days)
        if df.empty or "score" not in df.columns:
            return {"status": "no_current_data", "mode": mode, "window": effective_window}
        baseline_scores = _baseline_scores
        current_scores = pd.to_numeric(df["score"], errors="coerce").dropna().values / 100.0
        baseline_by_phase = _baseline_scores_by_phase

    if len(current_scores) == 0 or len(baseline_scores) == 0:
        return {"status": "no_current_data", "mode": mode, "window": effective_window}
    result = monitor_drift(baseline_scores, current_scores)
    phase_drift = _compute_phase_drift(df, baseline_by_phase=baseline_by_phase)
    response = {
        **result,
        "mode": mode,
        "window": effective_window,
        "source": source,
        "baseline_id": _baseline_id if mode == "prod" else None,
        "phase_drift": phase_drift,
        "history_file": str(DRIFT_HISTORY_JSONL),
    }
    if mode == "oot":
        request_id = uuid4().hex[:8]
        oot_event = log_monitoring_event(
            DRIFT_HISTORY_JSONL,
            event_type="oot_snapshot",
            data={
                "baseline_id": "train_scored",
                "n_students": int(len(current_scores)),
                "n_baseline": result.get("n_baseline"),
                "n_current": result.get("n_current"),
                "phase_drift": phase_drift,
                **result,
            },
        )
        n_rec = int(len(current_scores))
        log_event(
            logger, logging.INFO, "OOT snapshot logged",
            event_type="drift",
            request_id=request_id,
            psi=oot_event["psi"],
            psi_severity=oot_event["severity"],
            baseline_id="train_scored",
            window="oot",
            n_records=n_rec,
            psi_note="low_sample" if n_rec < 100 else None,
        )
    return response


@router.get("/metrics/drift/history")
def drift_history(
    limit: int = Query(default=100, ge=1, le=1000),
    mode: str = Query(default="prod", pattern="^(internal|oot|prod)$"),
    window: str = Query(default="30d"),
):
    """Return latest monitoring drift events."""
    events = load_jsonl(DRIFT_HISTORY_JSONL)
    if not events:
        return {"n_events": 0, "events": [], "mode": mode}

    if mode == "internal":
        filtered = [e for e in events if str(e.get("event_type")) == "baseline_snapshot"]
        effective_window = None
    elif mode == "oot":
        filtered = [e for e in events if str(e.get("event_type")) == "oot_snapshot"]
        effective_window = None
    else:
        filtered = [e for e in events if str(e.get("event_type")) == "predict_batch"]
        window_days = _parse_window_days(window)
        effective_window = f"{window_days}d"
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        tmp: list[dict] = []
        for e in filtered:
            ts_txt = str(e.get("timestamp_utc", "")).strip()
            try:
                ts = datetime.fromisoformat(ts_txt.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts >= cutoff:
                tmp.append(e)
        filtered = tmp

    return {
        "n_events": len(filtered),
        "mode": mode,
        "window": effective_window,
        "events": filtered[-limit:],
    }


@router.get("/metrics/logs")
def monitoring_logs(
    lines: int = Query(default=80, ge=1, le=500),
):
    """Return last N lines of monitoring.log and parsed KPIs (requests, errors, p95 latency)."""
    import re as _re

    raw_lines: list[str] = []
    if MONITORING_LOG.exists():
        try:
            all_lines = MONITORING_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
            raw_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        except Exception:
            pass

    # parse KPIs from the full file
    n_requests = 0
    n_errors = 0
    latencies: list[float] = []
    if MONITORING_LOG.exists():
        try:
            text = MONITORING_LOG.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if "event_type" not in line:
                    continue
                if "event_type=error" in line:
                    n_errors += 1
                elif "event_type=predict_batch" in line and "Predict batch complete" in line:
                    n_requests += 1
                    m = _re.search(r"latency_ms=(\d+)", line)
                    if m:
                        latencies.append(float(m.group(1)))
        except Exception:
            pass

    import numpy as _np
    p95 = float(_np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "n_lines": len(raw_lines),
        "content": "\n".join(reversed(raw_lines)),
        "kpis": {
            "n_requests": n_requests,
            "n_errors": n_errors,
            "p95_latency_ms": round(p95, 1),
        },
    }
