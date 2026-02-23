"""
Utility helpers for persistence, logging and drift monitoring.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ── String / text utilities ───────────────────────────────────────────────────

def strip_accents(value: str) -> str:
    """Remove diacritics (accents) from a string via NFKD decomposition."""
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_colname(col: str) -> str:
    """Normalize a column name to lowercase ASCII with underscores.

    Steps: strip accents → lowercase → special chars → spaces → underscore
    → remove non-alnum → collapse underscores → strip leading/trailing.
    """
    col = strip_accents(col).strip().lower()
    col = re.sub(r"[\/\-\.,\(\)\[\]\{\}]+", " ", col)
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^a-z0-9_]", "", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


# ── Numeric / array utilities ─────────────────────────────────────────────────

def topk_mask(scores: np.ndarray, k_pct: float) -> np.ndarray:
    """Boolean mask selecting the top-K% elements by descending score.

    Always marks at least 1 element.  ``k_pct`` is a percentage (e.g. 15.0).
    """
    n = len(scores)
    k = max(1, int(np.ceil(n * k_pct / 100)))
    idx = np.argsort(scores)[::-1][:k]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def setup_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure root logging once for CLI/API usage."""
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def save_json(obj: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def phase_sort_key(phase: Any) -> int:
    """Sort ALFA/F0 first, then numeric phases."""
    s = str(phase).upper().strip()
    if s in {"ALFA", "F0", "FASE0", "0"}:
        return 0
    for token in s.replace("_", " ").split():
        if token.isdigit():
            return int(token)
        if token.startswith("FASE") and token[4:].isdigit():
            return int(token[4:])
        if token.startswith("F") and token[1:].isdigit():
            return int(token[1:])
    if s.isdigit():
        return int(s)
    return 99


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index on score distributions."""
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    if len(b) == 0 or len(c) == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    b_pct = np.histogram(b, bins=edges)[0] / max(len(b), 1)
    c_pct = np.histogram(c, bins=edges)[0] / max(len(c), 1)

    b_pct = np.clip(b_pct, eps, None)
    c_pct = np.clip(c_pct, eps, None)
    psi = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
    return float(psi)


def monitor_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.25,
) -> dict:
    """Drift summary object used by API/dashboard."""
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    psi = compute_psi(b, c)
    if psi < 0.1:
        severity = "low"
    elif psi < threshold:
        severity = "moderate"
    else:
        severity = "high"
    return {
        "psi": round(float(psi), 4),
        "threshold": float(threshold),
        "drift_detected": bool(psi >= threshold),
        "severity": severity,
        "baseline_mean": round(float(np.mean(b)) if len(b) else 0.0, 4),
        "current_mean": round(float(np.mean(c)) if len(c) else 0.0, 4),
        "baseline_std": round(float(np.std(b)) if len(b) else 0.0, 4),
        "current_std": round(float(np.std(c)) if len(c) else 0.0, 4),
        "n_baseline": int(len(b)),
        "n_current": int(len(c)),
        "status": "DRIFT DETECTED" if psi >= threshold else "stable",
    }


def log_monitoring_event(
    path: str | Path,
    event_type: str,
    data: dict,
) -> dict:
    """Append a structured monitoring event to JSONL."""
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **data,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload
