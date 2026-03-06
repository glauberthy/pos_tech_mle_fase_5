"""
Dashboard t4 — Monitoramento Drift × Coerência (Bootstrap).
Ajustes principais vs v3:
- UI alinhada ao print-alvo (cards, tabs, matriz 2x2 com colorbar).
- PSI exibido em escala "0–100" quando o valor calculado estiver em 0–1 (ex.: 0.1901 -> 19.01).
- Dois K distintos (como no print):
  - Top-K (filtro-topk): usado para lista de alertas + KPIs rápidos (Precision@K/Recall@K).
  - K do Trade-off (k-selector): usado nos cards "Coerência — Desempenho OOT" + gráficos de trade-off.
- Mantém OOT (treino×validação) e modo Produção (baseline×produção) quando API_BASE_URL estiver configurada.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update
from datetime import datetime, timezone
import dash_bootstrap_components as dbc

from dashboard.student_detail import build_student_detail

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
MODELS_DIR = ROOT_DIR / "models"
_EVAL_SUBDIR = MODELS_DIR / "evaluation"
_MONITORING_SUBDIR = MODELS_DIR / "monitoring"
_BASELINES_SUBDIR = MODELS_DIR / "baselines"

SCORED_HISTORY_CSV = _EVAL_SUBDIR / "scored_history.csv"
SCORED_CSV = _EVAL_SUBDIR / "valid_scored.csv"
TRAIN_SCORED_CSV = _EVAL_SUBDIR / "train_scored.csv"
EVAL_JSON = _EVAL_SUBDIR / "evaluation_results.json"

DRIFT_JSONL = _MONITORING_SUBDIR / "drift_history.jsonl"
MONITORING_LOG = _MONITORING_SUBDIR / "monitoring.log"
RETRAIN_META = _MONITORING_SUBDIR / "retrain_metadata.json"

CURRENT_BASELINE_JSON = _BASELINES_SUBDIR / "current_baseline.json"

# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def _safe_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _safe_jsonl(path: Path, default: list[dict]):
    if not path.exists():
        return default
    rows: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    except Exception:
        return default
    return rows


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_scored() -> pd.DataFrame:
    df = _safe_csv(SCORED_HISTORY_CSV)
    if df.empty:
        df = _safe_csv(SCORED_CSV)
    if df.empty:
        return df
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "ano_base" in df.columns:
        df["ano_base"] = pd.to_numeric(df["ano_base"], errors="coerce").astype("Int64")
    return df


# --------------------------------------------------------------------------------------
# Business helpers
# --------------------------------------------------------------------------------------
def _phase_sort_key(value) -> int:
    text = str(value).upper().strip()
    if text in {"ALFA", "F0", "FASE0", "0"}:
        return 0
    if text.isdigit():
        return int(text)
    for token in text.replace("_", " ").split():
        if token.isdigit():
            return int(token)
        if token.startswith("FASE") and token[4:].isdigit():
            return int(token[4:])
    return 99


def _compute_topk_alerts(df: pd.DataFrame, k_pct: float) -> pd.DataFrame:
    out = df.copy()
    out["alerta"] = False
    if out.empty or "fase" not in out.columns or "score" not in out.columns:
        return out
    for _, grp in out.groupby("fase"):
        n = len(grp)
        k = max(1, int(np.ceil(n * k_pct / 100)))
        top_idx = grp["score"].nlargest(k).index
        out.loc[top_idx, "alerta"] = True
    return out


def _reason_text(row: pd.Series) -> str:
    rules = [
        ("matem", "Baixo desempenho em Matemática"),
        ("portug", "Baixo desempenho em Português"),
        ("ieg", "Engajamento reduzido (IEG)"),
        ("ips", "Sinal psicossocial de atenção (IPS)"),
    ]
    reasons: list[str] = []
    for col, msg in rules:
        if col in row.index and pd.notna(row[col]):
            try:
                if float(row[col]) < 6:
                    reasons.append(msg)
            except Exception:
                pass
    if not reasons:
        reasons = ["Padrão de risco global acima da referência da fase"]
    return ", ".join(reasons[:3])


def _compute_psi(baseline_scores: np.ndarray, current_scores: np.ndarray, n_bins: int = 10) -> float:
    """PSI entre distribuições de scores (0–1)."""
    if len(baseline_scores) < 10 or len(current_scores) < 10:
        return 0.0
    b = np.clip(baseline_scores, 0, 1)
    c = np.clip(current_scores, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    p_b, _ = np.histogram(b, bins=bins)
    p_c, _ = np.histogram(c, bins=bins)
    p_b = (p_b + 1e-6) / (p_b.sum() + 1e-6)
    p_c = (p_c + 1e-6) / (p_c.sum() + 1e-6)
    psi = np.sum((p_c - p_b) * np.log((p_c + 1e-10) / (p_b + 1e-10)))
    return float(max(0.0, psi))


def _view_auc(df: pd.DataFrame) -> float | None:
    if df.empty or not {"target", "score"}.issubset(df.columns):
        return None
    y = pd.to_numeric(df["target"], errors="coerce")
    s = pd.to_numeric(df["score"], errors="coerce") / 100.0
    mask = y.notna() & s.notna()
    if mask.sum() == 0:
        return None
    yv = y[mask].astype(int).values
    sv = s[mask].values
    if np.unique(yv).size < 2:
        return None
    order = np.argsort(sv)[::-1]
    y_sorted = yv[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))
    return abs(auc)


def _precision_at_topk(df: pd.DataFrame, k_pct: int) -> float | None:
    if df.empty or not {"target", "score"}.issubset(df.columns):
        return None
    work = df.copy()
    work["target"] = pd.to_numeric(work["target"], errors="coerce")
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work.dropna(subset=["target", "score"])
    if work.empty:
        return None
    k = max(1, int(np.ceil(len(work) * (k_pct / 100.0))))
    return float(work.nlargest(k, "score")["target"].mean())


def _recall_at_topk(df: pd.DataFrame, k_pct: int) -> float | None:
    if df.empty or not {"target", "score"}.issubset(df.columns):
        return None
    work = df.copy()
    work["target"] = pd.to_numeric(work["target"], errors="coerce")
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work.dropna(subset=["target", "score"])
    if work.empty:
        return None
    positives = float(work["target"].sum())
    if positives <= 0:
        return None
    k = max(1, int(np.ceil(len(work) * (k_pct / 100.0))))
    return float(work.nlargest(k, "score")["target"].sum() / positives)


def _lift_at_topk(df: pd.DataFrame, k_pct: int) -> float | None:
    if df.empty or not {"target", "score"}.issubset(df.columns):
        return None
    work = df.copy()
    work["target"] = pd.to_numeric(work["target"], errors="coerce")
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work.dropna(subset=["target", "score"])
    if work.empty:
        return None
    base_rate = float(work["target"].mean())
    if base_rate <= 0:
        return None
    p_at_k = _precision_at_topk(work, k_pct)
    return float(p_at_k / base_rate) if p_at_k is not None else None


# --------------------------------------------------------------------------------------
# Monitoring helpers (API or local)
# --------------------------------------------------------------------------------------
def _fetch_drift_api(api_url: str, mode: str, window: str) -> dict:
    if not api_url:
        return {}
    try:
        qs = urllib.parse.urlencode({"mode": mode, "window": window})
        req = urllib.request.Request(f"{api_url.rstrip('/')}/metrics/drift?{qs}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode())
    except Exception:
        pass
    return {}


def _fetch_drift_history(api_url: str, mode: str, window: str, limit: int = 200) -> list:
    if not api_url:
        return []
    try:
        qs = urllib.parse.urlencode({"mode": mode, "window": window, "limit": limit})
        req = urllib.request.Request(f"{api_url.rstrip('/')}/metrics/drift/history?{qs}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                if isinstance(data, dict):
                    return data.get("history", data.get("events", []))
                return []
    except Exception:
        pass
    return []


def _fetch_logs_api(api_url: str, lines: int = 80) -> dict | None:
    """Chama GET /metrics/logs na API. Retorna dict com 'content' e 'kpis', ou None se falhar."""
    if not api_url:
        return None
    try:
        qs = urllib.parse.urlencode({"lines": lines})
        req = urllib.request.Request(f"{api_url.rstrip('/')}/metrics/logs?{qs}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode())
    except Exception:
        pass
    return None


def _get_drift_history(mode: str, window: str, limit: int = 200) -> list:
    api_url = os.environ.get("API_BASE_URL", "").rstrip("/")
    hist = _fetch_drift_history(api_url, mode, window, limit)
    if hist:
        return hist
    events = _safe_jsonl(DRIFT_JSONL, [])
    if mode == "oot":
        return [e for e in events if str(e.get("event_type")) == "oot_snapshot"]
    if mode == "prod":
        return [e for e in events if str(e.get("event_type")) == "predict_batch"]
    return events


def _read_monitoring_log(lines: int = 80) -> str:
    api_url = os.environ.get("API_BASE_URL", "").rstrip("/")
    if api_url:
        data = _fetch_logs_api(api_url, lines)
        if data is not None:
            return data.get("content", "").strip() or "(arquivo vazio)"
    if not MONITORING_LOG.exists():
        return "Arquivo monitoring.log não encontrado. Inicie a API para gerar logs."
    try:
        all_lines = MONITORING_LOG.read_text(encoding="utf-8", errors="replace").splitlines(True)
        last = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return "".join(reversed(last)).strip() or "(arquivo vazio)"
    except Exception as e:
        return f"Erro ao ler log: {e}"


def _parse_monitoring_log_local() -> tuple[int, int, float]:
    """Retorna (n_requests, n_errors, p95_ms). Tenta API primeiro, depois arquivo local."""
    api_url = os.environ.get("API_BASE_URL", "").rstrip("/")
    if api_url:
        data = _fetch_logs_api(api_url, lines=80)
        if data is not None and "kpis" in data:
            kpis = data["kpis"]
            return (
                int(kpis.get("n_requests", 0)),
                int(kpis.get("n_errors", 0)),
                float(kpis.get("p95_latency_ms", 0.0)),
            )
    n_requests = 0
    n_errors = 0
    latencies: list[float] = []
    if not MONITORING_LOG.exists():
        return 0, 0, 0.0
    try:
        text = MONITORING_LOG.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if "event_type" not in line:
                continue
            if "event_type=error" in line:
                n_errors += 1
            elif "event_type=predict_batch" in line and "Predict batch complete" in line:
                n_requests += 1
                m = re.search(r"latency_ms=(\d+)", line)
                if m:
                    latencies.append(float(m.group(1)))
    except Exception:
        pass
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    return n_requests, n_errors, p95


# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------
def kpi_card(title: str, value_id: str, subtitle_id: str | None = None, badge_id: str | None = None):
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Div(title, className="text-muted")),
                        dbc.Col(html.Div(id=badge_id), width="auto") if badge_id else dbc.Col(width="auto"),
                    ],
                    align="center",
                    className="mb-2",
                ),
                html.H3(id=value_id, className="mb-0"),
                html.Small(id=subtitle_id, className="text-muted") if subtitle_id else None,
            ]
        ),
        className="shadow-sm h-100",
    )


def _normalize_status(status: str) -> str:
    """Padroniza status para apenas 'ok' ou 'critical' (look and feel do print)."""
    s = (status or "").lower().strip()
    if s in ("critical", "high", "danger", "warning", "medium", "warn"):
        return "critical"
    return "ok"


def _badge_component(status: str):
    """Badge padrão: 'ok' verde / 'critical' vermelho."""
    s = _normalize_status(status)
    if s == "critical":
        return dbc.Badge("critical", color="danger", className="ms-1")
    return dbc.Badge("ok", color="success", className="ms-1")


def _status_psi(v: float | None) -> str:
    if v is None:
        return "ok"
    if v < 0.10:
        return "ok"
    if v < 0.20:
        return "warning"
    return "critical"


def _status_auc(v: float | None) -> str:
    if v is None:
        return "warning"
    if v >= 0.70:
        return "ok"
    if v >= 0.65:
        return "warning"
    return "critical"


def _status_precision(v: float | None) -> str:
    if v is None:
        return "warning"
    if v >= 0.75:
        return "ok"
    if v >= 0.65:
        return "warning"
    return "critical"


def _status_recall(v: float | None) -> str:
    if v is None:
        return "warning"
    if v >= 0.35:
        return "ok"
    if v >= 0.25:
        return "warning"
    return "critical"


def _status_lift(v: float | None) -> str:
    if v is None:
        return "warning"
    if v >= 1.8:
        return "ok"
    if v >= 1.3:
        return "warning"
    return "critical"


def _psi_display(v: float | None) -> str:
    """No print-alvo o PSI aparece como ~19.01 (i.e., 0.1901 * 100).
    Regra: se 0 <= v <= 1.0, exibe em %*100; senão, exibe bruto.
    """
    if v is None:
        return "—"
    if 0 <= v <= 1.0:
        return f"{(v*100):.4f}"
    return f"{v:.4f}"


def _filtered_df(df: pd.DataFrame, ano, fases, turmas, ra_query, topk) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if ano is not None and "ano_base" in out.columns:
        out = out[out["ano_base"] == int(ano)]
    if fases:
        out = out[out["fase"].astype(str).isin([str(f) for f in fases])]
    if turmas:
        out = out[out["turma"].astype(str).isin([str(t) for t in turmas])]
    if ra_query:
        needle = str(ra_query).strip().lower()
        if needle and "ra" in out.columns:
            out = out[out["ra"].astype(str).str.lower().str.contains(needle, na=False)]
    out = _compute_topk_alerts(out, topk)
    out["motivos"] = out.apply(_reason_text, axis=1)
    return out


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------
def _build_main_layout():
    df = _load_scored()
    years = sorted(pd.to_numeric(df["ano_base"], errors="coerce").dropna().astype(int).unique().tolist()) if "ano_base" in df.columns else []
    phases = sorted(df["fase"].dropna().astype(str).unique().tolist(), key=_phase_sort_key) if "fase" in df.columns else []
    turmas = sorted(df["turma"].dropna().astype(str).unique().tolist()) if "turma" in df.columns else []

    return dbc.Container(
        fluid=True,
        children=[
            dbc.Card(
                className="shadow-sm mb-3",
                children=[
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4("Monitoramento do Modelo — Drift × Coerência", className="mb-0"),
                                            html.Div(id="ui-subtitle", className="text-muted"),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Badge(id="ui-context-badge", color="primary", className="me-2"),
                                            dbc.Badge(id="ui-model-badge", color="secondary"),
                                        ],
                                        width="auto",
                                        className="text-end",
                                    ),
                                ],
                                align="center",
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div("Contexto", className="text-muted small"),
                                            dbc.RadioItems(
                                                id="context-mode",
                                                options=[
                                                    {"label": "🔍 OOT (Treino×Validação)", "value": "oot"},
                                                    {"label": "📡 Produção (Baseline×Produção)", "value": "prod"},
                                                ],
                                                value="oot",
                                                inline=True,
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div("Filtros", className="text-muted small"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Small("Ano-base"),
                                                            dcc.Dropdown(
                                                                id="filtro-ano",
                                                                options=[{"label": str(y), "value": y} for y in years],
                                                                value=max(years) if years else None,
                                                                clearable=True,
                                                            ),
                                                        ],
                                                        md=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Small("Fase(s)"),
                                                            dcc.Dropdown(
                                                                id="filtro-fase",
                                                                options=[{"label": p, "value": p} for p in phases],
                                                                value=phases,
                                                                multi=True,
                                                            ),
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Small("Turma(s)"),
                                                            dcc.Dropdown(
                                                                id="filtro-turma",
                                                                options=[{"label": t, "value": t} for t in turmas],
                                                                value=[],
                                                                multi=True,
                                                            ),
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Small("Top-K%"),
                                                            dcc.Dropdown(
                                                                id="filtro-topk",
                                                                options=[{"label": f"{k}%", "value": k} for k in [10, 15, 20, 25]],
                                                                value=15,
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        md=2,
                                                    ),
                                                    dbc.Col(
                                                        id="filtro-janela-wrapper",
                                                        children=[
                                                            html.Small("Janela prod"),
                                                            dcc.Dropdown(
                                                                id="filtro-janela",
                                                                options=[{"label": w, "value": w} for w in ["7d", "14d", "30d", "60d", "90d"]],
                                                                value="30d",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        md=2,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ]
                            ),
                        ]
                    )
                ],
            ),
            dbc.Tabs(
                id="main-tabs",
                active_tab="tab-overview",
                children=[
                    dbc.Tab(
                        label="Visão Geral (Diagnóstico)",
                        tab_id="tab-overview",
                        children=[
                            html.Div(className="mt-3"),
                            dbc.Card(
                                className="shadow-sm mb-3",
                                children=[
                                    dbc.CardBody(
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div("Diagnóstico automático", className="text-muted"),
                                                        html.H4(id="diag-status-title", className="mb-1"),
                                                        html.Div(id="diag-status-text", className="text-muted"),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div("Regra", className="text-muted"),
                                                        html.Div(id="diag-rule-text", className="small"),
                                                    ],
                                                    md=4,
                                                ),
                                            ],
                                            align="center",
                                        )
                                    )
                                ],
                            ),
                            dbc.Row(
                                className="g-3 mb-3",
                                children=[
                                    dbc.Col(
                                        dbc.Card(
                                            className="shadow-sm h-100",
                                            children=[
                                                dbc.CardBody(
                                                    [
                                                        html.Div("Matriz Drift × Coerência", className="text-muted"),
                                                        dcc.Graph(id="drift-coherence-matrix", config={"displayModeBar": False}),
                                                        html.Div(id="matrix-caption", className="text-muted small mt-2"),
                                                    ]
                                                )
                                            ],
                                        ),
                                        md=5,
                                    ),
                                    dbc.Col(
                                        dbc.Row(
                                            className="g-3",
                                            children=[
                                                dbc.Col(kpi_card("PSI (global)", "kpi-psi", "kpi-psi-sub", "kpi-psi-badge"), md=6),
                                                dbc.Col(kpi_card("AUC (OOT)", "kpi-auc", "kpi-auc-sub", "kpi-auc-badge"), md=6),
                                                dbc.Col(kpi_card("Precision@K", "kpi-prec", "kpi-prec-sub", "kpi-prec-badge"), md=6),
                                                dbc.Col(kpi_card("Recall@K", "kpi-rec", "kpi-rec-sub", "kpi-rec-badge"), md=6),
                                            ],
                                        ),
                                        md=7,
                                    ),
                                ],
                            ),
                            dbc.Row(
                                className="g-3",
                                children=[
                                    dbc.Col(
                                        children=[
                                            dbc.Card(
                                                className="shadow-sm mb-3",
                                                children=[
                                                    dbc.CardBody(
                                                        [
                                                            html.Div("Drift — Estabilidade dos dados/scores", className="text-muted"),
                                                            dcc.Graph(id="psi-trend", config={"displayModeBar": False}),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(html.Div(id="psi-meta", className="text-muted small")),
                                                                    dbc.Col(html.Div(id="psi-window-meta", className="text-muted small"), className="text-end"),
                                                                ]
                                                            ),
                                                        ]
                                                    )
                                                ],
                                            ),
                                            dbc.Card(
                                                className="shadow-sm mb-3",
                                                children=[
                                                    dbc.CardBody(
                                                        [
                                                            html.Div("Onde está o drift?", className="text-muted"),
                                                            dcc.Graph(id="drift-top-contributors", config={"displayModeBar": False}),
                                                            html.Hr(),
                                                            html.Div("Detalhe (Top contribuições)", className="text-muted small mb-2"),
                                                            dash_table.DataTable(
                                                                id="drift-contrib-table",
                                                                columns=[
                                                                    {"name": "bin/feature", "id": "name"},
                                                                    {"name": "baseline_pct", "id": "baseline_pct"},
                                                                    {"name": "current_pct", "id": "current_pct"},
                                                                    {"name": "contrib", "id": "contrib"},
                                                                ],
                                                                data=[],
                                                                page_size=8,
                                                                style_table={"overflowX": "auto"},
                                                                style_cell={"fontFamily": "system-ui", "fontSize": 12, "padding": "6px"},
                                                                style_header={"fontWeight": "600"},
                                                            ),
                                                        ]
                                                    )
                                                ],
                                            ),
                                            dbc.Card(
                                                className="shadow-sm",
                                                children=[
                                                    dbc.CardBody(
                                                        [
                                                            html.Div("Qualidade de dados (resumo)", className="text-muted"),
                                                            dbc.Row(
                                                                className="g-3",
                                                                children=[
                                                                    dbc.Col(kpi_card("% Missing (Top)", "kpi-missing", "kpi-missing-sub"), md=4),
                                                                    dbc.Col(kpi_card("Categorias novas", "kpi-newcats", "kpi-newcats-sub"), md=4),
                                                                    dbc.Col(kpi_card("Outliers", "kpi-outliers", "kpi-outliers-sub"), md=4),
                                                                ],
                                                            ),
                                                            html.Div(id="dq-notes", className="text-muted small mt-2"),
                                                        ]
                                                    )
                                                ],
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        children=[
                                            dbc.Card(
                                                className="shadow-sm mb-3",
                                                children=[
                                                    dbc.CardBody(
                                                        [
                                                            html.Div("Coerência — Desempenho OOT", className="text-muted"),
                                                            dbc.Row(
                                                                className="g-3",
                                                                children=[
                                                                    dbc.Col(kpi_card("AUC OOT", "kpi-auc2", "kpi-auc2-sub", "kpi-auc2-badge"), md=3),
                                                                    dbc.Col(kpi_card("Precision@K", "kpi-prec2", "kpi-prec2-sub", "kpi-prec2-badge"), md=3),
                                                                    dbc.Col(kpi_card("Recall@K", "kpi-rec2", "kpi-rec2-sub", "kpi-rec2-badge"), md=3),
                                                                    dbc.Col(kpi_card("Lift@K", "kpi-lift2", "kpi-lift2-sub", "kpi-lift2-badge"), md=3),
                                                                ],
                                                            ),
                                                            html.Div(id="perf-note", className="text-muted small mt-2"),
                                                        ]
                                                    )
                                                ],
                                            ),
                                            dbc.Card(
                                                className="shadow-sm mb-3",
                                                children=[
                                                    dbc.CardBody(
                                                        [
                                                            html.Div("Trade-off por K (Prec vs Recall)", className="text-muted"),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div("K% (alerta)", className="text-muted small"),
                                                                            dcc.Dropdown(
                                                                                id="k-selector",
                                                                                options=[{"label": f"{k}%", "value": k} for k in [10, 15, 20, 25]],
                                                                                value=15,
                                                                                clearable=False,
                                                                            ),
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div("Sugestão automática", className="text-muted small"),
                                                                            html.Div(id="k-suggestion", className="small"),
                                                                        ],
                                                                        md=8,
                                                                    ),
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                            dcc.Graph(id="k-tradeoff-graph", config={"displayModeBar": False}),
                                                        ]
                                                    )
                                                ],
                                            ),
                                            dbc.Card(
                                                className="shadow-sm",
                                                children=[
                                                    dbc.CardBody(
                                                        [
                                                            html.Div("Calibração (OOT)", className="text-muted"),
                                                            dcc.Graph(id="calibration-curve", config={"displayModeBar": False}),
                                                            html.Hr(),
                                                            html.Div("Tabela por faixas de score", className="text-muted small mb-2"),
                                                            dash_table.DataTable(
                                                                id="calibration-table",
                                                                columns=[
                                                                    {"name": "faixa_score", "id": "bucket"},
                                                                    {"name": "N", "id": "n"},
                                                                    {"name": "target_rate", "id": "rate"},
                                                                ],
                                                                data=[],
                                                                page_size=8,
                                                                style_table={"overflowX": "auto"},
                                                                style_cell={"fontFamily": "system-ui", "fontSize": 12, "padding": "6px"},
                                                                style_header={"fontWeight": "600"},
                                                            ),
                                                        ]
                                                    )
                                                ],
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                            ),
                            dbc.Card(
                                className="shadow-sm mt-3 mb-4",
                                children=[
                                    dbc.CardBody(
                                        [
                                            html.Div("Ações recomendadas", className="text-muted"),
                                            html.Ul(id="action-list", className="mb-2"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Button("Exportar snapshot", id="btn-export-snapshot", color="secondary"), width="auto"),
                                                    dbc.Col(dbc.Button("Gerar relatório", id="btn-generate-report", color="primary"), width="auto"),
                                                    dbc.Col(html.Div(id="action-footer", className="text-muted small"), className="text-end"),
                                                ],
                                                justify="between",
                                                align="center",
                                            ),
                                        ]
                                    )
                                ],
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Alertas",
                        tab_id="tab-alerts",
                        children=[
                            html.Div(className="mt-3"),
                            dbc.Card(
                                className="shadow-sm",
                                children=[
                                    dbc.CardBody(
                                        [
                                            html.Div("Lista de alunos em alerta (Top-K)", className="text-muted"),
                                            html.Div(id="alerts-filters"),
                                            dash_table.DataTable(
                                                id="alerts-table",
                                                columns=[
                                                    {"name": "RA", "id": "ra"},
                                                    {"name": "Fase", "id": "fase"},
                                                    {"name": "Turma", "id": "turma"},
                                                    {"name": "Score", "id": "score"},
                                                    {"name": "Motivos (top3)", "id": "motivos"},
                                                    {"name": "Detalhe", "id": "link", "presentation": "markdown"},
                                                ],
                                                data=[],
                                                page_size=12,
                                                markdown_options={"link_target": "_self"},
                                                style_cell={"padding": "6px"},
                                                style_header={"fontWeight": "600"},
                                            ),
                                            html.Div(id="alerts-note", className="text-muted small mt-2"),
                                        ]
                                    )
                                ],
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Distribuição por Fase",
                        tab_id="tab-phase",
                        children=[
                            html.Div(className="mt-3"),
                            dbc.Row(
                                className="g-3",
                                children=[
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.Div("Alertas por fase", className="text-muted"),
                                                    dcc.Graph(id="alerts-by-phase-2", config={"displayModeBar": False}),
                                                ]
                                            ),
                                            className="shadow-sm",
                                        ),
                                        md=6,
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.Div("Score médio por fase", className="text-muted"),
                                                    dcc.Graph(id="score-by-phase-2", config={"displayModeBar": False}),
                                                ]
                                            ),
                                            className="shadow-sm",
                                        ),
                                        md=6,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Monitoramento (Logs)",
                        tab_id="tab-logs",
                        children=[
                            html.Div(className="mt-3"),
                            dcc.Interval(id="logs-refresh", interval=5000, n_intervals=0),
                            dbc.Row(
                                className="g-3",
                                children=[
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.Div("Logging de Monitoramento", className="text-muted mb-2"),
                                                    html.Div(
                                                        id="monitoring-log-content",
                                                        style={
                                                            "fontFamily": "monospace",
                                                            "fontSize": 12,
                                                            "backgroundColor": "#1e1e1e",
                                                            "color": "#4ec9b0",
                                                            "padding": "12px",
                                                            "borderRadius": "6px",
                                                            "maxHeight": "400px",
                                                            "overflowY": "auto",
                                                            "whiteSpace": "pre-wrap",
                                                        },
                                                        children="Carregando logs...",
                                                    ),
                                                ]
                                            ),
                                            className="shadow-sm",
                                        ),
                                        md=12,
                                    ),
                                    dbc.Col(
                                        dbc.Card(dbc.CardBody([html.Div("Tráfego e erros", className="text-muted"), dcc.Graph(id="requests-errors", config={"displayModeBar": False})]), className="shadow-sm"),
                                        md=6,
                                    ),
                                    dbc.Col(
                                        dbc.Card(dbc.CardBody([html.Div("Latência P95", className="text-muted"), dcc.Graph(id="latency-p95", config={"displayModeBar": False})]), className="shadow-sm"),
                                        md=6,
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.Div("Eventos (drift_history)", className="text-muted"),
                                                    dash_table.DataTable(
                                                        id="events-table",
                                                        columns=[
                                                            {"name": "timestamp", "id": "ts"},
                                                            {"name": "event_type", "id": "event"},
                                                            {"name": "model_version", "id": "model_version"},
                                                            {"name": "baseline_id", "id": "baseline_id"},
                                                            {"name": "n_students", "id": "n_students"},
                                                            {"name": "psi", "id": "psi"},
                                                            {"name": "severity", "id": "severity"},
                                                        ],
                                                        data=[],
                                                        page_size=12,
                                                        style_cell={"padding": "6px"},
                                                        style_header={"fontWeight": "600"},
                                                    ),
                                                ]
                                            ),
                                            className="shadow-sm",
                                        ),
                                        md=12,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Dados & Retreinamento",
                        tab_id="tab-retrain",
                        children=[
                            html.Div(className="mt-3"),
                            dbc.Card(
                                className="shadow-sm mb-3",
                                children=[
                                    dbc.CardBody(
                                        [
                                            html.H5("Baselines disponíveis", className="mb-2"),
                                            dash_table.DataTable(
                                                id="baselines-table",
                                                columns=[
                                                    {"name": "baseline_id", "id": "baseline_id"},
                                                    {"name": "n_students", "id": "n_students"},
                                                    {"name": "created_at", "id": "created_at"},
                                                ],
                                                data=[],
                                                page_size=10,
                                                style_cell={"padding": "6px"},
                                                style_header={"fontWeight": "600"},
                                            ),
                                        ]
                                    )
                                ],
                            ),
                            dbc.Card(
                                className="shadow-sm",
                                children=[dbc.CardBody([html.H5("Checklist de retreino", className="mb-2"), html.Div(id="retrain-checklist", className="small")])],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="store-context"),
            dcc.Store(id="store-data-snap"),
        ],
    )


def build_layout():
    return dbc.Container(
        fluid=True,
        children=[
            dcc.Location(id="url", refresh=False),
            html.Div(id="page-content", children=_build_main_layout()),
            dcc.Download(id="download-report"),
            dcc.Download(id="download-snapshot"),
        ],
    )


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Monitoramento Drift × Coerência - Passos Mágicos"
app.layout = build_layout()
app.server.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# CSS leve para aproximar do print (sem mexer no Bootstrap)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
          .nav-tabs .nav-link { color:#2563eb; }
          .nav-tabs .nav-link.active { color:#111827; font-weight:600; }
          .card { border-color: rgba(17,24,39,.12); }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


@app.server.errorhandler(404)
def _serve_app(_e):
    return app.index()


# --------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname and pathname.startswith("/aluno/"):
        ra_raw = pathname.split("/aluno/", 1)[1].strip()
        ra = ra_raw.replace("%20", " ").strip()
        df = _load_scored()
        if df.empty:
            return dbc.Alert("Dados não encontrados. Execute train.py.", color="warning")
        dff = _compute_topk_alerts(df.copy(), 15)
        dff["motivos"] = dff.apply(_reason_text, axis=1)
        eval_data = _safe_json(EVAL_JSON, {})
        return build_student_detail(ra, dff, eval_data)
    return _build_main_layout()


# --------------------------------------------------------------------------------------
# Topbar
# --------------------------------------------------------------------------------------
@app.callback(
    Output("ui-subtitle", "children"),
    Output("ui-context-badge", "children"),
    Output("ui-model-badge", "children"),
    Output("filtro-ano", "options"),
    Output("filtro-ano", "value"),
    Output("filtro-fase", "options"),
    Output("filtro-fase", "value"),
    Output("filtro-turma", "options"),
    Output("filtro-turma", "value"),
    Input("context-mode", "value"),
)
def update_topbar(context):
    df = _load_scored()
    years = sorted(pd.to_numeric(df.get("ano_base", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist()) if not df.empty else []
    phases = sorted(df.get("fase", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(), key=_phase_sort_key) if not df.empty else []
    turmas = sorted(df.get("turma", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) if not df.empty else []
    retrain = _safe_json(RETRAIN_META, {})
    model_ver = retrain.get("model_version", "1.0.0")
    auc = retrain.get("auc", "—")
    ctx_label = "OOT (Treino×Validação)" if context == "oot" else "Produção (Baseline×Produção)"
    return (
        "Passos Mágicos — Datathon",
        ctx_label,
        f"v{model_ver} | AUC={auc}",
        [{"label": str(y), "value": y} for y in years],
        max(years) if years else None,
        [{"label": p, "value": p} for p in phases],
        phases,
        [{"label": t, "value": t} for t in turmas],
        [],
    )


# --------------------------------------------------------------------------------------
# Alertas
# --------------------------------------------------------------------------------------
@app.callback(
    Output("alerts-filters", "children"),
    Output("alerts-table", "data"),
    Output("alerts-note", "children"),
    Input("filtro-ano", "value"),
    Input("filtro-fase", "value"),
    Input("filtro-turma", "value"),
    Input("filtro-topk", "value"),
)
def update_alerts(ano, fases, turmas, topk):
    topk = int(topk or 15)
    df = _load_scored()
    dff = _filtered_df(df, ano, fases or [], turmas or [], None, topk)
    if dff.empty:
        return html.Div("Sem dados carregados.", className="small text-muted"), [], "0 alunos em alerta."
    alert_df = dff[dff["alerta"]].sort_values("score", ascending=False).copy()
    alert_df["link"] = alert_df["ra"].apply(lambda x: f"[Abrir](/aluno/{str(x).strip()})")
    cols = ["ra", "fase", "turma", "score", "motivos", "link"]
    data = alert_df[[c for c in cols if c in alert_df.columns]].to_dict("records") if not alert_df.empty else []
    return (
        html.Div(f"Filtros: Ano={ano or 'todos'}, Fase={fases or 'todas'}, K={topk}%", className="small text-muted"),
        data,
        f"{len(alert_df)} alunos em alerta.",
    )


# --------------------------------------------------------------------------------------
# Distribuição por fase
# --------------------------------------------------------------------------------------
@app.callback(
    Output("alerts-by-phase-2", "figure"),
    Output("score-by-phase-2", "figure"),
    Input("filtro-ano", "value"),
    Input("filtro-fase", "value"),
    Input("filtro-turma", "value"),
    Input("filtro-topk", "value"),
)
def update_phase(ano, fases, turmas, topk):
    topk = int(topk or 15)
    df = _load_scored()
    dff = _filtered_df(df, ano, fases or [], turmas or [], None, topk)
    if dff.empty:
        fig = go.Figure().update_layout(height=280)
        return fig, fig
    by_phase = dff.groupby("fase", as_index=False).agg(total=("ra", "count"), alertas=("alerta", "sum"), score_medio=("score", "mean"))
    by_phase["pct_alerta"] = np.where(by_phase["total"] > 0, 100 * by_phase["alertas"] / by_phase["total"], 0)
    fig1 = px.bar(by_phase, x="fase", y="pct_alerta", title="% em alerta por fase").update_layout(height=280, margin=dict(t=45, b=10, l=10, r=10))
    fig2 = px.bar(by_phase, x="fase", y="score_medio", title="Score médio por fase").update_layout(height=280, margin=dict(t=45, b=10, l=10, r=10))
    return fig1, fig2


# --------------------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------------------
@app.callback(Output("monitoring-log-content", "children"), Input("logs-refresh", "n_intervals"), Input("main-tabs", "active_tab"))
def update_monitoring_log(_n, active_tab):
    if active_tab != "tab-logs":
        return no_update
    return _read_monitoring_log(80)


@app.callback(
    Output("requests-errors", "figure"),
    Output("latency-p95", "figure"),
    Output("events-table", "data"),
    Input("context-mode", "value"),
    Input("filtro-janela", "value"),
    Input("logs-refresh", "n_intervals"),
)
def update_logs(context, janela, _refresh):
    n_requests, n_errors, p95_ms = _parse_monitoring_log_local()

    fig_traffic = go.Figure(data=[go.Bar(x=["Requests", "Erros"], y=[n_requests, n_errors])])
    fig_traffic.update_layout(title=f"Requests: {n_requests} | Erros: {n_errors}", height=200, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))

    fig_latency = go.Figure(go.Indicator(mode="number", value=p95_ms, number={"suffix": " ms", "font": {"size": 48}}, title={"text": "P95 Latência"}))
    fig_latency.update_layout(height=200, margin=dict(t=25, b=10, l=10, r=10))

    drift_mode = "oot" if context == "oot" else "prod"
    window = (janela or "30d") if context == "prod" else "30d"
    events = _get_drift_history(drift_mode, window)

    rows = []
    for e in events[-30:]:
        psi_val = e.get("psi")
        if psi_val is None or (isinstance(psi_val, str) and "low_sample" in str(psi_val).lower()):
            psi_str = "low_sample"
        else:
            try:
                psi_str = round(float(psi_val), 4)
            except Exception:
                psi_str = "—"
        rows.append(
            {
                "ts": (e.get("timestamp_utc", "") or "")[:19],
                "event": e.get("event_type", ""),
                "model_version": e.get("model_version", ""),
                "baseline_id": e.get("baseline_id", ""),
                "n_students": e.get("n_students", ""),
                "psi": psi_str,
                "severity": e.get("severity", ""),
            }
        )
    return fig_traffic, fig_latency, rows


@app.callback(Output("filtro-janela-wrapper", "style"), Input("context-mode", "value"))
def toggle_janela(context):
    # Manter componente sempre visível no DOM (evita IndexError em callbacks)
    return {"display": "block"} if context == "prod" else {"display": "block", "opacity": 0.5}


# --------------------------------------------------------------------------------------
# Botões Gerar relatório e Exportar snapshot
# --------------------------------------------------------------------------------------
@app.callback(
    Output("download-report", "data"),
    Input("btn-generate-report", "n_clicks"),
    State("filtro-ano", "value"),
    State("filtro-fase", "value"),
    State("filtro-turma", "value"),
    State("filtro-topk", "value"),
    State("k-selector", "value"),
    State("context-mode", "value"),
    State("filtro-janela", "value"),
    prevent_initial_call=True,
)
def generate_report(n_clicks, ano, fases, turmas, topk_alert, k_tradeoff, context, janela):
    if not n_clicks:
        return no_update
    topk_alert = int(topk_alert or 15)
    k_tradeoff = int(k_tradeoff or 15)
    janela = (janela or "30d").strip()
    df = _load_scored()
    dff = _filtered_df(df, ano, fases or [], turmas or [], None, topk_alert)

    psi_raw = None
    api_url = os.environ.get("API_BASE_URL", "").rstrip("/")
    if api_url and context == "prod":
        drift_sum = _fetch_drift_api(api_url, "prod", janela)
        if drift_sum.get("psi") is not None:
            try:
                psi_raw = float(drift_sum.get("psi"))
            except Exception:
                pass
    else:
        train_df = _safe_csv(TRAIN_SCORED_CSV)
        if train_df.empty:
            train_df = df.head(min(600, len(df)))
        valid_df = dff if not dff.empty else df.head(min(765, len(df)))
        if not train_df.empty and not valid_df.empty and "score" in train_df.columns and "score" in valid_df.columns:
            psi_raw = _compute_psi(
                pd.to_numeric(train_df["score"], errors="coerce").dropna().values / 100.0,
                pd.to_numeric(valid_df["score"], errors="coerce").dropna().values / 100.0,
                n_bins=10,
            )

    auc_val = _view_auc(dff)
    prec = _precision_at_topk(dff, topk_alert)
    rec = _recall_at_topk(dff, topk_alert)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filtros": {"ano": ano, "fases": fases, "turmas": turmas, "topk_alert": topk_alert, "k_tradeoff": k_tradeoff, "context": context, "janela": janela},
        "metricas": {
            "psi": float(psi_raw) if psi_raw is not None else None,
            "auc": float(auc_val) if auc_val is not None else None,
            "precision_at_k": float(prec) if prec is not None else None,
            "recall_at_k": float(rec) if rec is not None else None,
        },
        "resumo": {"n_registros": len(dff), "n_alertas": int(dff["alerta"].sum()) if "alerta" in dff.columns else 0},
    }
    content = json.dumps(report, indent=2, ensure_ascii=False)
    return dict(content=content, filename=f"relatorio_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", type="application/json")


@app.callback(
    Output("download-snapshot", "data"),
    Input("btn-export-snapshot", "n_clicks"),
    State("filtro-ano", "value"),
    State("filtro-fase", "value"),
    State("filtro-turma", "value"),
    State("filtro-topk", "value"),
    prevent_initial_call=True,
)
def export_snapshot(n_clicks, ano, fases, turmas, topk_alert):
    if not n_clicks:
        return no_update
    topk_alert = int(topk_alert or 15)
    df = _load_scored()
    dff = _filtered_df(df, ano, fases or [], turmas or [], None, topk_alert)
    if dff.empty:
        return no_update
    return dcc.send_data_frame(dff.to_csv, f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)


# --------------------------------------------------------------------------------------
# Overview / Diagnóstico
# --------------------------------------------------------------------------------------
@app.callback(
    Output("kpi-psi", "children"),
    Output("kpi-psi-sub", "children"),
    Output("kpi-psi-badge", "children"),
    Output("kpi-auc", "children"),
    Output("kpi-auc-sub", "children"),
    Output("kpi-auc-badge", "children"),
    Output("kpi-prec", "children"),
    Output("kpi-prec-sub", "children"),
    Output("kpi-prec-badge", "children"),
    Output("kpi-rec", "children"),
    Output("kpi-rec-sub", "children"),
    Output("kpi-rec-badge", "children"),
    Output("drift-coherence-matrix", "figure"),
    Output("matrix-caption", "children"),
    Output("psi-trend", "figure"),
    Output("psi-meta", "children"),
    Output("psi-window-meta", "children"),
    Output("drift-top-contributors", "figure"),
    Output("drift-contrib-table", "data"),
    Output("kpi-missing", "children"),
    Output("kpi-missing-sub", "children"),
    Output("kpi-newcats", "children"),
    Output("kpi-newcats-sub", "children"),
    Output("kpi-outliers", "children"),
    Output("kpi-outliers-sub", "children"),
    Output("dq-notes", "children"),
    Output("kpi-auc2", "children"),
    Output("kpi-auc2-sub", "children"),
    Output("kpi-auc2-badge", "children"),
    Output("kpi-prec2", "children"),
    Output("kpi-prec2-sub", "children"),
    Output("kpi-prec2-badge", "children"),
    Output("kpi-rec2", "children"),
    Output("kpi-rec2-sub", "children"),
    Output("kpi-rec2-badge", "children"),
    Output("kpi-lift2", "children"),
    Output("kpi-lift2-sub", "children"),
    Output("kpi-lift2-badge", "children"),
    Output("perf-note", "children"),
    Output("k-suggestion", "children"),
    Output("k-tradeoff-graph", "figure"),
    Output("calibration-curve", "figure"),
    Output("calibration-table", "data"),
    Output("action-list", "children"),
    Output("action-footer", "children"),
    Input("filtro-ano", "value"),
    Input("filtro-fase", "value"),
    Input("filtro-turma", "value"),
    Input("filtro-topk", "value"),      # K do alerta (print: 25% no card de cima)
    Input("k-selector", "value"),       # K do trade-off (print: 15% no box de coerência)
    Input("context-mode", "value"),
    Input("filtro-janela", "value"),
    Input("main-tabs", "active_tab"),   # Reexecuta ao exibir aba Visão Geral
)
def update_overview(ano, fases, turmas, topk_alert, k_tradeoff, context, janela, active_tab):
    if active_tab != "tab-overview":
        return [no_update] * 45

    topk_alert = int(topk_alert or 15)
    k_tradeoff = int(k_tradeoff or 15)
    janela = (janela or "30d").strip()

    df = _load_scored()
    dff = _filtered_df(df, ano, fases or [], turmas or [], None, topk_alert)

    # -----------------------
    # 1) PSI (OOT ou PROD)
    # -----------------------
    psi_raw: float | None = None
    api_url = os.environ.get("API_BASE_URL", "").rstrip("/")

    if api_url and context == "prod":
        drift_sum = _fetch_drift_api(api_url, "prod", janela)
        if drift_sum.get("psi") is not None:
            try:
                psi_raw = float(drift_sum.get("psi"))
            except Exception:
                psi_raw = None
        psi_status = drift_sum.get("severity", _status_psi(psi_raw))
    else:
        train_df = _safe_csv(TRAIN_SCORED_CSV)
        if train_df.empty:
            train_df = df.head(min(600, len(df)))
        valid_df = dff if not dff.empty else df.head(min(765, len(df)))

        if (not train_df.empty) and (not valid_df.empty) and ("score" in train_df.columns) and ("score" in valid_df.columns):
            psi_raw = _compute_psi(
                pd.to_numeric(train_df["score"], errors="coerce").dropna().values / 100.0,
                pd.to_numeric(valid_df["score"], errors="coerce").dropna().values / 100.0,
                n_bins=10,
            )
        psi_status = _status_psi(psi_raw)

    psi_str = _psi_display(psi_raw)
    psi_badge = _badge_component(psi_status)

    # -----------------------
    # 2) Métricas OOT (sempre offline / validação)
    #   - KPIs rápidos: Precision/Recall usam Top-K (alerta)
    #   - Painel de coerência: usa K do tradeoff
    # -----------------------
    auc_val = _view_auc(dff)

    prec_alert = _precision_at_topk(dff, topk_alert)
    rec_alert = _recall_at_topk(dff, topk_alert)

    prec_k = _precision_at_topk(dff, k_tradeoff)
    rec_k = _recall_at_topk(dff, k_tradeoff)
    lift_k = _lift_at_topk(dff, k_tradeoff)

    auc_str = f"{auc_val:.4f}" if auc_val is not None else "—"

    prec_alert_str = f"{prec_alert * 100:.2f}%" if prec_alert is not None else "—"
    rec_alert_str = f"{rec_alert * 100:.2f}%" if rec_alert is not None else "—"

    prec_k_str = f"{prec_k * 100:.2f}%" if prec_k is not None else "—"
    rec_k_str = f"{rec_k * 100:.2f}%" if rec_k is not None else "—"
    lift_k_str = f"{lift_k:.2f}" if lift_k is not None else "—"

    auc_badge = _badge_component(_status_auc(auc_val))
    prec_alert_badge = _badge_component(_status_precision(prec_alert))
    rec_alert_badge = _badge_component(_status_recall(rec_alert))

    prec_k_badge = _badge_component(_status_precision(prec_k))
    rec_k_badge = _badge_component(_status_recall(rec_k))
    lift_k_badge = _badge_component(_status_lift(lift_k))

    # -----------------------
    # 3) Matriz Drift × Coerência (como o print: tudo verde, só célula atual rosa)
    # -----------------------
    has_drift = (psi_raw is not None) and (psi_raw >= 0.10)
    is_coerente = (auc_val is not None and auc_val >= 0.70) and (prec_alert is not None and prec_alert >= 0.65)

    x_labels = ["Coerente", "Não coerente"]
    y_labels = ["Drift", "Estável"]
    row = 0 if has_drift else 1
    col = 0 if is_coerente else 1

    z = [[0, 0], [0, 0]]
    z[row][col] = 1

    fig_matriz = go.Figure()
    fig_matriz.add_trace(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            zmin=0,
            zmax=1,
            colorscale=[[0.0, "#eafff2"], [1.0, "#ffd6d6"]],
            showscale=True,  # print tem barra 0..1
            colorbar=dict(thickness=12, len=0.75, tickvals=[0, 0.5, 1], ticktext=["0", "0.5", "1"]),
            hovertemplate="Status: %{y} × %{x}<extra></extra>",
        )
    )
    fig_matriz.add_trace(go.Scatter(x=[x_labels[col]], y=[y_labels[row]], mode="markers", marker=dict(size=18, color="#111827"), hoverinfo="skip", showlegend=False))
    fig_matriz.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(tickfont=dict(size=12)), yaxis=dict(tickfont=dict(size=12), autorange="reversed"))
    matrix_caption = "Drift (eixo Y) | Coerência (eixo X) — posição atual marcada."

    # -----------------------
    # 4) PSI trend (histórico)
    # -----------------------
    drift_mode = "oot" if context == "oot" else "prod"
    window = janela if context == "prod" else "30d"
    events = _get_drift_history(drift_mode, window)

    fig_psi = go.Figure()
    fig_psi.update_layout(height=280, margin=dict(t=35, b=10, l=10, r=10))
    fig_psi.add_hline(y=0.10, line_dash="dot", line_width=2)
    fig_psi.add_hline(y=0.20, line_dash="dot", line_width=2)
    fig_psi.update_layout(title="PSI (histórico)", yaxis_title="PSI", xaxis_title="")

    psi_meta = f"N eventos: {len(events)}"
    psi_window = f"Janela: {window}"

    if events:
        ts_vals = [str(e.get("timestamp_utc", ""))[:16] for e in events[-30:]]
        psi_plot = []
        for e in events[-30:]:
            p = e.get("psi")
            if p is not None and not (isinstance(p, str) and "low_sample" in str(p).lower()):
                try:
                    psi_plot.append(float(p))
                except Exception:
                    psi_plot.append(np.nan)
            else:
                psi_plot.append(np.nan)
        fig_psi.add_trace(go.Scatter(x=ts_vals, y=psi_plot, mode="lines+markers", name="PSI"))

    # -----------------------
    # 5) Drift contributors (bins 0..1 em 10 bins)
    # -----------------------
    fig_contrib = go.Figure().update_layout(height=220, margin=dict(t=35, b=10, l=10, r=10), title="Contribuição por bin")
    contrib_data = []

    train_df = _safe_csv(TRAIN_SCORED_CSV)
    if train_df.empty:
        train_df = df.head(min(600, len(df)))

    if (not train_df.empty) and (not dff.empty) and ("score" in train_df.columns) and ("score" in dff.columns):
        b = pd.to_numeric(train_df["score"], errors="coerce").dropna().values / 100.0
        c = pd.to_numeric(dff["score"], errors="coerce").dropna().values / 100.0
        edges = np.linspace(0.0, 1.0, 11)

        b_counts, _ = np.histogram(np.clip(b, 0, 1), bins=edges)
        c_counts, _ = np.histogram(np.clip(c, 0, 1), bins=edges)

        b_pct = (b_counts + 1e-6) / (b_counts.sum() + 1e-6)
        c_pct = (c_counts + 1e-6) / (c_counts.sum() + 1e-6)

        contrib = (c_pct - b_pct) * np.log((c_pct + 1e-10) / (b_pct + 1e-10))
        contrib = np.maximum(contrib, 0)  # só positivo no visual

        labels = [f"[{edges[i]:.1f},{edges[i+1]:.1f})" for i in range(10)]
        fig_contrib.add_trace(go.Bar(x=labels, y=contrib.tolist(), name="contrib"))

        contrib_rows = []
        for i in range(10):
            contrib_rows.append({"name": labels[i], "baseline_pct": f"{(b_pct[i]*100):.1f}%", "current_pct": f"{(c_pct[i]*100):.1f}%", "contrib": f"{float(contrib[i]):.2f}"})
        contrib_data = sorted(contrib_rows, key=lambda r: float(r["contrib"]), reverse=True)

    # -----------------------
    # 6) Qualidade de dados (atualiza com dff filtrado)
    # -----------------------
    pct_missing = 0.0
    n_outliers = 0
    n_newcats = 0
    num_cols_list: list = []

    if not dff.empty:
        num_cols_list = dff.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols_list:
            denom = len(dff) * len(num_cols_list)
            if denom > 0:
                pct_missing = float(dff[num_cols_list].isna().sum().sum() / denom * 100)
            # Outliers (IQR): Q1-1.5*IQR ou Q3+1.5*IQR
            for col in num_cols_list[:20]:  # top 20 cols para performance
                s = pd.to_numeric(dff[col], errors="coerce").dropna()
                if len(s) >= 4:
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        n_outliers += int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        # Categorias novas: valores em cols object que não estavam no train
        train_df = _safe_csv(TRAIN_SCORED_CSV)
        if not train_df.empty:
            for col in dff.select_dtypes(include=["object"]).columns[:5]:
                if col in train_df.columns:
                    train_vals = set(train_df[col].dropna().astype(str).unique())
                    current_vals = set(dff[col].dropna().astype(str).unique())
                    n_newcats += len(current_vals - train_vals)

    kpi_missing = f"{pct_missing:.1f}%"
    kpi_missing_sub = f"{len(num_cols_list)} cols num"
    kpi_newcats = str(n_newcats) if n_newcats > 0 else "0"
    kpi_newcats_sub = "vs baseline"
    kpi_outliers = str(n_outliers) if n_outliers > 0 else "0"
    kpi_outliers_sub = "IQR (top 20 cols)"
    n_num = len(num_cols_list) if not dff.empty else 0
    dq_notes = f"Linhas: {len(dff)} | Cols num: {n_num}"

    # -----------------------
    # 7) Tradeoff K
    # -----------------------
    perf_note = "Métricas OOT (validação 2023→2024)."

    k_suggestion = ""
    if prec_k is not None and rec_k is not None:
        if rec_k < 0.30:
            k_suggestion = f"Com K={k_tradeoff}%, Precision≈{prec_k*100:.0f}% e Recall≈{rec_k*100:.0f}%. Para maior cobertura, considere K=20%."
        else:
            k_suggestion = f"Com K={k_tradeoff}%, Precision≈{prec_k*100:.0f}% e Recall≈{rec_k*100:.0f}%. Bom equilíbrio."

    fig_tradeoff = go.Figure().update_layout(height=250, margin=dict(t=35, b=10, l=10, r=10), title="Trade-off por K (Prec vs Recall)")
    if not dff.empty and "target" in dff.columns and "score" in dff.columns:
        ks = [10, 15, 20, 25]
        prec_vals, rec_vals = [], []
        for k in ks:
            p = _precision_at_topk(dff, k)
            r = _recall_at_topk(dff, k)
            prec_vals.append(p if p is not None else 0)
            rec_vals.append(r if r is not None else 0)
        fig_tradeoff.add_trace(go.Scatter(x=ks, y=prec_vals, mode="lines+markers", name="Precision"))
        fig_tradeoff.add_trace(go.Scatter(x=ks, y=rec_vals, mode="lines+markers", name="Recall"))

    # -----------------------
    # 8) Calibração
    # -----------------------
    fig_cal = go.Figure().update_layout(height=280, margin=dict(t=35, b=10, l=10, r=10), title="Calibração (OOT)", xaxis_title="Score médio predito", yaxis_title="Fração de positivos")
    cal_table = []
    if not dff.empty and "target" in dff.columns and "score" in dff.columns:
        y = pd.to_numeric(dff["target"], errors="coerce")
        s = pd.to_numeric(dff["score"], errors="coerce") / 100.0
        mask = y.notna() & s.notna()
        if mask.sum() > 10:
            yv, sv = y[mask].values, s[mask].values
            bins = np.linspace(0, 1, 6)
            mean_pred_list, mean_true_list = [], []
            for i in range(len(bins) - 1):
                m = (sv >= bins[i]) & (sv < bins[i + 1])
                if m.sum() > 0:
                    bucket_mean = float(sv[m].mean())
                    rate_val = float(yv[m].mean())
                    cal_table.append({"bucket": f"[{bins[i]:.1f}-{bins[i+1]:.1f})", "n": int(m.sum()), "rate": f"{rate_val:.4f}"})
                    mean_pred_list.append(bucket_mean)
                    mean_true_list.append(rate_val)
            if mean_pred_list:
                fig_cal.add_trace(go.Scatter(x=mean_pred_list, y=mean_true_list, mode="lines+markers", name="Modelo"))
                fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(dash="dash")))

    # -----------------------
    # 9) Ações
    # -----------------------
    actions = []
    if psi_raw is not None and psi_raw >= 0.20:
        actions.append(html.Li("Drift alto: manter monitoramento e preparar plano de retreino quando houver novos dados."))
    if auc_val is not None and auc_val < 0.70:
        actions.append(html.Li("AUC OOT abaixo de 0.70: investigar mudança de dados/features."))
    if prec_alert is not None and prec_alert < 0.65:
        actions.append(html.Li("Precision@K (Top-K) baixa: ajustar K ou revisar critério de alerta."))
    if not actions:
        actions.append(html.Li("Coerente com drift: manter uso e monitorar PSI + métricas OOT periodicamente."))

    action_footer = f"Snapshot: {len(dff)} registros | modo={context} | janela={window}"

    # -----------------------
    # Retorno (status normalizado: ok/critical para look and feel)
    # -----------------------
    return (
        psi_str, _normalize_status(psi_status), psi_badge,
        auc_str, "OOT (offline)", auc_badge,
        prec_alert_str, f"K={topk_alert}%", prec_alert_badge,
        rec_alert_str, f"K={topk_alert}%", rec_alert_badge,

        fig_matriz, matrix_caption,

        fig_psi, psi_meta, psi_window,

        fig_contrib, contrib_data,

        kpi_missing, kpi_missing_sub,
        kpi_newcats, kpi_newcats_sub,
        kpi_outliers, kpi_outliers_sub,
        dq_notes,

        auc_str, "OOT", auc_badge,
        prec_k_str, f"K={k_tradeoff}%", prec_k_badge,
        rec_k_str, f"K={k_tradeoff}%", rec_k_badge,
        lift_k_str, f"K={k_tradeoff}%", lift_k_badge,

        perf_note,
        k_suggestion,
        fig_tradeoff,
        fig_cal,
        cal_table,

        actions,
        action_footer,
    )


# --------------------------------------------------------------------------------------
# Retreino tab
# --------------------------------------------------------------------------------------
@app.callback(Output("baselines-table", "data"), Output("retrain-checklist", "children"), Input("main-tabs", "active_tab"))
def update_retrain_tab(active_tab):
    if active_tab != "tab-retrain":
        return no_update, no_update
    baselines = []
    if _BASELINES_SUBDIR.exists():
        for d in _BASELINES_SUBDIR.iterdir():
            if d.is_dir():
                mf = d / "baseline_manifest.json"
                if mf.exists():
                    m = _safe_json(mf, {})
                    baselines.append({"baseline_id": d.name, "n_students": m.get("n_students", "—"), "created_at": (m.get("created_at") or "—")[:19]})
    ptr = _safe_json(CURRENT_BASELINE_JSON, {})
    checklist = [html.Li("Baseline atual: " + str(ptr.get("baseline_id", "—"))), html.Li("Modelo carregado: verificar retrain_metadata.json")]
    return baselines, html.Ul(checklist)


# --------------------------------------------------------------------------------------
# Diagnóstico card
# --------------------------------------------------------------------------------------
@app.callback(Output("diag-status-title", "children"), Output("diag-status-text", "children"), Output("diag-rule-text", "children"), Input("filtro-ano", "value"))
def update_diag(ano):
    if ano == 2022:
        return ("Avaliação In-Sample (Treino)", "Pode apresentar overfitting. Não usar como métrica de produção.", "PSI≥0.1 e métricas OOT ok")
    if ano == 2023:
        return ("Avaliação OOT (Validação Temporal)", "Métrica válida para decisão de deploy.", "PSI≥0.1 e métricas OOT ok")
    return ("Diagnóstico", "Selecione o ano para ver o tipo de avaliação.", "PSI≥0.1 e métricas OOT ok")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8505)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
