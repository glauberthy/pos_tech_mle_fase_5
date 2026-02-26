from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html, dash_table
from dash.exceptions import PreventUpdate

ROOT_DIR     = Path(__file__).resolve().parent.parent
MODELS_DIR   = ROOT_DIR / "models"
DATA_DIR     = ROOT_DIR / "data"

_MODEL_SUBDIR      = MODELS_DIR / "model"
_EVAL_SUBDIR       = MODELS_DIR / "evaluation"
_MONITORING_SUBDIR = MODELS_DIR / "monitoring"

SCORED_HISTORY_CSV  = _EVAL_SUBDIR / "scored_history.csv"
SCORED_CSV          = _EVAL_SUBDIR / "valid_scored.csv"
EVAL_JSON           = _EVAL_SUBDIR / "evaluation_results.json"
COHORT_SUMMARY_JSON = _MODEL_SUBDIR / "cohort_summary.json"
DRIFT_JSONL         = _MONITORING_SUBDIR / "drift_history.jsonl"
RETRAIN_META_JSON   = _MONITORING_SUBDIR / "retrain_metadata.json"
MONITORING_LOG      = _MONITORING_SUBDIR / "monitoring.log"


def _safe_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


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


def _available_filters(df: pd.DataFrame) -> tuple[list[int], list[str], list[str]]:
    years = sorted(pd.to_numeric(df["ano_base"], errors="coerce").dropna().astype(int).unique().tolist()) if "ano_base" in df.columns else []
    phases = sorted(df["fase"].dropna().astype(str).unique().tolist(), key=_phase_sort_key) if "fase" in df.columns else []
    turmas = sorted(df["turma"].dropna().astype(str).unique().tolist()) if "turma" in df.columns else []
    return years, phases, turmas


def _feedback_payload(message: str, seconds: int = 20) -> dict:
    _ = seconds
    return {"message": str(message)}


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
    reasons = []
    for col, msg in rules:
        if col in row.index and pd.notna(row[col]) and float(row[col]) < 6:
            reasons.append(msg)
    if not reasons:
        reasons = ["Padrão de risco global acima da referência da fase"]
    return ", ".join(reasons[:3])


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


def _view_auc(df: pd.DataFrame) -> float | None:
    if df.empty or not {"target", "score"}.issubset(df.columns):
        return None
    y = pd.to_numeric(df["target"], errors="coerce")
    s = pd.to_numeric(df["score"], errors="coerce") / 100.0
    mask = y.notna() & s.notna()
    if mask.sum() == 0:
        return None
    y = y[mask].astype(int).values
    s = s[mask].values
    if np.unique(y).size < 2:
        return None
    return float(_roc_auc_local(y, s))


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
    top = work.nlargest(k, "score")
    return float(top["target"].mean())


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
    top = work.nlargest(k, "score")
    tp = float(top["target"].sum())
    return float(tp / positives)


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
    if p_at_k is None:
        return None
    return float(p_at_k / base_rate)


def _safe_tail_text(path: Path, max_chars: int = 5000) -> str:
    if not path.exists():
        return "Sem log de monitoramento disponível."
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "Falha ao ler log de monitoramento."
    if len(text) <= max_chars:
        return text or "Log vazio."
    return text[-max_chars:]


def _roc_auc_local(y_true: np.ndarray, scores: np.ndarray) -> float:
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


def _kpi_cards(df: pd.DataFrame, global_auc: float | None) -> html.Div:
    total = len(df)
    n_alerta = int(df["alerta"].sum()) if "alerta" in df.columns else 0
    pct_alerta = (100 * n_alerta / total) if total else 0
    local_auc = _view_auc(df)

    def _card(title: str, value: str):
        return html.Div(
            [html.Div(title, style={"fontSize": "12px", "color": "#667"}), html.Div(value, style={"fontSize": "22px", "fontWeight": "bold"})],
            style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "background": "white"},
        )

    auc_text = f"{local_auc:.4f}" if local_auc is not None else (f"{global_auc:.4f}" if global_auc is not None else "—")
    auc_title = "AUC (recorte atual)" if local_auc is not None else "AUC (validação global)"
    return html.Div(
        [
            _card("Total de alunos", f"{total:,}".replace(",", ".")),
            _card("Quantidade em alerta", f"{n_alerta:,}".replace(",", ".")),
            _card("% em alerta", f"{pct_alerta:.1f}%"),
            _card(auc_title, auc_text),
        ],
        style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginBottom": "12px"},
    )


def _layout(df: pd.DataFrame):
    years, phases, turmas = _available_filters(df)

    return html.Div(
        style={"fontFamily": "Arial", "padding": "16px", "maxWidth": "1280px", "margin": "0 auto"},
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="refresh-token", data=0),
            dcc.Store(id="retrain-feedback-store", data={}),
            html.H2("Painel de Monitoramento - Risco de Defasagem"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr 1fr", "gap": "12px", "marginBottom": "12px"},
                children=[
                    html.Div([html.Label("Ano-base"), dcc.Dropdown(id="filtro-ano", options=[{"label": str(y), "value": y} for y in years], value=(max(years) if years else None), clearable=False)]),
                    html.Div([html.Label("Fase"), dcc.Dropdown(id="filtro-fase", options=[{"label": str(v), "value": str(v)} for v in phases], value=phases, multi=True)]),
                    html.Div([html.Label("Turma"), dcc.Dropdown(id="filtro-turma", options=[{"label": t, "value": t} for t in turmas], value=[], multi=True)]),
                    html.Div([html.Label("Buscar RA"), dcc.Input(id="filtro-ra", type="text", placeholder="Ex.: RA-001", style={"width": "100%"})]),
                    html.Div([html.Label("Top-K% por fase"), dcc.Slider(id="filtro-topk", min=10, max=25, step=5, value=15, marks={10: "10%", 15: "15%", 20: "20%", 25: "25%"})]),
                ],
            ),
            dcc.Tabs(
                id="tabs",
                value="inicio",
                children=[
                    dcc.Tab(label="Início", value="inicio"),
                    dcc.Tab(label="Alertas", value="alertas"),
                    dcc.Tab(label="Distribuição por Fase", value="fase"),
                    dcc.Tab(label="Saúde do Modelo", value="saude"),
                    dcc.Tab(label="Monitoramento (Logs + Drift)", value="monitoramento"),
                    dcc.Tab(label="Dados e Retreinamento", value="retrain"),
                ],
            ),
            html.Hr(),
            html.Div(id="page-content"),
        ],
    )


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Monitoramento de Risco Escolar"
app.layout = _layout(_load_scored())
app.server.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB


@app.callback(
    Output("filtro-ano", "options"),
    Output("filtro-ano", "value"),
    Output("filtro-fase", "options"),
    Output("filtro-fase", "value"),
    Output("filtro-turma", "options"),
    Output("filtro-turma", "value"),
    Input("refresh-token", "data"),
    State("filtro-ano", "value"),
    State("filtro-fase", "value"),
    State("filtro-turma", "value"),
)
def refresh_filter_options(_refresh, current_year, current_phases, current_turmas):
    df = _load_scored()
    years, phases, turmas = _available_filters(df)

    year_options = [{"label": str(y), "value": int(y)} for y in years]
    phase_options = [{"label": str(p), "value": str(p)} for p in phases]
    turma_options = [{"label": t, "value": t} for t in turmas]

    if years:
        year_value = int(current_year) if current_year in years else max(years)
    else:
        year_value = None

    valid_phases = [str(p) for p in (current_phases or []) if str(p) in set(phases)]
    if not valid_phases:
        valid_phases = phases

    valid_turmas = [str(t) for t in (current_turmas or []) if str(t) in set(turmas)]
    return year_options, year_value, phase_options, valid_phases, turma_options, valid_turmas


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("tabs", "value"),
    Input("filtro-ano", "value"),
    Input("filtro-fase", "value"),
    Input("filtro-turma", "value"),
    Input("filtro-ra", "value"),
    Input("filtro-topk", "value"),
    Input("refresh-token", "data"),
    Input("retrain-feedback-store", "data"),
)
def render_page(pathname, tab, ano, fases, turmas, ra_query, topk, _refresh, feedback):
    df = _load_scored()
    eval_data = _safe_json(EVAL_JSON, {})
    global_auc = eval_data.get("auc")
    if df.empty:
        return html.Div("Dados não encontrados. Execute train.py para gerar artefatos em models/.")

    # Enquanto estiver na rota de aluno, só prende na visão de detalhe na aba "Início".
    # Ao trocar para qualquer outra aba, o dashboard volta ao conteúdo normal do painel.
    if pathname and pathname.startswith("/aluno/") and tab == "inicio":
        ra = pathname.split("/aluno/", 1)[1].strip()
        row = df[df["ra"].astype(str) == str(ra)]
        if row.empty:
            return html.Div([html.H4("Aluno não encontrado"), dcc.Link("Voltar", href="/")])
        r = row.sort_values("ano_base").iloc[-1]
        hist = df[df["ra"].astype(str) == str(ra)].sort_values("ano_base")
        fig_hist = px.line(hist, x="ano_base", y="score", markers=True, title="Histórico de score por ano-base")
        indicator_cols = [c for c in ["matem", "portug", "ingles", "ieg", "iaa", "ips", "ipp"] if c in r.index]
        indicators = pd.DataFrame({"Indicador": indicator_cols, "Valor": [r[c] for c in indicator_cols]})
        return html.Div(
            [
                dcc.Link("← Voltar para o painel", href="/"),
                html.H3(f"Aluno {r.get('ra')}"),
                html.Div(f"Ano-base: {r.get('ano_base')} | Fase: {r.get('fase')} | Turma: {r.get('turma')}"),
                html.H4(f"Score de risco: {float(r.get('score', 0)):.2f}"),
                html.Div(_reason_text(r), style={"color": "#444", "marginBottom": "8px"}),
                dcc.Graph(figure=fig_hist),
                dash_table.DataTable(
                    columns=[{"name": "Indicador", "id": "Indicador"}, {"name": "Valor", "id": "Valor"}],
                    data=indicators.to_dict("records"),
                    style_cell={"padding": "8px"},
                ),
            ]
        )

    dff = _filtered_df(df, ano, fases or [], turmas or [], ra_query, topk)
    cards = _kpi_cards(dff, global_auc)

    if tab == "inicio":
        by_phase = dff.groupby("fase", as_index=False)["alerta"].sum().rename(columns={"alerta": "alertas"})
        fig = px.bar(by_phase, x="fase", y="alertas", title="Alertas por fase")
        return html.Div([cards, dcc.Graph(figure=fig)])

    if tab == "alertas":
        alert_df = dff[dff["alerta"]].sort_values("score", ascending=False).copy()
        alert_df["detalhe"] = alert_df["ra"].apply(lambda x: f"[Abrir](/aluno/{x})")
        cols = [c for c in ["ra", "fase", "turma", "score", "motivos", "detalhe"] if c in alert_df.columns]
        return html.Div(
            [
                cards,
                dash_table.DataTable(
                    columns=[{"name": "RA", "id": "ra"}, {"name": "Fase", "id": "fase"}, {"name": "Turma", "id": "turma"}, {"name": "Score", "id": "score", "type": "numeric", "format": {"specifier": ".2f"}}, {"name": "Motivos", "id": "motivos"}, {"name": "Detalhe", "id": "detalhe", "presentation": "markdown"}],
                    data=alert_df[cols].to_dict("records"),
                    filter_action="native",
                    sort_action="native",
                    page_size=15,
                    markdown_options={"link_target": "_self"},
                    style_cell={"padding": "8px", "whiteSpace": "normal", "height": "auto"},
                ),
            ]
        )

    if tab == "fase":
        by_phase = dff.groupby("fase", as_index=False).agg(total=("ra", "count"), alertas=("alerta", "sum"), score_medio=("score", "mean"))
        by_phase["pct_alerta"] = np.where(by_phase["total"] > 0, 100 * by_phase["alertas"] / by_phase["total"], 0)
        fig1 = px.bar(by_phase, x="fase", y="pct_alerta", title="% em alerta por fase")
        fig2 = px.bar(by_phase, x="fase", y="score_medio", title="Score médio por fase")
        return html.Div([cards, html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"})])

    if tab == "saude":
        rows = []
        for k in [10, 15, 20, 25]:
            if f"recall_top{k}" in eval_data:
                rows.append({"Métrica": f"Recall@Top{k}", "Valor": round(float(eval_data[f"recall_top{k}"]), 4)})
                rows.append({"Métrica": f"Precision@Top{k}", "Valor": round(float(eval_data[f"precision_top{k}"]), 4)})
                rows.append({"Métrica": f"Lift@Top{k}", "Valor": round(float(eval_data[f"lift_top{k}"]), 4)})
        perf_df = dff.copy()
        perf_rows = [
            {
                "Recorte": "Atual",
                "Métrica": "AUC",
                "Valor": f"{_view_auc(perf_df):.4f}" if _view_auc(perf_df) is not None else "—",
            }
        ]
        for k in [10, 15, 20]:
            r = _recall_at_topk(perf_df, k)
            p = _precision_at_topk(perf_df, k)
            l = _lift_at_topk(perf_df, k)
            perf_rows.append({"Recorte": "Atual", "Métrica": f"Recall@Top{k}", "Valor": f"{r:.4f}" if r is not None else "—"})
            perf_rows.append({"Recorte": "Atual", "Métrica": f"Precision@Top{k}", "Valor": f"{p:.4f}" if p is not None else "—"})
            perf_rows.append({"Recorte": "Atual", "Métrica": f"Lift@Top{k}", "Valor": f"{l:.4f}" if l is not None else "—"})

        by_year_rows = []
        if {"ano_base", "target", "score"}.issubset(dff.columns):
            for year, grp in dff.groupby("ano_base"):
                auc = _view_auc(grp)
                rec15 = _recall_at_topk(grp, 15)
                pre15 = _precision_at_topk(grp, 15)
                lif15 = _lift_at_topk(grp, 15)
                by_year_rows.append(
                    {
                        "Ano-base": int(year) if pd.notna(year) else "—",
                        "Qtd alunos": int(len(grp)),
                        "AUC": f"{auc:.4f}" if auc is not None else "—",
                        "Recall@Top15": f"{rec15:.4f}" if rec15 is not None else "—",
                        "Precision@Top15": f"{pre15:.4f}" if pre15 is not None else "—",
                        "Lift@Top15": f"{lif15:.4f}" if lif15 is not None else "—",
                    }
                )

        return html.Div(
            [
                cards,
                html.H4("Performance global do modelo (artefato de avaliação)"),
                dash_table.DataTable(
                    columns=[{"name": "Métrica", "id": "Métrica"}, {"name": "Valor", "id": "Valor"}],
                    data=rows,
                    style_cell={"padding": "8px"},
                    page_size=12,
                ),
                html.H4("Performance no recorte atual do dashboard"),
                dash_table.DataTable(
                    columns=[{"name": "Recorte", "id": "Recorte"}, {"name": "Métrica", "id": "Métrica"}, {"name": "Valor", "id": "Valor"}],
                    data=perf_rows,
                    style_cell={"padding": "8px"},
                    page_size=14,
                ),
                html.H4("Performance por ano-base (recorte atual)"),
                dash_table.DataTable(
                    columns=[
                        {"name": "Ano-base", "id": "Ano-base"},
                        {"name": "Qtd alunos", "id": "Qtd alunos"},
                        {"name": "AUC", "id": "AUC"},
                        {"name": "Recall@Top15", "id": "Recall@Top15"},
                        {"name": "Precision@Top15", "id": "Precision@Top15"},
                        {"name": "Lift@Top15", "id": "Lift@Top15"},
                    ],
                    data=by_year_rows,
                    style_cell={"padding": "8px"},
                    page_size=10,
                ),
            ]
        )

    if tab == "monitoramento":
        import os
        import json
        import urllib.request
        
        api_url = os.environ.get("API_BASE_URL", "").rstrip("/")
        drift_events = []
        log_text = "A aguardar eventos em tempo real da API..."
        
        # 1. Buscar dados dinâmicos da API
        if api_url:
            try:
                # AJUSTE: Inclusão do header 'User-Agent' para evitar bloqueio do Hugging Face
                req = urllib.request.Request(
                    f"{api_url}/metrics/drift/history",
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode())
                        drift_events = data.get("events", [])
            except Exception as e:
                log_text = f"Erro ao conectar à API para buscar logs: {e}\n\nMostrando dados locais (fallback)."
        
        # 2. Fallback para os ficheiros locais caso a API falhe ou não tenha eventos
        if not drift_events:
            drift_events = _safe_json(DRIFT_JSONL, [])
            log_text = _safe_tail_text(MONITORING_LOG)
        else:
            # Constrói um log em modo texto baseado nos eventos reais da API
            log_text = "=== LOGS DE MONITORAMENTO EM TEMPO REAL (API) ===\n\n"
            for ev in drift_events[-15:]:  # Mostra os últimos 15 eventos
                dt = ev.get('timestamp_utc', 'N/A')
                tipo = ev.get('event_type', 'N/A')
                psi = ev.get('psi', 0.0)
                sev = ev.get('severity', 'N/A')
                n_alunos = ev.get('n_students', 0)
                log_text += f"[{dt}] INFO | Evento: {tipo} | Alunos processados: {n_alunos} | PSI Global: {psi:.4f} | Status: {sev}\n"

        drift_df = pd.DataFrame(drift_events)
        graph = html.Div("Sem histórico de drift.")
        drift_table_rows = []
        
        if not drift_df.empty and {"timestamp_utc", "psi"}.issubset(drift_df.columns):
            drift_df["timestamp_utc"] = pd.to_datetime(drift_df["timestamp_utc"], errors="coerce")
            drift_plot_df = drift_df.dropna(subset=["timestamp_utc"]).copy()
            if not drift_plot_df.empty:
                fig = px.line(drift_plot_df, x="timestamp_utc", y="psi", markers=True, title="Evolução do Population Stability Index (PSI)")
                fig.add_hline(y=0.10, line_dash="dot", line_color="orange", annotation_text="Atenção (0.10)")
                fig.add_hline(y=0.25, line_dash="dot", line_color="red", annotation_text="Crítico (0.25)")
                graph = dcc.Graph(figure=fig)
            
            # Formatar para a tabela
            drift_table_rows = drift_df.tail(20).fillna("").to_dict("records")

        return html.Div(
            [
                cards,
                html.H4("Dashboard de Data Drift (Comunicação com API)"),
                graph,
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in (drift_df.columns.tolist() if not drift_df.empty else ["status"])],
                    data=(drift_table_rows if drift_table_rows else [{"status": "Sem eventos de drift registrados."}]),
                    style_cell={"padding": "8px"},
                    page_size=10,
                ),
                html.H4("Logging de Monitoramento"),
                html.Pre(
                    log_text,
                    style={
                        "background": "#0f172a",
                        "color": "#10b981", # Cor verde estilo terminal
                        "padding": "12px",
                        "borderRadius": "8px",
                        "whiteSpace": "pre-wrap",
                        "maxHeight": "320px",
                        "overflowY": "auto",
                        "fontFamily": "monospace"
                    },
                ),
            ]
        )

    return html.Div(
        [
            cards,
            html.H4("Adicionar novo arquivo para retreinar"),
            dcc.Upload(
                id="upload-xls",
                children=html.Div(["Arraste e solte ou ", html.A("selecione um .xlsx/.xls")]),
                style={
                    "width": "100%",
                    "height": "70px",
                    "lineHeight": "70px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "10px",
                    "textAlign": "center",
                },
                multiple=False,
                accept=".xlsx,.xls",
            ),
            html.Div(id="upload-info", style={"marginTop": "8px", "color": "#444"}),
            html.Br(),
            html.Button("Executar retreinamento", id="btn-retrain", n_clicks=0),
            dcc.Loading(
                type="circle",
                children=html.Div(
                    id="retrain-status",
                    children=str((feedback or {}).get("message", "")),
                    style={"marginTop": "10px"},
                ),
            ),
        ]
    )


@app.callback(
    Output("upload-info", "children"),
    Input("upload-xls", "filename"),
    prevent_initial_call=True,
)
def upload_info_callback(filename):
    if not filename:
        raise PreventUpdate
    if isinstance(filename, list):
        safe = Path(filename[0]).name if filename else ""
    else:
        safe = Path(str(filename)).name
    if not safe:
        raise PreventUpdate
    return f"Arquivo anexado: {safe}"


@app.callback(
    Output("retrain-feedback-store", "data"),
    Output("refresh-token", "data"),
    Input("btn-retrain", "n_clicks"),
    State("upload-xls", "contents"),
    State("upload-xls", "filename"),
    State("refresh-token", "data"),
    running=[
        (Output("btn-retrain", "disabled"), True, False),
        (Output("btn-retrain", "children"), "Treinando...", "Executar retreinamento"),
    ],
    prevent_initial_call=True,
)
def retrain_callback(n_clicks, contents, filename, token):
    if not n_clicks:
        raise PreventUpdate
    if not contents or not filename:
        return _feedback_payload("Envie um arquivo .xlsx para retreinar."), token
    if isinstance(filename, list):
        filename = filename[0] if filename else ""
    filename = str(filename)
    if not filename.lower().endswith((".xlsx", ".xls")):
        return _feedback_payload("Formato inválido. Envie .xlsx/.xls."), token

    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
    except Exception:
        return _feedback_payload("Falha ao ler o arquivo enviado. Tente anexar novamente."), token

    file_hash = hashlib.sha256(decoded).hexdigest()
    retrain_meta = _safe_json(RETRAIN_META_JSON, {})
    if retrain_meta.get("last_file_sha256") == file_hash:
        df_current = _load_scored()
        years_current, _, _ = _available_filters(df_current)
        years_current_txt = ", ".join(str(y) for y in years_current) if years_current else "nenhum"
        last_train_at = retrain_meta.get("last_trained_at_utc", "desconhecido")
        return (
            _feedback_payload(
                f"Arquivo com conteúdo já utilizado no último treino (SHA-256 igual). "
                f"Mesmo que o nome tenha mudado, o conteúdo é idêntico; retreino ignorado para evitar duplicidade. "
                f"Último treino: {last_train_at}. "
                f"Anos disponíveis no dashboard: {years_current_txt}."
            ),
            token,
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    upload_path = DATA_DIR / f"dash_upload_{safe_name}"

    try:
        xls = pd.ExcelFile(io.BytesIO(decoded))
        years_uploaded = sorted(
            [
                int(m.group(1))
                for name in xls.sheet_names
                for m in [re.match(r"PEDE(\d{4})$", str(name).strip(), flags=re.IGNORECASE)]
                if m
            ]
        )
    except Exception:
        years_uploaded = []

    with open(upload_path, "wb") as f:
        f.write(decoded)

    cmd = [sys.executable, str(ROOT_DIR / "train.py"), "--xls", str(upload_path), "--model-dir", str(MODELS_DIR), "--log-level", "INFO"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR), timeout=3600)
    except subprocess.TimeoutExpired:
        return _feedback_payload("Retreino excedeu o tempo limite (1h). Tente novamente com um arquivo menor."), token

    if proc.returncode != 0:
        return _feedback_payload(f"Falha no retreino: {(proc.stderr or proc.stdout)[-1200:]}"), token

    df_after = _load_scored()
    years_after, _, _ = _available_filters(df_after)
    years_uploaded_txt = ", ".join(str(y) for y in years_uploaded) if years_uploaded else "não identificados"
    years_after_txt = ", ".join(str(y) for y in years_after) if years_after else "nenhum"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    retrain_payload = {
        "last_file_name": safe_name,
        "last_file_sha256": file_hash,
        "last_trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "years_in_uploaded_file": years_uploaded,
        "years_available_after_training": years_after,
    }
    with open(RETRAIN_META_JSON, "w", encoding="utf-8") as f:
        json.dump(retrain_payload, f, ensure_ascii=False, indent=2)

    message = (
        f"Retreino concluído com sucesso. "
        f"Arquivo anexado: {safe_name}. "
        f"Anos no arquivo: {years_uploaded_txt}. "
        f"Anos disponíveis no dashboard após treino: {years_after_txt}."
    )
    return _feedback_payload(message), int(token) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dash dashboard server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)