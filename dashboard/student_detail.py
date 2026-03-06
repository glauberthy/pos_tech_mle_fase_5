"""
Layout de detalhe do aluno — Passos Mágicos.
Acessível via [Abrir](/aluno/RA-xxx) na tela de Alertas.
"""
from __future__ import annotations

import ast
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


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


def _fetch_explain(ra: str) -> dict:
    api_url = os.environ.get("API_BASE_URL", "").rstrip("/")
    if not api_url:
        return {}
    try:
        req = urllib.request.Request(
            f"{api_url}/explain/{urllib.parse.quote(ra)}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode())
    except Exception:
        pass
    return {}


def _normalize_ra_for_match(ra_raw: str) -> list[str]:
    ra = str(ra_raw).strip()
    candidates = [ra]
    if ra.upper().startswith("RA-"):
        candidates.append(ra[3:].strip())
    elif ra.upper().startswith("RA"):
        candidates.append(ra[2:].lstrip("- ").strip())
    if ra.isdigit() or (ra.replace("-", "").isdigit()):
        candidates.append(f"RA-{ra}")
    return list(dict.fromkeys(candidates))


def _parse_top3(row: pd.Series) -> list[dict]:
    factors = row.get("top3_factors")
    values = row.get("top3_values")
    if factors is None or values is None:
        return []
    if isinstance(factors, str):
        try:
            factors = ast.literal_eval(factors)
        except Exception:
            return []
    if isinstance(values, str):
        try:
            values = ast.literal_eval(values)
        except Exception:
            return []
    if not isinstance(factors, (list, tuple)) or not isinstance(values, (list, tuple)):
        return []
    result = []
    for feat, val in zip(factors[:5], values[:5]):
        sv = float(val) if val is not None else 0
        student_val = float(row[feat]) if feat in row.index and pd.notna(row[feat]) else None
        result.append({
            "feature": str(feat),
            "value": student_val,
            "reference": "—",
            "impact": round(sv, 4),
            "direction": "↑ risco" if sv > 0 else "↓ risco",
        })
    return result


def _factors_from_reason(row: pd.Series) -> list[dict]:
    reasons = []
    for col, msg in [("matem", "Matemática"), ("portug", "Português"), ("ieg", "IEG"), ("ips", "IPS")]:
        if col in row.index and pd.notna(row[col]):
            v = float(row[col])
            if v < 6:
                reasons.append({"feature": msg, "value": v, "reference": "≥6", "impact": 6 - v, "direction": "↑ risco"})
    return reasons[:5]


def build_student_detail(ra: str, df: pd.DataFrame, eval_json: dict) -> dbc.Container:
    """Constrói o layout completo do detalhe do aluno."""
    ra_candidates = _normalize_ra_for_match(ra)
    row = pd.DataFrame()
    for cand in ra_candidates:
        row = df[df["ra"].astype(str).str.strip() == str(cand).strip()]
        if not row.empty:
            break
    if row.empty:
        return dbc.Container(fluid=True, children=[
            dbc.Alert("Aluno não encontrado.", color="warning"),
            dcc.Link("← Voltar para o painel", href="/"),
        ])

    r = row.sort_values("ano_base").iloc[-1]
    ra_matched = str(r["ra"]).strip()
    hist = df[df["ra"].astype(str).str.strip() == ra_matched].sort_values("ano_base")
    explain_data = _fetch_explain(ra)

    score = float(r.get("score", 0))
    ano = r.get("ano_base", "—")
    fase = str(r.get("fase", ""))
    turma = str(r.get("turma", ""))
    alerta = bool(r.get("alerta", False))
    motivos = _reason_text(r)

    top_factors = explain_data.get("top_factors", [])
    if not top_factors:
        factors_from_row = _parse_top3(r)
        if not factors_from_row:
            factors_from_row = _factors_from_reason(r)
        factors_table = [{"feature": f["feature"], "value": f.get("value"), "reference": f.get("reference", "—"), "impact": f.get("impact"), "direction": f.get("direction", "")} for f in factors_from_row]
    else:
        factors_table = []
        for t in top_factors:
            factors_table.append({
                "feature": t.get("feature", ""),
                "value": t.get("student_value"),
                "reference": "—",
                "impact": round(t.get("shap_value", 0), 4),
                "direction": "↑ risco" if (t.get("shap_value") or 0) > 0 else "↓ risco",
            })

    if factors_table:
        fig_shap = go.Figure(go.Bar(
            x=[f["impact"] for f in factors_table],
            y=[f["feature"] for f in factors_table],
            orientation="h",
            marker_color=["#ef4444" if f["impact"] > 0 else "#22c55e" for f in factors_table],
        ))
        fig_shap.update_layout(title="Impacto dos fatores (SHAP)", height=220, margin=dict(l=120))
    else:
        fig_shap = go.Figure().add_annotation(text="Sem dados SHAP", x=0.5, y=0.5, showarrow=False)
        fig_shap.update_layout(height=220)

    if len(hist) > 1:
        fig_timeline = go.Figure(go.Scatter(
            x=hist["ano_base"].astype(str),
            y=hist["score"],
            mode="lines+markers",
            name="Score",
        ))
        fig_timeline.update_layout(title="Histórico de score", height=220, margin=dict(t=30, b=40))
    else:
        fig_timeline = go.Figure().add_annotation(text="Um único registro", x=0.5, y=0.5, showarrow=False)
        fig_timeline.update_layout(height=220)

    fase_df = df[df["fase"].astype(str) == str(fase)] if fase else pd.DataFrame()
    if not fase_df.empty and "score" in fase_df.columns:
        fig_dist = go.Figure(go.Histogram(x=fase_df["score"], nbinsx=20, name="Fase"))
        fig_dist.add_vline(x=score, line_dash="dash", line_color="red", annotation_text="Aluno")
        fig_dist.update_layout(title="Distribuição de scores na fase", height=220)
    else:
        fig_dist = go.Figure().update_layout(height=220)

    retrain = {}
    try:
        p = Path(__file__).resolve().parent.parent / "models" / "monitoring" / "retrain_metadata.json"
        if p.exists():
            retrain = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass

    return dbc.Container(fluid=True, children=[

        dbc.Card(className="shadow-sm mb-3", children=[
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H4("Detalhe do Aluno", className="mb-0"),
                        html.Div("Passos Mágicos — Datathon", className="text-muted"),
                    ]),
                    dbc.Col([
                        dbc.Badge("Em alerta" if alerta else "Monitorar", color="danger" if alerta else "secondary", className="me-2"),
                        dbc.Badge(f"Score: {score:.1f}", color="primary"),
                    ], width="auto", className="text-end")
                ], align="center"),
                html.Hr(className="my-2"),
                dbc.Row(className="g-2", children=[
                    dbc.Col([
                        html.Div("Aluno", className="text-muted small"),
                        html.H5(ra_matched, className="mb-0"),
                    ], md=4),
                    dbc.Col([
                        dcc.Link(
                            dbc.Button("← Voltar para Alertas", color="secondary", outline=True),
                            href="/",
                            style={"textDecoration": "none"},
                        ),
                    ], md=8, className="text-end"),
                ])
            ])
        ]),

        dbc.Row(className="g-3 mb-3", children=[
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("RA/ID", className="text-muted"), html.H4(ra_matched, className="mb-0")]), className="shadow-sm h-100"), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Ano-base", className="text-muted"), html.H4(str(ano), className="mb-0")]), className="shadow-sm h-100"), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Fase / Turma", className="text-muted"), html.H4(f"{fase} / {turma}", className="mb-0")]), className="shadow-sm h-100"), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Score", className="text-muted"), html.H4(f"{score:.2f}", className="mb-0"), dbc.Badge("Alerta" if alerta else "Ok", color="danger" if alerta else "success")]), className="shadow-sm h-100"), md=3),
        ]),

        dbc.Card(className="shadow-sm mb-3", children=[
            dbc.CardBody([
                html.Div("Resumo executivo", className="text-muted"),
                html.Div(motivos, className="mt-1"),
                html.Ul([html.Li(f"Score de risco: {score:.2f} (escala 0–100)"), html.Li(f"Fase {fase}, Turma {turma}")], className="mb-0 mt-2"),
            ])
        ]),

        dbc.Row(className="g-3 mb-3", children=[
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Evidência do alerta", className="text-muted"),
                dcc.Graph(figure=fig_dist, config={"displayModeBar": False}),
            ]), className="shadow-sm h-100"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Histórico temporal", className="text-muted"),
                dcc.Graph(figure=fig_timeline, config={"displayModeBar": False}),
            ]), className="shadow-sm h-100"), md=6),
        ]),

        dbc.Row(className="g-3 mb-3", children=[
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Principais fatores de risco (SHAP/heurísticas)", className="text-muted"),
                dcc.Graph(figure=fig_shap, config={"displayModeBar": False}),
                html.Hr(),
                dash_table.DataTable(
                    columns=[{"name": "Fator", "id": "feature"}, {"name": "Valor", "id": "value"}, {"name": "Impacto", "id": "impact"}, {"name": "Direção", "id": "direction"}],
                    data=factors_table,
                    page_size=8,
                    style_cell={"padding": "6px"},
                    style_header={"fontWeight": "600"},
                ),
            ]), className="shadow-sm"), md=12),
        ]),

        dbc.Card(className="shadow-sm mb-3", children=[
            dbc.CardBody([
                html.Div("Sugestões de atenção (apoio pedagógico)", className="text-muted"),
                html.Div(motivos, className="mt-1"),
                html.Ul([
                    html.Li("Revisar indicadores de desempenho (matemática, português)."),
                    html.Li("Acompanhar engajamento (IEG) e sinais psicossociais (IPS)."),
                    html.Li("O score é estimativa probabilística; não substitui avaliação pedagógica."),
                ], className="mb-0"),
            ])
        ]),

        dbc.Accordion(children=[
            dbc.AccordionItem(
                title="Detalhes técnicos (auditoria)",
                children=[
                    html.Div(f"model_version: 1.0.0 | baseline: valid_scored | último treino: {retrain.get('last_trained_at_utc', 'N/A')[:19]}"),
                    html.Div("O score é uma estimativa probabilística e não substitui avaliação pedagógica.", className="text-muted small mt-2"),
                ]
            )
        ]),
    ])
