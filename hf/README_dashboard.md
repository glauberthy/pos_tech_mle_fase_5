---
title: Passos Mágicos – Dashboard de Risco Escolar
emoji: 📊
colorFrom: green
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Dashboard Plotly/Dash para análise de risco escolar (PEDE)
---

# Passos Mágicos – Dashboard de Risco Escolar

Dashboard interativo construído com **Plotly Dash** para análise de risco de
defasagem escolar, monitoramento de drift e retreino do modelo via upload de
planilha PEDE.

## Funcionalidades

- Visualização de scores por fase, turma e ano
- Alertas Top-K% estratificados por fase
- Explicações SHAP dos alunos em risco
- Histórico de avaliação e drift PSI do modelo
- Retreino completo via upload de nova planilha PEDE (`.xlsx`)

## Configuração (Settings → Variables no Space)

| Variável | Valor | Descrição |
|----------|-------|-----------|
| `API_BASE_URL` | `https://<user>-passos-magicos-api.hf.space` | URL do Space da API (opcional) |

## Stack

- **Plotly Dash** + Flask
- **CatBoost** (modelo pré-treinado incluso na imagem)
- Porta exposta: **7860** (padrão HF Spaces)

## Deploy local (teste)

```bash
docker build -f Dockerfile.dashboard -t pm-dashboard .
docker run -p 7860:7860 pm-dashboard
```
