---
title: Passos Mágicos – API de Risco Escolar
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: FastAPI que prediz risco de defasagem escolar (PEDE/CatBoost)
---

# Passos Mágicos – API de Risco Escolar

API REST construída com **FastAPI** para prever o risco de entrada em defasagem
escolar no próximo ciclo, usando dados longitudinais do PEDE.

## Endpoints principais

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/predict` | Pontua um lote de alunos |
| `GET`  | `/alert` | Lista Top-K% em alerta por fase |
| `GET`  | `/explain/{ra}` | Explicação SHAP de um aluno |
| `GET`  | `/health` | Health-check |
| `GET`  | `/metrics/drift` | Relatório de drift PSI |

Documentação interativa disponível em `/docs` (Swagger UI).

## Stack

- **FastAPI** + Uvicorn
- **CatBoost** (modelo pré-treinado incluso na imagem)
- **SHAP** para explicabilidade
- Porta exposta: **7860** (padrão HF Spaces)

## Deploy local (teste)

```bash
docker build -f Dockerfile.api -t pm-api .
docker run -p 7860:7860 pm-api
```
