# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile.dashboard  –  Hugging Face Space: passos-magicos-dashboard
# Multi-stage: compilação separada do runtime → imagem final enxuta
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# ── Purge de dependências transitivas pesadas desnecessárias ──────────────────
# shap puxa numba/llvmlite (GPU) - TreeExplainer não precisa
# pyarrow não é necessário (usamos CSV/Excel)
# graphviz, IPython: opcionais do catboost/shap
RUN pip uninstall -y \
        numba llvmlite \
        pyarrow \
        matplotlib fonttools pillow \
        graphviz \
        ipython ipykernel jupyter_client jupyter_core \
    2>/dev/null || true

RUN pip uninstall -y pip setuptools wheel 2>/dev/null || true

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# ── Código-fonte ──────────────────────────────────────────────────────────────
COPY dashboard/ dashboard/
COPY src/       src/

# ── Modelo pré-treinado (baked-in na imagem) ──────────────────────────────────
COPY models/ models/

# ── Diretório de dados (uploads de retreino em runtime) ───────────────────────
RUN mkdir -p data

# ── Wrapper de retreino (chamado via subprocess pelo dashapp.py) ──────────────
COPY train.py .

ENV PYTHONPATH=/app

# Defina em Settings → Variables do Space:
#   API_BASE_URL = https://<user>-passos-magicos-api.hf.space
ENV API_BASE_URL=""

EXPOSE 7860

CMD ["python", "dashboard/dashapp.py", "--host", "0.0.0.0", "--port", "7860"]
