"""
FastAPI – Passos Mágicos Risk Prediction API.

Endpoints:
    POST /predict          → score a batch of students
    GET  /alert            → retrieve stratified Top-K% alert list
    GET  /explain/{ra}     → SHAP explanation for one student
    GET  /health           → health check
    GET  /metrics/drift    → PSI drift report vs training baseline

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# application logic lives in a separate module
from . import routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # called at startup
    routes.load_state()
    yield
    # shutdown logic (if any) can go here

app = FastAPI(
    title="Passos Mágicos – Sistema de Previsão de Risco Escolar",
    description=(
        "Prevê o risco de entrada em defasagem escolar no próximo ciclo "
        "usando dados longitudinais do PEDE."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# include router and register startup helper
app.include_router(routes.router)

# schemas are defined in api/schemas.py and imported above


# application state is managed inside api/routes.py


# helper functions live in api/routes.py


# startup is handled by the lifespan context manager defined above


# endpoints now live in ``api/routes.py``; they are automatically
# included when the router is added to this application

# all of the endpoint handlers have been moved into ``api/routes.py``
# and are registered via ``app.include_router(routes.router)`` above.
