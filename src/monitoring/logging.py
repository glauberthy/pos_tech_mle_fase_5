"""
Helper único de logging para monitoramento.

Formato obrigatório: TIMESTAMP | LEVEL | MODULE | MESSAGE | k=v | k=v

event_type (enum fixo): startup | predict_batch | drift | alert | explain | error
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

EVENT_TYPES = frozenset({"startup", "predict_batch", "drift", "alert", "explain", "error"})


def _format_value(v: Any) -> str:
    """Formata valor para k=v. Nunca loga DataFrames."""
    if isinstance(v, pd.DataFrame):
        return ""  # skip
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        if isinstance(v, float):
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return str(v)
    return str(v)


def log_event(
    logger: logging.Logger,
    level: str | int,
    message: str,
    event_type: str,
    request_id: str | None = None,
    **fields: Any,
) -> None:
    """
    Registra evento de monitoramento no formato textual padronizado.

    Formato: TIMESTAMP | LEVEL | MODULE | MESSAGE | event_type=X | request_id=Y | k=v | k=v
    (MODULE vem do logger.name automaticamente)

    Args:
        logger: Logger Python (ex: logging.getLogger(__name__))
        level: Nível (INFO, WARNING, ERROR) ou constante logging.INFO etc.
        message: Mensagem principal (ex: "Predict batch complete")
        event_type: Tipo fixo: startup | predict_batch | drift | alert | explain | error
        request_id: ID da requisição (UUID curto), opcional
        **fields: Campos adicionais no formato k=v (não loga DataFrames)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    parts = [message, f"event_type={event_type}"]
    if request_id:
        parts.append(f"request_id={request_id}")
    for k, v in sorted(fields.items()):
        formatted = _format_value(v)
        if formatted != "":
            parts.append(f"{k}={formatted}")
    log_message = " | ".join(parts)
    logger.log(level, log_message)
