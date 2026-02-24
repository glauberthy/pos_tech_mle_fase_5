from typing import Optional

from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    ra: str = Field(..., description="Identificador do aluno")
    turma: str = Field(..., description="Turma")
    fase: str = Field(..., description="Fase (ex: ALFA, FASE1, FASE2, ...)")
    instituicao: str = Field(default="UNKNOWN", description="Instituição de ensino")
    genero: str = Field(default="NA", description="Gênero (M/F/NA)")
    ano_ingresso: float = Field(..., description="Ano de ingresso na Passos Mágicos")
    ieg: Optional[float] = Field(default=None, description="Indicador de Engajamento (0-10)")
    iaa: Optional[float] = Field(default=None, description="Autoavaliação (0-10)")
    ips: Optional[float] = Field(default=None, description="Indicador Psicossocial (0-10)")
    ipp: Optional[float] = Field(default=None, description="Indicador de Ponto de Virada")
    matem: Optional[float] = Field(default=None, description="Nota de Matemática (0-10)")
    portug: Optional[float] = Field(default=None, description="Nota de Português (0-10)")
    ingles: Optional[float] = Field(default=None, description="Nota de Inglês (0-10)")
    ano_base: int = Field(default=2024, description="Ano de referência dos dados")


class PredictRequest(BaseModel):
    students: list[StudentInput]
    k_pct: float = Field(default=15.0, ge=5.0, le=50.0, description="Percentual de alerta Top-K%")


class StudentScore(BaseModel):
    ra: str
    score: float
    fase: str
    turma: str
    alerta: bool
    top3_factors: list[str]
    top3_values: list[float]


class PredictResponse(BaseModel):
    n_students: int
    n_alerta: int
    k_pct: float
    students: list[StudentScore]
