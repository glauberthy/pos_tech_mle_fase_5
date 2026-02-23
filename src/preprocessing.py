"""
Preprocessing and longitudinal dataset assembly.

New target strategy from notebooks:
    target = 1 if defasagem_t1 > defasagem_t else 0
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import normalize_colname, strip_accents

logger = logging.getLogger(__name__)

# Backward-compatible aliases (used by existing tests and internal callers)
_strip_accents = strip_accents
_normalize_colname = normalize_colname


def _fuzzy_rename(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Compatibility helper for tests and mixed schemas."""
    actual_cols = list(df.columns)
    mapping: dict[str, str] = {}
    for target_name, canonical in rename_map.items():
        if target_name in actual_cols:
            mapping[target_name] = canonical
            continue
        target_ascii = target_name.encode("ascii", "ignore").decode().lower()
        for col in actual_cols:
            col_ascii = col.encode("ascii", "ignore").decode().lower()
            if col_ascii == target_ascii:
                mapping[col] = canonical
                break
    return df.rename(columns=mapping)


def _extract_phase(fase_raw) -> str:
    """Legacy textual phase normalization used in tests."""
    if pd.isna(fase_raw):
        return "UNKNOWN"
    s = str(fase_raw).strip().upper()
    if s == "ALFA":
        return "ALFA"
    if s.startswith("FASE"):
        digits = re.findall(r"\d+", s)
        if digits:
            return f"FASE{int(digits[0])}"
    if s.isdigit():
        n = int(s)
        return "ALFA" if n == 0 else f"FASE{n}"
    if s and s[0].isdigit():
        return f"FASE{int(s[0])}"
    return s


def _parse_fase_int(value):
    """Unified phase parser: ALFA→0, FASE N / N → int (0-9), else pd.NA."""
    if pd.isna(value):
        return pd.NA
    v = str(value).strip().upper()
    if v == "ALFA":
        return 0
    m = re.search(r"\d+", v)
    if m:
        fase = int(m.group())
        if 0 <= fase <= 9:
            return fase
    return pd.NA


# Backward-compatible alias kept for existing tests
_normalize_fase_numeric = _parse_fase_int


def _normalize_gender(value) -> str:
    if pd.isna(value):
        return "NA"
    s = _normalize_colname(str(value))
    if s in {"menino", "masculino", "m"}:
        return "M"
    if s in {"menina", "feminino", "f"}:
        return "F"
    return "NA"


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    normalized = {_normalize_colname(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    aliases = {
        "ra": "ra",
        "turma": "turma",
        "fase": "fase_raw",
        "genero": "genero",
        "instituicao_de_ensino": "instituicao",
        "instituicao": "instituicao",
        "escola": "escola",
        "ano_ingresso": "ano_ingresso",
        "ano_nasc": "ano_nasc",
        "idade": "idade",
        "matem": "matem",
        "mat": "matem",
        "portug": "portug",
        "por": "portug",
        "ingles": "ingles",
        "ing": "ingles",
        "ieg": "ieg",
        "iaa": "iaa",
        "ips": "ips",
        "ipp": "ipp",
        "defasagem": "defasagem",
        "defas": "defasagem",
    }
    for alias, canonical in aliases.items():
        col = normalized.get(alias)
        if col is not None:
            mapping[col] = canonical
    return df.rename(columns=mapping)


def _load_sheet(xl: pd.ExcelFile, sheet_name: str, year: int) -> pd.DataFrame:
    df = xl.parse(sheet_name)
    df = _canonicalize_columns(df)
    df["ano_base"] = int(year)

    required = [
        "ra",
        "ano_base",
        "fase_raw",
        "turma",
        "defasagem",
        "matem",
        "portug",
        "ingles",
        "ieg",
        "iaa",
        "ips",
        "ipp",
        "genero",
        "instituicao",
        "escola",
        "ano_ingresso",
        "ano_nasc",
        "idade",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df["ra"] = df["ra"].astype(str).str.strip()
    df["fase"] = df["fase_raw"].apply(_parse_fase_int).astype("Int64")
    df["genero"] = df["genero"].apply(_normalize_gender)

    for c in ["defasagem", "matem", "portug", "ingles", "ieg", "iaa", "ips", "ipp", "ano_ingresso", "ano_nasc", "idade"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "idade" in df.columns and "ano_nasc" in df.columns:
        missing_age = df["idade"].isna() & df["ano_nasc"].notna()
        df.loc[missing_age, "idade"] = df.loc[missing_age, "ano_base"] - df.loc[missing_age, "ano_nasc"]

    logger.info("Loaded %s (%d): %d rows", sheet_name, year, len(df))
    return df


def load_all_years(xls_path: str | Path) -> dict[int, pd.DataFrame]:
    """Load all PEDE#### sheets and standardize schema."""
    xl = pd.ExcelFile(xls_path, engine="openpyxl")
    by_year: dict[int, str] = {}
    for name in xl.sheet_names:
        m = re.match(r"PEDE(\d{4})$", str(name).strip(), flags=re.IGNORECASE)
        if not m:
            continue
        by_year[int(m.group(1))] = name
    if len(by_year) < 3:
        raise ValueError("É necessário pelo menos 3 abas PEDE#### para split temporal.")

    data: dict[int, pd.DataFrame] = {}
    for year in sorted(by_year):
        data[year] = _load_sheet(xl, by_year[year], year)
    return data


def _build_pair(df_t: pd.DataFrame, df_t1: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Build one (t -> t+1) pair table with new target:
    target = 1 if defasagem_t1 > defasagem_t else 0
    """
    current = df_t.copy()
    current["ra"] = current["ra"].astype(str).str.strip()
    nxt = df_t1[["ra", "defasagem"]].copy()
    nxt["ra"] = nxt["ra"].astype(str).str.strip()
    nxt = nxt.rename(columns={"defasagem": "defasagem_t1"})

    merged = current.merge(nxt, on="ra", how="inner")
    merged = merged.rename(columns={"defasagem": "defasagem_t"})
    merged["target"] = (merged["defasagem_t1"] > merged["defasagem_t"]).astype(int)

    n_pos = int(merged["target"].sum())
    n = len(merged)
    logger.info("%s: n=%d | n_pos=%d (%.1f%%)", label, n, n_pos, 100 * n_pos / n if n else 0)
    return merged


def build_longitudinal_dataset(data: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dynamic temporal split:
      train = all adjacent pairs except latest
      valid = latest adjacent pair
    """
    years = sorted(data.keys())
    if len(years) < 3:
        raise ValueError("É necessário pelo menos 3 anos para treino/validação temporal.")

    pair_dfs: list[tuple[tuple[int, int], pd.DataFrame]] = []
    for i in range(len(years) - 1):
        y_t, y_t1 = years[i], years[i + 1]
        pair = _build_pair(data[y_t], data[y_t1], f"pair ({y_t}->{y_t1})")
        pair["pair_label"] = f"{y_t}->{y_t1}"
        pair_dfs.append(((y_t, y_t1), pair))

    valid_pair, valid_df = pair_dfs[-1]
    train_parts = [df for _, df in pair_dfs[:-1]]
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else valid_df.iloc[0:0].copy()

    logger.info(
        "Temporal split | train_pairs=%s | valid_pair=%s",
        [f"{p[0]}->{p[1]}" for p, _ in pair_dfs[:-1]],
        f"{valid_pair[0]}->{valid_pair[1]}",
    )
    return train_df, valid_df.reset_index(drop=True)


# ============================================================
# Notebook-aligned helpers (preprocessing.ipynb)
# ============================================================


def normalizar_nome_colunas(col: str) -> str:
    """Normaliza nome de coluna: remove acentos, lowercase, underscore.

    Delegates to :func:`src.utils.normalize_colname`.
    """
    return normalize_colname(col)


def apply_canonical_renames(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas para nomes canônicos, cobrindo todas as variações dos dados PEDE."""
    df = df.copy()
    rename_candidates = {
        "ra": "ra",
        "turma": "turma",
        "nome": "nome",
        "nome_anonimizado": "nome",
        "genero": "genero",
        "instituicao_de_ensino": "instituicao",
        "instituicao": "instituicao",
        "escola": "escola",
        "ano_ingresso": "ano_ingresso",
        "fase": "fase_raw",
        "fase_raw": "fase_raw",
        "matem": "matem",
        "mat": "matem",
        "por": "portug",
        "portug": "portug",
        "ingles": "ingles",
        "ing": "ingles",
        "ieg": "ieg",
        "iaa": "iaa",
        "ips": "ips",
        "ipp": "ipp",
        "no_av": "no_av",
        "defas": "defasagem",
        "defasagem": "defasagem",
        "idade": "idade",
        "ano_nasc": "ano_nasc",
        "data_de_nasc": "data_nasc",
        "inde_22": "inde_2022",
        "inde_23": "inde_2023",
        "inde_24": "inde_2024",
        "pedra_21": "pedra_2021",
        "pedra_20": "pedra_2020",
        "pedra_22": "pedra_2022",
        "pedra_23": "pedra_2023",
        "pedra_24": "pedra_2024",
        "cf": "class_fase",
        "cg": "class_geral",
        "ct": "class_turma",
    }
    df = df.rename(columns={c: rename_candidates[c] for c in df.columns if c in rename_candidates})
    return df


def coerce_types_and_clean(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Coerce tipos, resolve datas, sanitiza strings e garante colunas mínimas."""
    df = df.copy()
    df["ano_base"] = int(year)

    must_have = [
        "ra", "ano_base", "fase_raw", "turma", "defasagem",
        "matem", "portug", "ingles", "ieg", "iaa", "ips", "ipp",
        "genero", "instituicao", "ano_ingresso",
    ]
    for c in must_have:
        if c not in df.columns:
            df[c] = np.nan

    df["ra"] = df["ra"].astype(str).str.strip()
    df["fase"] = (
        df["fase_raw"].astype(str).str.strip().str.upper()
        .replace({"NAN": np.nan, "nan": np.nan, "": np.nan})
    )

    if "data_nasc" in df.columns:
        df["data_nasc"] = pd.to_datetime(df["data_nasc"], errors="coerce", dayfirst=True)

    if "ano_nasc" in df.columns:
        df["ano_nasc"] = pd.to_numeric(df["ano_nasc"], errors="coerce").astype("Int64")
    else:
        df["ano_nasc"] = pd.NA

    if "data_nasc" in df.columns:
        mask = df["data_nasc"].notna()
        df.loc[mask, "ano_nasc"] = df.loc[mask, "data_nasc"].dt.year.astype("Int64")

    if "ano_nasc" in df.columns and not df["ano_nasc"].isna().all():
        ref_year = df["ano_base"].astype(int)
        idade = (ref_year - df["ano_nasc"]).astype("Int64")
        df["idade"] = idade.where((idade >= 0) & (idade <= 120), pd.NA)
    else:
        df["idade"] = pd.NA

    other_num_cols = [
        c for c in ["defasagem", "matem", "portug", "ingles", "ieg", "iaa", "ips", "ipp", "ano_ingresso", "no_av"]
        if c in df.columns
    ]
    for c in other_num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for col, fn in [
        ("turma", lambda s: s.astype(str).str.strip().replace({"NAN": np.nan, "nan": np.nan, "": np.nan})),
        ("genero", lambda s: s.astype(str).str.strip().str.upper().replace({"NAN": np.nan, "nan": np.nan, "": np.nan})),
        ("instituicao", lambda s: s.astype(str).str.strip().replace({"NAN": np.nan, "nan": np.nan, "": np.nan})),
    ]:
        if col in df.columns:
            df[col] = fn(df[col])

    front = [c for c in must_have + ["fase", "data_nasc", "ano_nasc", "idade", "no_av"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in front]
    return df[front + other_cols]


def standardize_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aplica apply_canonical_renames + coerce_types_and_clean num único passo."""
    df = apply_canonical_renames(df)
    df = coerce_types_and_clean(df, year=year)
    return df


def _first_series(df: pd.DataFrame, col: str) -> "pd.Series":
    """Retorna a primeira ocorrência da coluna como Series (evita erro com colunas duplicadas)."""
    col_data = df[col]
    if isinstance(col_data, pd.DataFrame):
        return col_data.iloc[:, 0]
    return col_data


def ensure_inde_pedra(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai as colunas inde e pedra do próprio ano base do DataFrame."""
    df = df.copy()
    if "ano_base" not in df.columns or len(df) == 0:
        df["inde"] = pd.NA
        df["pedra"] = pd.NA
        return df

    ano = df["ano_base"].iloc[0]
    if pd.isna(ano):
        df["inde"] = pd.NA
        df["pedra"] = pd.NA
        return df

    ano = int(ano)
    ano2 = str(ano)[-2:]

    inde_cols = [f"inde_{ano}", f"inde_{ano2}"]
    pedra_cols = [f"pedra_{ano}", f"pedra_{ano2}"]

    inde_col = next((c for c in inde_cols if c in df.columns), None)
    pedra_col = next((c for c in pedra_cols if c in df.columns), None)

    df["inde"] = _first_series(df, inde_col) if inde_col else pd.NA
    df["pedra"] = _first_series(df, pedra_col) if pedra_col else pd.NA
    return df


def drop_only_annual_inde_pedra(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas do tipo inde_YYYY / inde_YY / pedra_YYYY / pedra_YY."""
    df = df.copy()
    drop_cols = [c for c in df.columns if re.fullmatch(r"(inde|pedra)_(\d{2}|\d{4})", c)]
    return df.drop(columns=drop_cols, errors="ignore")


_GENERO_MAP = {
    "menino": "M",
    "masculino": "M",
    "menina": "F",
    "feminino": "F",
    "m": "M",
    "f": "F",
}

_PEDRA_VALID = {
    "quartzo": "quartzo",
    "ametista": "ametista",
    "topazio": "topazio",
    "agata": "agata",
    "incluir": "sem_pedra_fase_8_9",
}


_normalize_text = normalize_colname  # alias — delegates to utils

# _normalize_fase_value unified into _parse_fase_int above
_normalize_fase_value = _parse_fase_int


def normalize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza genero → M/F, pedra → valor canônico, fase → Int64."""
    df = df.copy()

    if "genero" in df.columns:
        def _norm_gen(x):
            if pd.isna(x):
                return pd.NA
            return _GENERO_MAP.get(_normalize_text(str(x)), pd.NA)
        df["genero"] = df["genero"].apply(_norm_gen)

    if "pedra" in df.columns:
        def _norm_pedra(x):
            if pd.isna(x):
                return pd.NA
            s = _normalize_text(str(x))
            return _PEDRA_VALID.get(s, pd.NA) if s else pd.NA
        df["pedra"] = df["pedra"].apply(_norm_pedra)

    if "fase" in df.columns:
        df["fase"] = df["fase"].apply(_parse_fase_int).astype("Int64")

    return df


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Força dtypes canônicos: Int64 para inteiros, float64 para scores, category para genero/turma."""
    df = df.copy()

    if "ra" in df.columns:
        df["ra"] = df["ra"].astype("string").str.strip()

    if "ano_nasc" in df.columns:
        num = pd.to_numeric(df["ano_nasc"], errors="coerce")
        df["ano_nasc"] = num.apply(lambda x: round(x) if pd.notna(x) else pd.NA).astype("Int64")
        if "data_nasc" in df.columns and pd.api.types.is_datetime64_any_dtype(df["data_nasc"]):
            df["ano_nasc"] = df["ano_nasc"].fillna(df["data_nasc"].dt.year.astype("Int64"))

    for col in ["idade", "ano_ingresso", "fase", "no_av", "defasagem"]:
        if col in df.columns:
            num = pd.to_numeric(df[col], errors="coerce")
            df[col] = num.apply(lambda x: round(x) if pd.notna(x) else pd.NA).astype("Int64")

    for col in ["ieg", "iaa", "ips", "ipp", "matem", "portug", "ingles"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["genero", "turma"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
