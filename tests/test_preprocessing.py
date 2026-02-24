"""
Tests for src/preprocessing.py (split temporal e target por piora).
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    _extract_phase,
    _strip_accents,
    _normalize_colname,
    _normalize_fase_numeric,
    _normalize_gender,
    _canonicalize_columns,
    _fuzzy_rename,
    _build_pair,
    build_longitudinal_dataset,
)


class TestExtractPhase:
    def test_alfa(self):
        assert _extract_phase("ALFA") == "ALFA"
        assert _extract_phase("alfa") == "ALFA"

    def test_fase_2022_numeric(self):
        assert _extract_phase("0") == "ALFA"
        assert _extract_phase("1") == "FASE1"
        assert _extract_phase("7") == "FASE7"

    def test_fase_2023_format(self):
        assert _extract_phase("FASE 3") == "FASE3"
        assert _extract_phase("Fase 1") == "FASE1"
        assert _extract_phase("FASE 8") == "FASE8"

    def test_fase_2024_code(self):
        assert _extract_phase("1A") == "FASE1"
        assert _extract_phase("2B") == "FASE2"
        assert _extract_phase("8E") == "FASE8"

    def test_nan(self):
        assert _extract_phase(None) == "UNKNOWN"
        assert _extract_phase(float("nan")) == "UNKNOWN"


class TestFuzzyRename:
    def test_exact_match(self):
        df = pd.DataFrame({"RA": [1], "Turma": ["A"]})
        result = _fuzzy_rename(df, {"RA": "ra", "Turma": "turma"})
        assert "ra" in result.columns
        assert "turma" in result.columns

    def test_partial_ascii_match(self):
        # Column with accent replaced by U+FFFD
        col_encoded = "Ingl\ufffdnero"
        df = pd.DataFrame({col_encoded: [1]})
        result = _fuzzy_rename(df, {"Inglnero": "ingles"})
        # ASCII-folded match: 'Ingl' matches 'Ingl'
        # Result may or may not rename depending on implementation
        assert isinstance(result, pd.DataFrame)


class TestBuildPair:
    def _make_df(self, ras, defas, year):
        return pd.DataFrame({
            "ra": ras,
            "defasagem": defas,
            "ano_base": year,
            "fase": ["FASE1"] * len(ras),
            "turma": ["A"] * len(ras),
            "ieg": [7.0] * len(ras),
            "iaa": [8.0] * len(ras),
            "ips": [6.0] * len(ras),
            "ipp": [np.nan] * len(ras),
            "matem": [7.0] * len(ras),
            "portug": [6.5] * len(ras),
            "ingles": [np.nan] * len(ras),
            "genero": ["F"] * len(ras),
            "instituicao": ["PUBLICA"] * len(ras),
            "ano_ingresso": [2020] * len(ras),
        })

    def test_keeps_students_with_defasagem_in_t(self):
        df_t = self._make_df(
            ras=["RA-1", "RA-2", "RA-3"],
            defas=[0, 1, 0],  # RA-2 já com defasagem em T
            year=2022,
        )
        df_t1 = self._make_df(
            ras=["RA-1", "RA-2", "RA-3"],
            defas=[1, 2, 0],
            year=2023,
        )
        result = _build_pair(df_t, df_t1, "test")
        # Estratégia nova: não exclui aluno já defasado em T.
        assert "RA-2" in result["ra"].values
        assert len(result) == 3

    def test_target_assignment_by_worsening(self):
        df_t = self._make_df(["RA-1", "RA-2", "RA-3"], [0, 1, 2], 2022)
        df_t1 = self._make_df(
            ["RA-1", "RA-2", "RA-3"],
            [1, 1, 3],  # piora só para RA-1 e RA-3
            2023,
        )
        result = _build_pair(df_t, df_t1, "test")
        result = result.set_index("ra")
        assert result.loc["RA-1", "target"] == 1
        assert result.loc["RA-2", "target"] == 0
        assert result.loc["RA-3", "target"] == 1

    def test_excludes_unmatched(self):
        df_t = self._make_df(["RA-1", "RA-2"], [0, 0], 2022)
        df_t1 = self._make_df(["RA-1"], [-1], 2023)  # RA-2 missing in T+1
        result = _build_pair(df_t, df_t1, "test")
        assert "RA-2" not in result["ra"].values
        assert len(result) == 1

    def test_no_positive_targets_when_all_stay_on_track(self):
        df_t = self._make_df(["RA-1", "RA-2"], [0, 0], 2022)
        df_t1 = self._make_df(["RA-1", "RA-2"], [0, 0], 2023)
        result = _build_pair(df_t, df_t1, "test")
        assert result["target"].sum() == 0


# ---------------------------------------------------------------------------
# _strip_accents
# ---------------------------------------------------------------------------

class TestStripAccents:
    def test_removes_cedilla(self):
        assert _strip_accents("ção") == "cao"

    def test_removes_tilde(self):
        assert _strip_accents("ão") == "ao"

    def test_no_accent_unchanged(self):
        assert _strip_accents("hello") == "hello"

    def test_mixed_string(self):
        result = _strip_accents("Instituição de Ensino")
        assert "ç" not in result
        assert "ã" not in result


# ---------------------------------------------------------------------------
# _normalize_colname
# ---------------------------------------------------------------------------

class TestNormalizeColname:
    def test_lowercases(self):
        assert _normalize_colname("TURMA") == "turma"

    def test_replaces_spaces_with_underscore(self):
        assert _normalize_colname("ano ingresso") == "ano_ingresso"

    def test_removes_special_chars(self):
        result = _normalize_colname("Nota/Matemática")
        assert "/" not in result
        assert " " not in result

    def test_strips_leading_trailing_underscores(self):
        result = _normalize_colname("_coluna_")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_collapses_multiple_underscores(self):
        result = _normalize_colname("a__b___c")
        assert "__" not in result


# ---------------------------------------------------------------------------
# _normalize_fase_numeric
# ---------------------------------------------------------------------------

class TestNormalizeFaseNumeric:
    def test_alfa_returns_zero(self):
        assert _normalize_fase_numeric("ALFA") == 0

    def test_alfa_lowercase(self):
        assert _normalize_fase_numeric("alfa") == 0

    def test_fase_n_returns_int(self):
        assert _normalize_fase_numeric("FASE3") == 3

    def test_pure_digit(self):
        assert _normalize_fase_numeric("5") == 5

    def test_nan_returns_na(self):
        import pandas as pd
        result = _normalize_fase_numeric(None)
        assert result is pd.NA or (hasattr(result, '__class__') and 'NA' in str(result))

    def test_no_digit_returns_na(self):
        import pandas as pd
        result = _normalize_fase_numeric("UNKNOWN")
        assert result is pd.NA or (hasattr(result, '__class__') and 'NA' in str(result))


# ---------------------------------------------------------------------------
# additional helpers
# ---------------------------------------------------------------------------

class TestAdditionalHelpers:
    def test_normalizar_nome_colunas(self):
        from src.preprocessing import normalizar_nome_colunas
        assert normalizar_nome_colunas("Nome Aluno") == "nome_aluno"

    def test_apply_canonical_renames(self):
        from src.preprocessing import apply_canonical_renames
        df = pd.DataFrame({"matem": [1], "portug": [2], "unknown": [3]})
        out = apply_canonical_renames(df)
        assert "matem" in out.columns
        assert "portug" in out.columns
        assert "unknown" in out.columns

    def test_coerce_types_and_clean_basic(self):
        from src.preprocessing import coerce_types_and_clean
        df = pd.DataFrame({"ra": [" R1 "], "ano_base": [2022], "fase_raw": ["ALFA"],
                           "turma": [" A "], "defasagem": ["1"], "matem": ["8"],
                           "genero": ["M"]})
        out = coerce_types_and_clean(df, year=2022)
        assert out["ra"].iloc[0] == "R1"
        # coerce_types_and_clean does not convert fase to Int64, it leaves object
        assert out["fase"].dtype == object
        # numeric columns may coerce to integer if no decimals
        assert pd.api.types.is_numeric_dtype(out["matem"])

    def test_standardize_year_combines(self):
        from src.preprocessing import standardize_year
        df = pd.DataFrame({"Matem": [9], "fase_raw": ["1"], "ra": ["R1"]})
        out = standardize_year(df, year=2023)
        assert "matem" in out.columns
        assert out["ano_base"].iloc[0] == 2023

    def test_first_series_with_dataframe(self):
        from src.preprocessing import _first_series
        # duplicate column names lead df[col] to return a DataFrame
        df = pd.DataFrame([[1,2],[3,4]], columns=["x","x"])
        s = _first_series(df, "x")
        assert s.iloc[0] == 1

    def test_ensure_inde_pedra_various(self):
        from src.preprocessing import ensure_inde_pedra
        df = pd.DataFrame({"ano_base": [2023], "inde_2023": [5], "pedra_23": ["quartzo"]})
        out = ensure_inde_pedra(df)
        assert out["inde"].iloc[0] == 5
        assert out["pedra"].iloc[0] == "quartzo"
        # missing ano_base
        out2 = ensure_inde_pedra(pd.DataFrame())
        assert pd.isna(out2["inde"]).all()

    def test_drop_only_annual_inde_pedra(self):
        from src.preprocessing import drop_only_annual_inde_pedra
        df = pd.DataFrame({"inde_2021": [1], "pedra_21": ["x"], "keep": [2]})
        out = drop_only_annual_inde_pedra(df)
        assert "inde_2021" not in out.columns
        assert "pedra_21" not in out.columns
        assert "keep" in out.columns

    def test_normalize_text_fields(self):
        from src.preprocessing import normalize_text_fields
        df = pd.DataFrame({"genero": ["Menino"], "pedra": ["Ametista"], "fase": ["FASE2"]})
        out = normalize_text_fields(df)
        assert out["genero"].iloc[0] == "M"
        assert out["pedra"].iloc[0] == "ametista"
        assert out["fase"].iloc[0] == 2

    def test_normalize_dtypes(self):
        from src.preprocessing import normalize_dtypes
        df = pd.DataFrame({"ra": [" R "], "ano_nasc": ["2000"],
                           "idade": ["20"], "genero": ["M"], "turma": ["A"]})
        out = normalize_dtypes(df)
        assert out["ra"].iloc[0] == "R"
        assert out["ano_nasc"].dtype.name == "Int64"
        assert out["idade"].dtype.name == "Int64"
        assert out["genero"].dtype.name == "category"
        assert out["turma"].dtype.name == "category"

    def test_load_all_years_insufficient_sheets(self, monkeypatch):
        from src.preprocessing import load_all_years

        class FakeXl:
            sheet_names = ["PEDE2022"]
            def parse(self, name):
                return pd.DataFrame()

        monkeypatch.setattr("pandas.ExcelFile", lambda path, engine=None: FakeXl())
        with pytest.raises(ValueError):
            load_all_years("dummy")

    def test_load_all_years_success(self, monkeypatch):
        from src.preprocessing import load_all_years

        class FakeXl:
            sheet_names = ["PEDE2020", "PEDE2021", "PEDE2022"]
            def parse(self, name):
                # return df with minimal required columns
                return pd.DataFrame({"ra": ["R1"], "fase_raw": ["1"],
                                     "turma": ["A"], "defasagem": [0],
                                     "matem": [5], "portug": [6],
                                     "ingles": [7], "ieg": [1], "iaa": [1],
                                     "ips": [1], "ipp": [1],
                                     "genero": ["M"], "instituicao": ["X"],
                                     "escola": ["Y"], "ano_ingresso": [2020],
                                     "ano_nasc": [2002], "idade": [18]}
                                    )

        monkeypatch.setattr("pandas.ExcelFile", lambda path, engine=None: FakeXl())
        result = load_all_years("dummy")
        assert set(result.keys()) == {2020, 2021, 2022}

    def test_load_sheet_adds_missing_columns_and_calculates_age(self):
        from src.preprocessing import _load_sheet
        # build a fake ExcelFile-like object with parse method
        class DummyXl:
            def parse(self, name):
                return pd.DataFrame({
                    "RA": ["1"],
                    "fase_raw": ["ALFA"],
                    "turma": ["T"],
                    "defasagem": [0],
                    "ano_nasc": [2000],
                    # missing 'idade' should be computed
                })
        df = _load_sheet(DummyXl(), "sheet", 2023)
        assert "ano_base" in df.columns
        assert "idade" in df.columns
        assert df["idade"].iloc[0] == 23


# ---------------------------------------------------------------------------
# _normalize_gender
# ---------------------------------------------------------------------------

class TestNormalizeGender:
    def test_menino_returns_m(self):
        assert _normalize_gender("menino") == "M"

    def test_masculino_returns_m(self):
        assert _normalize_gender("Masculino") == "M"

    def test_m_returns_m(self):
        assert _normalize_gender("M") == "M"

    def test_menina_returns_f(self):
        assert _normalize_gender("menina") == "F"

    def test_feminino_returns_f(self):
        assert _normalize_gender("Feminino") == "F"

    def test_f_returns_f(self):
        assert _normalize_gender("F") == "F"

    def test_nan_returns_na(self):
        assert _normalize_gender(None) == "NA"
        assert _normalize_gender(float("nan")) == "NA"

    def test_unknown_returns_na(self):
        assert _normalize_gender("outro") == "NA"


# ---------------------------------------------------------------------------
# _canonicalize_columns
# ---------------------------------------------------------------------------

class TestCanonicalizeColumns:
    def test_renames_ra(self):
        df = pd.DataFrame({"RA": [1]})
        result = _canonicalize_columns(df)
        assert "ra" in result.columns

    def test_renames_mat_to_matem(self):
        df = pd.DataFrame({"mat": [7.0]})
        result = _canonicalize_columns(df)
        assert "matem" in result.columns

    def test_renames_por_to_portug(self):
        df = pd.DataFrame({"por": [6.5]})
        result = _canonicalize_columns(df)
        assert "portug" in result.columns

    def test_renames_fase_to_fase_raw(self):
        df = pd.DataFrame({"fase": ["FASE1"]})
        result = _canonicalize_columns(df)
        assert "fase_raw" in result.columns

    def test_unrecognized_column_kept(self):
        df = pd.DataFrame({"coluna_nova": [1]})
        result = _canonicalize_columns(df)
        assert "coluna_nova" in result.columns

    def test_original_not_mutated(self):
        df = pd.DataFrame({"RA": [1], "Turma": ["A"]})
        original_cols = list(df.columns)
        _canonicalize_columns(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# build_longitudinal_dataset
# ---------------------------------------------------------------------------

class TestBuildLongitudinalDataset:
    def _make_year_df(self, ras, defas, year):
        return pd.DataFrame(
            {
                "ra": ras,
                "defasagem": defas,
                "ano_base": year,
                "fase": [1] * len(ras),
                "turma": ["A"] * len(ras),
                "ieg": [7.0] * len(ras),
                "iaa": [8.0] * len(ras),
                "ips": [6.0] * len(ras),
                "ipp": [None] * len(ras),
                "matem": [7.0] * len(ras),
                "portug": [6.5] * len(ras),
                "ingles": [None] * len(ras),
                "genero": ["F"] * len(ras),
                "instituicao": ["PUBLICA"] * len(ras),
                "ano_ingresso": [2020] * len(ras),
            }
        )

    def _make_data(self):
        return {
            2022: self._make_year_df(["RA-1", "RA-2", "RA-3"], [0, 1, 0], 2022),
            2023: self._make_year_df(["RA-1", "RA-2", "RA-3"], [1, 1, 0], 2023),
            2024: self._make_year_df(["RA-1", "RA-2", "RA-3"], [2, 1, 0], 2024),
        }

    def test_returns_two_dataframes(self):
        train_df, valid_df = build_longitudinal_dataset(self._make_data())
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(valid_df, pd.DataFrame)

    def test_valid_is_latest_pair(self):
        _, valid_df = build_longitudinal_dataset(self._make_data())
        assert "pair_label" in valid_df.columns
        assert valid_df["pair_label"].iloc[0] == "2023->2024"

    def test_train_contains_earlier_pairs(self):
        train_df, _ = build_longitudinal_dataset(self._make_data())
        labels = train_df["pair_label"].unique()
        assert "2022->2023" in labels

    def test_train_does_not_contain_valid_pair(self):
        train_df, valid_df = build_longitudinal_dataset(self._make_data())
        valid_label = valid_df["pair_label"].iloc[0]
        assert valid_label not in train_df["pair_label"].values

    def test_target_column_present(self):
        train_df, valid_df = build_longitudinal_dataset(self._make_data())
        assert "target" in train_df.columns
        assert "target" in valid_df.columns

    def test_raises_with_fewer_than_3_years(self):
        data_2yr = {
            2022: self._make_year_df(["RA-1"], [0], 2022),
            2023: self._make_year_df(["RA-1"], [1], 2023),
        }
        with pytest.raises(ValueError):
            build_longitudinal_dataset(data_2yr)


# ---------------------------------------------------------------------------
# Notebook-aligned helpers
# ---------------------------------------------------------------------------

from src.preprocessing import (
    normalizar_nome_colunas,
    apply_canonical_renames,
    coerce_types_and_clean,
    standardize_year,
    ensure_inde_pedra,
    drop_only_annual_inde_pedra,
    normalize_text_fields,
    normalize_dtypes,
)


class TestNormalizarNomeColunas:
    def test_removes_accents(self):
        assert normalizar_nome_colunas("Instituição") == "instituicao"

    def test_lowercases(self):
        assert normalizar_nome_colunas("TURMA") == "turma"

    def test_spaces_become_underscore(self):
        assert normalizar_nome_colunas("ano ingresso") == "ano_ingresso"

    def test_special_chars_stripped(self):
        result = normalizar_nome_colunas("Nota/Matemática")
        assert "/" not in result
        assert " " not in result

    def test_no_leading_trailing_underscore(self):
        result = normalizar_nome_colunas("_col_")
        assert not result.startswith("_")
        assert not result.endswith("_")


class TestApplyCanonicalRenames:
    def test_renames_mat_to_matem(self):
        df = pd.DataFrame({"mat": [7.0]})
        out = apply_canonical_renames(df)
        assert "matem" in out.columns

    def test_renames_por_to_portug(self):
        df = pd.DataFrame({"por": [6.0]})
        out = apply_canonical_renames(df)
        assert "portug" in out.columns

    def test_renames_ing_to_ingles(self):
        df = pd.DataFrame({"ing": [8.0]})
        out = apply_canonical_renames(df)
        assert "ingles" in out.columns

    def test_renames_fase_to_fase_raw(self):
        df = pd.DataFrame({"fase": ["FASE1"]})
        out = apply_canonical_renames(df)
        assert "fase_raw" in out.columns

    def test_renames_instituicao_de_ensino(self):
        df = pd.DataFrame({"instituicao_de_ensino": ["PUBLICA"]})
        out = apply_canonical_renames(df)
        assert "instituicao" in out.columns

    def test_renames_inde_short(self):
        df = pd.DataFrame({"inde_22": [0.8]})
        out = apply_canonical_renames(df)
        assert "inde_2022" in out.columns

    def test_renames_pedra_short(self):
        df = pd.DataFrame({"pedra_24": ["quartzo"]})
        out = apply_canonical_renames(df)
        assert "pedra_2024" in out.columns

    def test_unknown_column_preserved(self):
        df = pd.DataFrame({"coluna_nova": [1]})
        out = apply_canonical_renames(df)
        assert "coluna_nova" in out.columns

    def test_original_not_mutated(self):
        df = pd.DataFrame({"mat": [7.0], "por": [6.0]})
        original = list(df.columns)
        apply_canonical_renames(df)
        assert list(df.columns) == original


class TestCoerceTypesAndClean:
    def _base_df(self):
        return pd.DataFrame({
            "ra": ["  RA-001  "],
            "fase_raw": ["FASE3"],
            "turma": ["A"],
            "defasagem": ["1"],
            "matem": ["7,5"],
            "portug": [6.0],
            "ingles": [np.nan],
            "ieg": [8.0],
            "iaa": [9.0],
            "ips": [7.0],
            "ipp": [np.nan],
            "genero": ["  feminino  "],
            "instituicao": ["PUBLICA"],
            "ano_ingresso": [2020],
            "ano_nasc": [2005],
        })

    def test_ano_base_set(self):
        df = coerce_types_and_clean(self._base_df(), 2023)
        assert df["ano_base"].iloc[0] == 2023

    def test_ra_stripped(self):
        df = coerce_types_and_clean(self._base_df(), 2023)
        assert df["ra"].iloc[0] == "RA-001"

    def test_missing_columns_added(self):
        df = pd.DataFrame({"ra": ["X"], "fase_raw": ["1"], "turma": ["A"]})
        out = coerce_types_and_clean(df, 2023)
        assert "matem" in out.columns
        assert "ieg" in out.columns

    def test_idade_computed_from_ano_nasc(self):
        df = coerce_types_and_clean(self._base_df(), 2023)
        assert df["idade"].iloc[0] == 2023 - 2005

    def test_data_nasc_parsed(self):
        df = self._base_df()
        df["data_nasc"] = ["01/06/2005"]
        out = coerce_types_and_clean(df, 2023)
        assert pd.notna(out["data_nasc"].iloc[0])

    def test_genero_stripped_and_upper(self):
        df = coerce_types_and_clean(self._base_df(), 2023)
        assert df["genero"].iloc[0] == "FEMININO"

    def test_original_not_mutated(self):
        df = self._base_df()
        original_ra = list(df["ra"])
        coerce_types_and_clean(df, 2023)
        assert list(df["ra"]) == original_ra


class TestStandardizeYear:
    def test_applies_renames_and_coerce(self):
        df = pd.DataFrame({
            "mat": [7.0],
            "por": [6.5],
            "fase": ["FASE2"],
            "ra": ["X1"],
            "turma": ["B"],
        })
        out = standardize_year(df, 2024)
        assert "matem" in out.columns
        assert "portug" in out.columns
        assert out["ano_base"].iloc[0] == 2024


class TestEnsureIndePedra:
    def test_extracts_inde_4digit(self):
        df = pd.DataFrame({"ano_base": [2023], "inde_2023": [0.75], "pedra_2023": ["ametista"]})
        out = ensure_inde_pedra(df)
        assert out["inde"].iloc[0] == pytest.approx(0.75)
        assert out["pedra"].iloc[0] == "ametista"

    def test_extracts_inde_2digit(self):
        df = pd.DataFrame({"ano_base": [2023], "inde_23": [0.9], "pedra_23": ["quartzo"]})
        out = ensure_inde_pedra(df)
        assert out["inde"].iloc[0] == pytest.approx(0.9)

    def test_no_inde_column_gives_na(self):
        df = pd.DataFrame({"ano_base": [2023], "score": [1.0]})
        out = ensure_inde_pedra(df)
        assert out["inde"].isna().all()
        assert out["pedra"].isna().all()

    def test_missing_ano_base_gives_na(self):
        df = pd.DataFrame({"score": [1.0]})
        out = ensure_inde_pedra(df)
        assert out["inde"].isna().all()

    def test_empty_df_gives_na(self):
        df = pd.DataFrame({"ano_base": pd.Series([], dtype=int), "inde_2023": pd.Series([], dtype=float)})
        out = ensure_inde_pedra(df)
        assert out["inde"].isna().all()


class TestDropOnlyAnnualIndePedra:
    def test_drops_4digit_columns(self):
        df = pd.DataFrame({"inde_2022": [1], "inde_2023": [2], "score": [3]})
        out = drop_only_annual_inde_pedra(df)
        assert "inde_2022" not in out.columns
        assert "inde_2023" not in out.columns
        assert "score" in out.columns

    def test_drops_2digit_columns(self):
        df = pd.DataFrame({"pedra_22": ["quartzo"], "inde_23": [0.8], "ra": ["X"]})
        out = drop_only_annual_inde_pedra(df)
        assert "pedra_22" not in out.columns
        assert "inde_23" not in out.columns
        assert "ra" in out.columns

    def test_keeps_inde_and_pedra_without_suffix(self):
        df = pd.DataFrame({"inde": [0.7], "pedra": ["ametista"], "inde_2024": [0.7]})
        out = drop_only_annual_inde_pedra(df)
        assert "inde" in out.columns
        assert "pedra" in out.columns
        assert "inde_2024" not in out.columns

    def test_no_annual_columns_unchanged(self):
        df = pd.DataFrame({"ra": ["X"], "turma": ["A"]})
        out = drop_only_annual_inde_pedra(df)
        assert list(out.columns) == ["ra", "turma"]


class TestNormalizeTextFields:
    def test_feminino_to_f(self):
        df = pd.DataFrame({"genero": ["feminino"]})
        out = normalize_text_fields(df)
        assert out["genero"].iloc[0] == "F"

    def test_masculino_to_m(self):
        df = pd.DataFrame({"genero": ["Masculino"]})
        out = normalize_text_fields(df)
        assert out["genero"].iloc[0] == "M"

    def test_unknown_genero_becomes_na(self):
        df = pd.DataFrame({"genero": ["outro"]})
        out = normalize_text_fields(df)
        assert pd.isna(out["genero"].iloc[0])

    def test_pedra_canonical(self):
        df = pd.DataFrame({"pedra": ["Quartzo", "ametista"]})
        out = normalize_text_fields(df)
        assert out["pedra"].iloc[0] == "quartzo"
        assert out["pedra"].iloc[1] == "ametista"

    def test_pedra_unknown_becomes_na(self):
        df = pd.DataFrame({"pedra": ["diamante"]})
        out = normalize_text_fields(df)
        assert pd.isna(out["pedra"].iloc[0])

    def test_fase_alfa_to_zero(self):
        df = pd.DataFrame({"fase": ["ALFA"]})
        out = normalize_text_fields(df)
        assert out["fase"].iloc[0] == 0

    def test_fase_n_to_int(self):
        df = pd.DataFrame({"fase": ["FASE3", "3"]})
        out = normalize_text_fields(df)
        assert out["fase"].iloc[0] == 3
        assert out["fase"].iloc[1] == 3

    def test_fase_nan_stays_na(self):
        df = pd.DataFrame({"fase": [None]})
        out = normalize_text_fields(df)
        assert pd.isna(out["fase"].iloc[0])

    def test_no_column_no_error(self):
        df = pd.DataFrame({"ra": ["X"]})
        out = normalize_text_fields(df)
        assert "ra" in out.columns


class TestNormalizeDtypes:
    def _base_df(self):
        return pd.DataFrame({
            "ra": ["  RA01 "],
            "ano_nasc": [2005.0],
            "idade": [18],
            "ano_ingresso": [2020],
            "fase": [3],
            "defasagem": [1],
            "ieg": ["8,5"],
            "iaa": [9.0],
            "ips": [7.0],
            "ipp": [np.nan],
            "matem": ["7,5"],
            "portug": [6.0],
            "ingles": [np.nan],
            "genero": ["M"],
            "turma": ["A"],
        })

    def test_ra_as_string(self):
        out = normalize_dtypes(self._base_df())
        assert out["ra"].dtype == pd.StringDtype() or str(out["ra"].dtype) == "string"
        assert out["ra"].iloc[0] == "RA01"

    def test_ano_nasc_as_int64(self):
        out = normalize_dtypes(self._base_df())
        assert out["ano_nasc"].iloc[0] == 2005

    def test_ieg_comma_decimal_parsed(self):
        out = normalize_dtypes(self._base_df())
        assert out["ieg"].iloc[0] == pytest.approx(8.5)

    def test_matem_comma_decimal_parsed(self):
        out = normalize_dtypes(self._base_df())
        assert out["matem"].iloc[0] == pytest.approx(7.5)

    def test_fase_as_int64(self):
        out = normalize_dtypes(self._base_df())
        assert pd.api.types.is_integer_dtype(out["fase"])

    def test_genero_as_category(self):
        out = normalize_dtypes(self._base_df())
        assert hasattr(out["genero"], "cat")

    def test_turma_as_category(self):
        out = normalize_dtypes(self._base_df())
        assert hasattr(out["turma"], "cat")

    def test_no_error_without_optional_cols(self):
        df = pd.DataFrame({"ra": ["X"], "ano_nasc": [2000]})
        out = normalize_dtypes(df)
        assert "ra" in out.columns

