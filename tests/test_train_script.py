"""Lightweight tests for the top-level ``train.py`` script.

The goal is not to re-execute the full pipeline (which needs real data)
but rather to exercise the control flow in ``train.main`` by stubbing
every external dependency.  This gives us coverage of the script
itself without performing long operations.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
import src

# the train script lives at project root; import it lazily inside tests


class DummyArgs:
    def __init__(self, xls="dummy.xlsx", model_dir="models", log_level="INFO"):
        self.xls = xls
        self.model_dir = model_dir
        self.log_level = log_level


def _dummy_df():
    return pd.DataFrame({"ra": ["A"], "defasagem": [0], "turma": ["T"]})


def test_main_runs_all_steps(monkeypatch, tmp_path):
    """Monkeypatch every expensive function and call ``main()``.

    We assert that the directories are created and that the various helpers
    were invoked (via simple side-effect flags).
    """
    called = {}

    # import module under test so we can patch its attributes directly
    import src.train as train_script
    # stub each imported function on the train_script namespace
    monkeypatch.setattr(train_script, "load_all_years", lambda path: {2022: _dummy_df(), 2023: _dummy_df(), 2024: _dummy_df()})
    monkeypatch.setattr(train_script, "build_longitudinal_dataset", lambda d: (_dummy_df(), _dummy_df()))
    monkeypatch.setattr(train_script, "build_features", lambda df, lookup_tables=None, is_train=None: (_dummy_df(), {}))
    monkeypatch.setattr(train_script, "get_feature_columns", lambda df: ([], []))
    monkeypatch.setattr(train_script, "train", lambda *args, **kwargs: ("model", "pool", "pool"))
    monkeypatch.setattr(train_script, "evaluate", lambda df, **kw: {"auc": 0.5})
    monkeypatch.setattr(train_script, "print_report", lambda res: "report")

    # capture save_json and save_pickle calls
    def fake_save_json(obj, path):
        called.setdefault("json", []).append(str(path))
    def fake_save_pickle(obj, path):
        called.setdefault("pickle", []).append(str(path))

    # patch both the utility module and train_script namespace
    monkeypatch.setattr("src.utils.save_json", fake_save_json)
    monkeypatch.setattr("src.utils.save_pickle", fake_save_pickle)
    monkeypatch.setattr(train_script, "save_json", fake_save_json)
    monkeypatch.setattr(train_script, "save_pickle", fake_save_pickle)

    # patch argparse to return dummy args
    import src.train as train_script
    monkeypatch.setattr(train_script, "parse_args", lambda: DummyArgs(model_dir=str(tmp_path / "m")))
    # stub predict_proba on the model_training module so that the local import inside
    # ``train.main`` picks it up
    import numpy as np
    monkeypatch.setattr("src.model_training.predict_proba", lambda *args, **kwargs: np.ones(len(args[1]) if len(args) > 1 else 1))

    # run main
    res = train_script.main()

    assert isinstance(res, dict)
    # check directories created
    assert (tmp_path / "m" / "model").exists()
    assert (tmp_path / "m" / "evaluation").exists()
    assert (tmp_path / "m" / "monitoring").exists()
    # ensure saving routines were called at least once
    assert "json" in called
    assert "pickle" in called


def test_main_requires_three_years(monkeypatch):
    """If ``load_all_years`` returns fewer than 3 sheets, script should abort."""
    monkeypatch.setattr(src.train, "load_all_years", lambda path: {2022: _dummy_df()})
    import src.train as train_script
    monkeypatch.setattr(train_script, "parse_args", lambda: DummyArgs())
    # also stub predict_proba to avoid entering catboost
    import numpy as np
    monkeypatch.setattr("src.model_training.predict_proba", lambda *args, **kwargs: np.ones(1))
    with pytest.raises(ValueError):
        train_script.main()
