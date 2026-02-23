from pathlib import Path

import numpy as np
import pandas as pd


def find_col(columns: list[str], keywords: list[str]) -> str | None:
    for col in columns:
        s = str(col).lower()
        if any(k.lower() in s for k in keywords):
            return col
    return None


def main() -> None:
    rng = np.random.default_rng(42)
    source = Path("BASE DE DADOS PEDE 2024 - DATATHON.xlsx")
    target = Path("data/BASE DE DADOS PEDE 2025 - SINTETICA.xlsx")

    xl = pd.ExcelFile(source, engine="openpyxl")
    sheets = {name: xl.parse(name) for name in xl.sheet_names}
    if "PEDE2024" not in sheets:
        raise ValueError("A planilha de origem precisa conter a aba PEDE2024.")

    df24 = sheets["PEDE2024"]
    df25 = df24.copy()
    cols = list(df25.columns)

    col_def = find_col(cols, ["defas"])
    col_mat = find_col(cols, ["mat"])
    col_por = find_col(cols, ["por"])
    col_ing = find_col(cols, ["ing"])
    col_ieg = find_col(cols, ["ieg"])
    col_iaa = find_col(cols, ["iaa"])
    col_ips = find_col(cols, ["ips"])

    for col in [col_mat, col_por, col_ing, col_ieg, col_iaa, col_ips]:
        if col:
            df25[col] = pd.to_numeric(df25[col], errors="coerce")

    if col_def:
        d24 = pd.to_numeric(df24[col_def], errors="coerce").fillna(0)
        risk = np.zeros(len(df25), dtype=float)
        if col_mat:
            risk += np.nan_to_num((10 - pd.to_numeric(df25[col_mat], errors="coerce")) / 10, nan=0)
        if col_por:
            risk += np.nan_to_num((10 - pd.to_numeric(df25[col_por], errors="coerce")) / 10, nan=0)
        if col_ieg:
            risk += np.nan_to_num((10 - pd.to_numeric(df25[col_ieg], errors="coerce")) / 10, nan=0)
        risk = np.clip(risk / 3, 0, 1)

        # If student was on-track in 2024, probability of entering defasagem in 2025
        p = np.where(d24 == 0, 0.10 + 0.35 * risk, 0.75)
        transition = rng.random(len(df25)) < p
        d25 = np.where(d24 == 0, transition.astype(int), np.maximum(1, d24.astype(int)))
        df25[col_def] = d25

    # Add small realistic score noise for 2025
    for col in [col_mat, col_por, col_ing, col_ieg, col_iaa, col_ips]:
        if not col:
            continue
        s = pd.to_numeric(df25[col], errors="coerce")
        n = rng.normal(0, 0.6, len(df25))
        df25[col] = np.clip(s + n, 0, 10).round(2)

    sheets["PEDE2025"] = df25
    target.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"Arquivo sintético criado: {target}")


if __name__ == "__main__":
    main()
