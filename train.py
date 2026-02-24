"""
train.py  –  wrapper na raiz do projeto.

O dashboard (dashboard/dashapp.py) dispara retreino via subprocess
apontando para este arquivo:
    cmd = [sys.executable, str(ROOT_DIR / "train.py"), ...]

Este módulo apenas delega para src.train, que contém toda a lógica.
"""
from src.train import main

if __name__ == "__main__":
    main()
