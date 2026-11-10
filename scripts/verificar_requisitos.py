"""Diz se as dependências do venv estão em dia com o requirements.txt (usado por run.sh/run.bat).

Uso (a partir da raiz do projeto, com o Python do venv):
    python scripts/verificar_requisitos.py           → código 0 se em dia, 1 se precisa instalar
    python scripts/verificar_requisitos.py --marcar  → grava o carimbo depois de instalar

O carimbo é o SHA-256 do requirements.txt, guardado dentro do .venv. Se alguém do grupo
mudar as dependências, o carimbo deixa de bater e o run.sh/run.bat reinstala sozinho.
"""

import hashlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REQUISITOS = RAIZ / "requirements.txt"
CARIMBO = RAIZ / ".venv" / ".requisitos-instalados"


def carimbo_atual() -> str:
    return hashlib.sha256(REQUISITOS.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    if "--marcar" in argv:
        CARIMBO.write_text(carimbo_atual() + "\n", encoding="utf-8")
        return 0
    em_dia = CARIMBO.is_file() and CARIMBO.read_text(encoding="utf-8").strip() == carimbo_atual()
    return 0 if em_dia else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
