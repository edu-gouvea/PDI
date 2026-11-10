#!/usr/bin/env bash
# Abre o programa com um comando: ./run.sh [imagem.png]
#
# Na primeira vez: cria o ambiente virtual (.venv) e instala as dependências.
# Nas seguintes: abre direto (só reinstala se o requirements.txt mudar).
set -euo pipefail
cd "$(dirname "$0")"

falhar() {
    echo >&2
    echo "ERRO: $1" >&2
    exit 1
}

# 1) Python 3.11 ou mais novo
PYTHON=""
for candidato in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidato" >/dev/null 2>&1 &&
        "$candidato" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
        PYTHON="$candidato"
        break
    fi
done
[ -n "$PYTHON" ] || falhar "Python 3.11 ou mais novo não encontrado.
  Ubuntu/Debian: sudo apt install python3
  Outros: https://www.python.org/downloads/"

# 2) Ambiente virtual
if [ ! -x .venv/bin/python ]; then
    echo "Criando o ambiente virtual em .venv (só na primeira vez)..."
    "$PYTHON" -m venv .venv 2>/dev/null || {
        rm -rf .venv
        falhar "Não foi possível criar o ambiente virtual.
  Ubuntu/Debian: sudo apt install python3-venv   (e rode ./run.sh de novo)"
    }
fi

# 3) Dependências (só quando o requirements.txt mudou)
if ! .venv/bin/python scripts/verificar_requisitos.py; then
    echo "Instalando as dependências (pode levar alguns minutos na primeira vez)..."
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements.txt ||
        falhar "A instalação das dependências falhou. Confira a conexão com a internet."
    .venv/bin/python scripts/verificar_requisitos.py --marcar
fi

# 4) Abre o programa (python -m pdi já evita o crash do tema GTK no GNOME 50)
exec .venv/bin/python -m pdi "$@"
