"""Decomposição RGB e decomposição HSV (I1)."""

from typing import Any

import numpy as np

from pdi.core.contrato import Resultado
from pdi.core.registro import registrar_processo
from pdi.processos._stub import resultado_stub


@registrar_processo(
    id="decomposicao_rgb",
    nome="Decomposição RGB",
    grupo="Cor",
    responsavel=1,
    aceita_cinza=False,
    aceita_rgb=True,
    descricao="Separa a imagem em três imagens em cinza, uma por canal R, G e B.",
)
def decomposicao_rgb(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 1)  # TODO(I1): Etapa 3


@registrar_processo(
    id="decomposicao_hsv",
    nome="Decomposição HSV",
    grupo="Cor",
    responsavel=1,
    aceita_cinza=False,
    aceita_rgb=True,
    descricao=(
        "Converte para HSV e mostra matiz (H), saturação (S) e valor (V) como três "
        "imagens em cinza."
    ),
)
def decomposicao_hsv(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 1)  # TODO(I1): Etapa 3
