"""Filtros espaciais: média gaussiana, mediana, mínimo e máximo (I2).

Especificações (nome, parâmetros, limites) propostas pelo I1 como stubs; o I2 pode
ajustar limites e textos. Ao implementar, substitua o corpo das funções e documente
na ``descricao`` a política de volta para uint8 (clip ou min-max, ver core/util.py).
"""

from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.registro import registrar_processo
from pdi.processos._stub import resultado_stub


def _janela(padrao: int) -> Param:
    return Param(
        "janela",
        "Tamanho da janela",
        TipoParam.INT,
        padrao,
        minimo=3,
        maximo=31,
        passo=2,
        somente_impar=True,
        ajuda="Lado da janela n×n (ímpar). Maior janela = efeito mais forte e mais lento.",
    )


@registrar_processo(
    id="filtro_gaussiano",
    nome="Filtro de média gaussiana",
    grupo="Espacial",
    responsavel=2,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        Param(
            "sigma",
            "Desvio padrão σ",
            TipoParam.FLOAT,
            1.0,
            minimo=0.1,
            maximo=20.0,
            passo=0.1,
            ajuda="Maior σ = mais desfoque. Regra prática: janela ≈ 6σ.",
        ),
        _janela(7),
    ),
    descricao="Média ponderada pelo kernel exp(−(x²+y²)/2σ²) normalizado. Suaviza a imagem.",
)
def filtro_gaussiano(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 2)  # TODO(I2)


@registrar_processo(
    id="filtro_ordem",
    nome="Filtros de mediana, mínimo e máximo",
    grupo="Espacial",
    responsavel=2,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        Param(
            "tipo",
            "Filtro",
            TipoParam.ESCOLHA,
            "mediana",
            opcoes=("mediana", "mínimo", "máximo"),
            ajuda="Mediana remove sal e pimenta; mínimo remove sal; máximo remove pimenta.",
        ),
        _janela(3),
    ),
    descricao="Substitui cada pixel pela mediana, mínimo ou máximo da sua vizinhança.",
)
def filtro_ordem(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 2)  # TODO(I2)
