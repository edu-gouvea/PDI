"""Máscara de aguçamento, realce por Laplaciano e gradiente de Sobel (I3).

Especificações (nome, parâmetros, limites) propostas pelo I1 como stubs; o I3 pode
ajustar limites e textos. Ao implementar, substitua o corpo das funções e documente
na ``descricao`` a política de volta para uint8 (clip ou min-max, ver core/util.py).
"""

from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.registro import registrar_processo
from pdi.processos._stub import resultado_stub


def _janela(maximo: int, ajuda: str) -> Param:
    return Param(
        "janela",
        "Tamanho da janela",
        TipoParam.INT,
        3,
        minimo=3,
        maximo=maximo,
        passo=2,
        somente_impar=True,
        ajuda=ajuda,
    )


@registrar_processo(
    id="mascara_agucamento",
    nome="Máscara de aguçamento",
    grupo="Espacial",
    responsavel=3,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        Param(
            "ganho",
            "Ganho k",
            TipoParam.FLOAT,
            1.0,
            minimo=0.0,
            maximo=10.0,
            passo=0.1,
            ajuda="k = 1: unsharp masking; k > 1: highboost (realce mais forte).",
        ),
        _janela(
            31,
            "Janela da suavização usada para obter a máscara. Maior = realça bordas mais largas.",
        ),
    ),
    descricao="g = f + k·(f − f_suavizada). Realça bordas e detalhes finos.",
)
def mascara_agucamento(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 3)  # TODO(I3)


@registrar_processo(
    id="realce_laplaciano",
    nome="Realce por Laplaciano",
    grupo="Espacial",
    responsavel=3,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(_janela(15, "3 = kernel Laplaciano clássico; maiores = máscaras estendidas."),),
    descricao="g = f − ∇²f. Mostra a imagem realçada e o Laplaciano ajustado (min-max).",
)
def realce_laplaciano(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 3)  # TODO(I3): incluir o Laplaciano ajustado nos acessórios


@registrar_processo(
    id="gradiente_sobel",
    nome="Gradiente de Sobel",
    grupo="Espacial",
    responsavel=3,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(_janela(15, "3 = Sobel clássico; maiores = Sobel estendido."),),
    descricao="Magnitude do gradiente √(Gx² + Gy²) com kernels de Sobel. Destaca bordas.",
)
def gradiente_sobel(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 3)  # TODO(I3)
