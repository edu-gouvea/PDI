"""Realce espacial por canal, com bordas refletidas sem repetir a extremidade (I3)."""

from math import comb
from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.espacial import convoluir, correlacionar
from pdi.core.registro import registrar_processo
from pdi.core.util import aplicar_por_canal, clip_para_uint8, normalizar_para_uint8


def _binomial(ordem: int) -> np.ndarray:
    return np.array([comb(ordem, k) for k in range(ordem + 1)], dtype=np.float64)


def _kernel_laplaciano(janela: int) -> np.ndarray:
    """Laplaciano de quatro vizinhos convolvido com suavização binomial n−2.

    n=3 devolve [[0,1,0],[1,-4,1],[0,1,0]]. Nas janelas maiores, a soma dos
    pesos de suavização é 1 e a soma do kernel final é 0 (constantes são anuladas).
    """
    vetor = _binomial(janela - 3) / 2.0 ** (janela - 3)
    suave = np.outer(vetor, vetor)
    kernel = np.zeros((janela, janela), dtype=np.float64)
    lado = janela - 2
    for y, x, peso in ((0, 1, 1), (1, 0, 1), (1, 1, -4), (1, 2, 1), (2, 1, 1)):
        kernel[y : y + lado, x : x + lado] += peso * suave
    return kernel


def _kernels_sobel(janela: int) -> tuple[np.ndarray, np.ndarray]:
    """Sobel separável estendido: binomial n−1 × derivada de binomial n−3.

    Para n=3, a suavização é [1,2,1] e a derivada é [-1,0,1]. Não normaliza
    os kernels: a magnitude final é normalizada globalmente para exibição.
    """
    suave = _binomial(janela - 1)
    derivada = np.convolve(_binomial(janela - 3), [-1.0, 0.0, 1.0])
    horizontal = np.outer(suave, derivada)
    return horizontal, horizontal.T.copy()


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
    descricao=(
        "g = f + k·(f − média uniforme da janela). Por canal, bordas refletidas; "
        "arredondamento e clip [0, 255] preservam o ganho k."
    ),
)
def mascara_agucamento(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    kernel = np.full((p["janela"], p["janela"]), 1.0 / p["janela"] ** 2)
    suave = aplicar_por_canal(convoluir, img, kernel)
    original = img.astype(np.float64)
    saida = clip_para_uint8(original + p["ganho"] * (original - suave))
    return Resultado(
        saidas=[("Aguçada", saida)], info={"ganho k": p["ganho"], "janela": p["janela"]}
    )


@registrar_processo(
    id="realce_laplaciano",
    nome="Realce por Laplaciano",
    grupo="Espacial",
    responsavel=3,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        _janela(
            15, "3 = Laplaciano de quatro vizinhos; maiores = suavização binomial + Laplaciano."
        ),
    ),
    descricao=(
        "g = f − ∇²f, com clip [0, 255]. Janela 3: quatro vizinhos; maiores: Laplaciano "
        "suavizado por binomial. Por canal, bordas refletidas. Acessório ajustado por min-max."
    ),
)
def realce_laplaciano(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    laplaciano = aplicar_por_canal(convoluir, img, _kernel_laplaciano(p["janela"]))
    saida = clip_para_uint8(img.astype(np.float64) - laplaciano)
    return Resultado(
        saidas=[("Realçada (Laplaciano)", saida)],
        acessorios=[("Laplaciano ajustado", normalizar_para_uint8(laplaciano))],
        info={"janela": p["janela"]},
    )


@registrar_processo(
    id="gradiente_sobel",
    nome="Gradiente de Sobel",
    grupo="Espacial",
    responsavel=3,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(_janela(15, "3 = Sobel clássico; maiores = Sobel estendido."),),
    descricao=(
        "Magnitude √(Gx² + Gy²) do Sobel binomial estendido à janela escolhida. "
        "Por canal, bordas refletidas; normalização min-max global para uint8."
    ),
)
def gradiente_sobel(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    kx, ky = _kernels_sobel(p["janela"])
    gx = aplicar_por_canal(correlacionar, img, kx)
    gy = aplicar_por_canal(correlacionar, img, ky)
    saida = normalizar_para_uint8(np.hypot(gx, gy))
    return Resultado(saidas=[("Gradiente de Sobel", saida)], info={"janela": p["janela"]})
