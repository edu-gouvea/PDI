"""Suavização espacial em NumPy, por canal e com bordas refletidas (I2)."""

from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.espacial import convoluir, iterar_vizinhancas
from pdi.core.registro import registrar_processo
from pdi.core.util import aplicar_por_canal, clip_para_uint8


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
    descricao=(
        "Média pelo kernel exp(−(x²+y²)/2σ²) normalizado, em duas passagens separáveis. "
        "Por canal, bordas refletidas sem repetir a extremidade; arredondamento e clip [0, 255]."
    ),
)
def filtro_gaussiano(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    raio = p["janela"] // 2
    eixo = np.arange(-raio, raio + 1, dtype=np.float64)
    pesos = np.exp(-(eixo**2) / (2.0 * p["sigma"] ** 2))
    pesos /= pesos.sum()

    def suavizar(canal: np.ndarray) -> np.ndarray:
        horizontal = convoluir(canal, pesos[np.newaxis, :])
        return convoluir(horizontal, pesos[:, np.newaxis])

    saida = clip_para_uint8(aplicar_por_canal(suavizar, img))
    return Resultado(
        saidas=[("Suavizada (gaussiana)", saida)],
        info={"σ": p["sigma"], "janela": p["janela"]},
    )


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
    descricao=(
        "Substitui cada pixel pela mediana, mínimo ou máximo da vizinhança, por canal. "
        "Bordas refletidas sem repetir a extremidade. Seleção exata em uint8, sem reescala."
    ),
)
def filtro_ordem(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    reducao = {"mediana": np.median, "mínimo": np.min, "máximo": np.max}[p["tipo"]]

    def filtrar(canal: np.ndarray) -> np.ndarray:
        # Para janela ímpar, a mediana é um dos níveis uint8 de entrada. Não há
        # soma nem produto em uint8: todas as reduções selecionam níveis exatos.
        saida = np.empty_like(canal)
        for linhas, vizinhancas in iterar_vizinhancas(canal, p["janela"]):
            saida[linhas] = reducao(vizinhancas, axis=(-2, -1))
        return saida

    saida = aplicar_por_canal(filtrar, img)
    return Resultado(
        saidas=[(f"Filtro de {p['tipo']}", saida)],
        info={"tipo": p["tipo"], "janela": p["janela"]},
    )
