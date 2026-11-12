"""Mediana adaptativa de dois estágios, com vizinhanças em blocos (I3)."""

from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.espacial import iterar_vizinhancas
from pdi.core.registro import registrar_processo
from pdi.core.util import aplicar_por_canal


def _mediana_adaptativa(canal: np.ndarray, janela_max: int) -> np.ndarray:
    saida = canal.copy()
    niveis = np.count_nonzero(np.bincount(canal.ravel(), minlength=256))
    if niveis == 1:
        return saida
    # Com apenas dois níveis, nenhuma mediana pode estar estritamente entre
    # mínimo e máximo. O estágio A nunca passa: equivale à mediana em S_max.
    tamanhos = (janela_max,) if niveis == 2 else range(3, janela_max + 1, 2)
    pendentes = np.ones(canal.shape, dtype=bool)
    for tamanho in tamanhos:
        if not pendentes.any():
            break
        for linhas, vizinhancas in iterar_vizinhancas(canal, tamanho):
            mascara = pendentes[linhas]
            if not mascara.any():
                continue
            # A seleção copia apenas os pixels pendentes deste bloco, nunca
            # todas as vizinhanças H×W×S_max². Os níveis permanecem uint8.
            selecionadas = vizinhancas[mascara]
            minimo = selecionadas.min(axis=(-2, -1))
            maximo = selecionadas.max(axis=(-2, -1))
            mediana = np.median(selecionadas, axis=(-2, -1))
            original = canal[linhas][mascara]
            # Comparações diretas evitam subtrações com overflow em uint8.
            estagio_a = (minimo < mediana) & (mediana < maximo)
            estagio_b = (minimo < original) & (original < maximo)
            valores = np.where(estagio_a & estagio_b, original, mediana)
            resolver = estagio_a | (tamanho == janela_max)
            y, x = np.nonzero(mascara)
            y, x = linhas.start + y[resolver], x[resolver]
            saida[y, x] = valores[resolver]
            pendentes[y, x] = False
    return saida


@registrar_processo(
    id="mediana_adaptativa",
    nome="Filtro adaptativo de mediana",
    grupo="Restauração",
    responsavel=3,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        Param(
            "janela_max",
            "Tamanho máximo da janela S_max",
            TipoParam.INT,
            7,
            minimo=3,
            maximo=31,
            passo=2,
            somente_impar=True,
            ajuda="A janela cresce de 3 até S_max enquanto a mediana for um ruído impulsivo.",
        ),
    ),
    descricao=(
        "Estágio A: cresce de 3 até S_max até mínimo < mediana < máximo. Estágio B: "
        "preserva o pixel não extremo ou usa a mediana; em S_max, usa a mediana se A falhar. "
        "Por canal, bordas refletidas; seleção exata uint8, sem normalização."
    ),
)
def mediana_adaptativa(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    saida = aplicar_por_canal(_mediana_adaptativa, img, p["janela_max"])
    return Resultado(saidas=[("Mediana adaptativa", saida)], info={"S_max": p["janela_max"]})
