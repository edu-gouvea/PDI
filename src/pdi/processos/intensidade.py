"""Limiarização, log, potência, equalização de histograma e fatiamento (I2).

Especificações (nome, parâmetros, limites) propostas pelo I1 como stubs; o I2 pode
ajustar limites e textos. Ao implementar, substitua o corpo das funções e documente
na ``descricao`` a política de volta para uint8 (clip ou min-max, ver core/util.py).
"""

import math
from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.registro import registrar_processo
from pdi.processos._stub import resultado_stub


@registrar_processo(
    id="limiarizacao",
    nome="Limiarização",
    grupo="Intensidade",
    responsavel=2,
    aceita_cinza=True,
    aceita_rgb=False,
    parametros=(
        Param(
            "k",
            "Limiar k",
            TipoParam.INT,
            127,
            minimo=0,
            maximo=255,
            passo=1,
            ajuda="Pixels com r > k viram 255 (branco); os demais viram 0 (preto).",
        ),
    ),
    descricao="Binariza a imagem: s = 255 se r > k, senão 0.",
)
def limiarizacao(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 2)  # TODO(I2)


@registrar_processo(
    id="transformacao_log",
    nome="Transformação logarítmica",
    grupo="Intensidade",
    responsavel=2,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        Param(
            "c",
            "Ganho c",
            TipoParam.FLOAT,
            round(255 / math.log(256), 2),  # 45,99: leva r = 255 para s = 255
            minimo=0.0,
            maximo=200.0,
            passo=0.5,
            ajuda="Com c ≈ 45,99 a saída ocupa [0, 255]. Maior c clareia e satura mais.",
        ),
    ),
    descricao="s = c·log(1 + r). Expande tons escuros e comprime os claros.",
)
def transformacao_log(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 2)  # TODO(I2)


@registrar_processo(
    id="transformacao_potencia",
    nome="Transformação de potência (gama)",
    grupo="Intensidade",
    responsavel=2,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        Param(
            "c",
            "Ganho c",
            TipoParam.FLOAT,
            1.0,
            minimo=0.0,
            maximo=10.0,
            passo=0.1,
            ajuda="Multiplica a saída. c = 1 mantém a faixa [0, 255].",
        ),
        Param(
            "gama",
            "Gama γ",
            TipoParam.FLOAT,
            1.0,
            minimo=0.01,
            maximo=25.0,
            passo=0.05,
            ajuda="γ < 1 clareia (realça escuros); γ > 1 escurece (realça claros).",
        ),
    ),
    descricao="s = c·r^γ, com r normalizado em [0, 1] e reescalado para [0, 255].",
)
def transformacao_potencia(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 2)  # TODO(I2)


@registrar_processo(
    id="equalizacao_histograma",
    nome="Equalização de histograma",
    grupo="Intensidade",
    responsavel=2,
    aceita_cinza=True,
    aceita_rgb=False,
    descricao=(
        "Redistribui as intensidades pela CDF: s_k = round(255·Σ p(r_j)). "
        "Mostra os histogramas antes e depois."
    ),
)
def equalizacao_histograma(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 2)  # TODO(I2): incluir os 2 Histograma nos acessórios


@registrar_processo(
    id="fatiamento_intensidade",
    nome="Fatiamento por intensidade",
    grupo="Intensidade",
    responsavel=2,
    aceita_cinza=True,
    aceita_rgb=False,
    parametros=(
        Param(
            "a",
            "Início da faixa A",
            TipoParam.INT,
            100,
            minimo=0,
            maximo=255,
            passo=1,
            ajuda="Menor intensidade realçada.",
        ),
        Param(
            "b",
            "Fim da faixa B",
            TipoParam.INT,
            180,
            minimo=0,
            maximo=255,
            passo=1,
            ajuda="Maior intensidade realçada. Deve ser ≥ A.",
        ),
        Param(
            "fundo",
            "Fora da faixa",
            TipoParam.ESCOLHA,
            "preservar",
            opcoes=("preservar", "zerar"),
            ajuda="Preservar = mantém os pixels fora de [A, B]; zerar = viram 0 (preto).",
        ),
    ),
    descricao="Pixels em [A, B] viram 255; os demais são preservados ou zerados.",
)
def fatiamento_intensidade(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    # TODO(I2): levantar ValueError("A deve ser menor ou igual a B.") se a > b
    return resultado_stub(img, 2)
