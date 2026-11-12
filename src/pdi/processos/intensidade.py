"""Transformações de intensidade em NumPy; log e potência operam por canal (I2)."""

import math
from typing import Any

import numpy as np

from pdi.core.contrato import Histograma, Param, Resultado, TipoParam
from pdi.core.histograma import calcular_histograma
from pdi.core.registro import registrar_processo
from pdi.core.util import clip_para_uint8


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
    saida = np.where(img > p["k"], 255, 0).astype(np.uint8)
    return Resultado(saidas=[("Limiarizada", saida)], info={"limiar k": p["k"]})


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
    descricao=(
        "s = c·ln(1 + r), por canal. Expande tons escuros e comprime os claros. "
        "Arredondamento e clip em [0, 255], preservando o efeito do ganho c."
    ),
)
def transformacao_log(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    saida = clip_para_uint8(p["c"] * np.log1p(img.astype(np.float64)))
    return Resultado(saidas=[("Logarítmica", saida)], info={"ganho c": p["c"]})


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
    descricao=(
        "s = 255·c·(r/255)^γ, por canal. Arredondamento e clip em [0, 255], "
        "preservando o efeito do ganho c."
    ),
)
def transformacao_potencia(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    r = img.astype(np.float64) / 255.0
    saida = clip_para_uint8(255.0 * p["c"] * np.power(r, p["gama"]))
    return Resultado(saidas=[("Potência", saida)], info={"ganho c": p["c"], "gama γ": p["gama"]})


@registrar_processo(
    id="equalizacao_histograma",
    nome="Equalização de histograma",
    grupo="Intensidade",
    responsavel=2,
    aceita_cinza=True,
    aceita_rgb=False,
    descricao=(
        "Redistribui as intensidades pela CDF: s_k = round(255·Σ p(r_j)). "
        "Arredondamento e clip em [0, 255], sem subtrair a CDF mínima. "
        "Mostra os histogramas antes e depois."
    ),
)
def equalizacao_histograma(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    antes = calcular_histograma(img)
    # CDF discreta do Gonzalez: o primeiro nível ocupado também participa da soma.
    # Não subtrai CDF_min (variante do OpenCV); uma imagem constante vira branca.
    tabela = clip_para_uint8(antes.cumsum() * 255.0 / img.size)
    saida = tabela[img]
    return Resultado(
        saidas=[("Equalizada", saida)],
        acessorios=[Histograma("Antes", antes), Histograma("Depois", calcular_histograma(saida))],
    )


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
    if p["a"] > p["b"]:
        raise ValueError("A deve ser menor ou igual a B.")
    faixa = (img >= p["a"]) & (img <= p["b"])
    fundo = img if p["fundo"] == "preservar" else 0
    saida = np.where(faixa, 255, fundo).astype(np.uint8)
    return Resultado(
        saidas=[("Fatiada", saida)], info={"A": p["a"], "B": p["b"], "fundo": p["fundo"]}
    )
