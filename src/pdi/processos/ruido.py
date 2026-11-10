"""Ruído aditivo gaussiano e ruídos sal, pimenta e sal e pimenta (I1)."""

from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.registro import registrar_processo
from pdi.processos._stub import resultado_stub

SEMENTE = Param(
    "semente",
    "Semente",
    TipoParam.INT,
    0,
    minimo=0,
    maximo=999_999,
    passo=1,
    ajuda="0 = ruído diferente a cada execução. Outro valor = mesmo ruído sempre (reprodutível).",
)


@registrar_processo(
    id="ruido_gaussiano",
    nome="Ruído aditivo gaussiano",
    grupo="Ruído",
    responsavel=1,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        Param(
            "media",
            "Média μ",
            TipoParam.FLOAT,
            0.0,
            minimo=-128.0,
            maximo=128.0,
            passo=1.0,
            ajuda="Deslocamento médio do ruído. μ > 0 clareia, μ < 0 escurece.",
        ),
        Param(
            "desvio",
            "Desvio padrão σ",
            TipoParam.FLOAT,
            20.0,
            minimo=0.0,
            maximo=128.0,
            passo=1.0,
            ajuda="Intensidade do ruído. Maior σ = granulado mais forte.",
        ),
        Param(
            "fracao",
            "Fração de pixels afetados (0 a 1)",
            TipoParam.FLOAT,
            1.0,
            minimo=0.0,
            maximo=1.0,
            passo=0.05,
            ajuda="1 = ruído em todos os pixels; 0,3 = em 30% deles, sorteados.",
        ),
        SEMENTE,
    ),
    descricao="Soma ruído η ~ N(μ, σ²) aos pixels: g = f + η. Saturação por clip em [0, 255].",
)
def ruido_gaussiano(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 1)  # TODO(I1): Etapa 3


@registrar_processo(
    id="ruido_sal_pimenta",
    nome="Ruído sal, pimenta e sal e pimenta",
    grupo="Ruído",
    responsavel=1,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        Param(
            "tipo",
            "Tipo",
            TipoParam.ESCOLHA,
            "sal e pimenta",
            opcoes=("sal e pimenta", "sal", "pimenta"),
            ajuda="Sal = pixels brancos (255); pimenta = pixels pretos (0).",
        ),
        Param(
            "probabilidade",
            "Probabilidade p (0 a 1)",
            TipoParam.FLOAT,
            0.05,
            minimo=0.0,
            maximo=1.0,
            passo=0.01,
            ajuda="Fração de pixels corrompidos. p = 0,1 corrompe ~10% da imagem.",
        ),
        Param(
            "proporcao_sal",
            "Proporção de sal (0 a 1)",
            TipoParam.FLOAT,
            0.5,
            minimo=0.0,
            maximo=1.0,
            passo=0.05,
            ajuda="Só para 'sal e pimenta': 0,5 = metade sal, metade pimenta.",
        ),
        SEMENTE,
    ),
    descricao="Com probabilidade p, cada pixel vira 255 (sal) ou 0 (pimenta).",
)
def ruido_sal_pimenta(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 1)  # TODO(I1): Etapa 3
