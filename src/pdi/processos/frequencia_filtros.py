"""Filtros passa-alta e passa-baixa gaussiano e Butterworth na frequência (I3).

Especificações (nome, parâmetros, limites) propostas pelo I1 como stubs; o I3 pode
ajustar limites e textos. Ao implementar, substitua o corpo das funções e documente
na ``descricao`` a política de volta para uint8 (clip ou min-max, ver core/util.py).
"""

from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.registro import registrar_processo
from pdi.processos._stub import resultado_stub

TIPO = Param(
    "tipo",
    "Tipo",
    TipoParam.ESCOLHA,
    "passa-baixa",
    opcoes=("passa-baixa", "passa-alta"),
    ajuda="Passa-baixa suaviza (mantém o centro do espectro); passa-alta realça bordas.",
)

D0 = Param(
    "d0",
    "Frequência de corte D0",
    TipoParam.FLOAT,
    30.0,
    minimo=1.0,
    maximo=1000.0,
    passo=1.0,
    ajuda="Raio de corte em pixels do espectro. Passa-baixa: menor D0 = mais desfoque.",
)


@registrar_processo(
    id="filtro_gaussiano_frequencia",
    nome="Passa-alta e passa-baixa gaussiano",
    grupo="Frequência",
    responsavel=3,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(TIPO, D0),
    descricao="H = exp(−D²/2D0²) (passa-baixa) ou 1 − H (passa-alta). Sem ringing.",
)
def filtro_gaussiano_frequencia(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 3)  # TODO(I3)


@registrar_processo(
    id="filtro_butterworth",
    nome="Passa-alta e passa-baixa Butterworth",
    grupo="Frequência",
    responsavel=3,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        TIPO,
        D0,
        Param(
            "ordem",
            "Ordem n",
            TipoParam.INT,
            2,
            minimo=1,
            maximo=20,
            passo=1,
            ajuda="Maior n = transição mais abrupta, próxima do filtro ideal (aparece ringing).",
        ),
    ),
    descricao="H = 1/(1 + (D/D0)^(2n)) (passa-baixa) ou 1 − H (passa-alta).",
)
def filtro_butterworth(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 3)  # TODO(I3)
