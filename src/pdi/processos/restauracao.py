"""Filtro adaptativo de mediana (I3).

Especificação (nome, parâmetros, limites) proposta pelo I1 como stub; o I3 pode
ajustar limites e textos. Ao implementar, substitua o corpo da função.
"""

from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.registro import registrar_processo
from pdi.processos._stub import resultado_stub


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
        "Mediana com janela que cresce até S_max. Remove sal e pimenta denso "
        "preservando mais detalhes que a mediana comum."
    ),
)
def mediana_adaptativa(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return resultado_stub(img, 3)  # TODO(I3)
