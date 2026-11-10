"""Corpo provisório dos processos ainda não implementados.

Devolve a própria imagem de entrada, para que a interface possa ser testada com os
17 processos antes de as implementações reais chegarem. Apague este arquivo quando
nenhum processo o usar mais (``grep -rn resultado_stub src``).
"""

import numpy as np

from pdi.core.contrato import Resultado


def resultado_stub(img: np.ndarray, integrante: int) -> Resultado:
    return Resultado(
        saidas=[("Saída (stub)", img.copy())],
        info={"status": f"stub — aguardando Integrante {integrante}"},
    )
