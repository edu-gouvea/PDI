"""Histograma de 256 níveis, reutilizável sem dependências de interface (I2)."""

import numpy as np

from pdi.core.util import TIPO_CINZA, tipo_da_imagem


def calcular_histograma(img: np.ndarray) -> np.ndarray:
    """Conta os pixels de cada nível de cinza, incluindo os níveis ausentes.

    Devolve contagens inteiras de shape ``(256,)``. Para RGB, o chamador deve
    selecionar explicitamente um canal; misturar os canais ocultaria as cores.
    """
    if tipo_da_imagem(img) != TIPO_CINZA:
        raise ValueError("O histograma requer uma imagem em escala de cinza ou um canal RGB.")
    return np.bincount(img.ravel(), minlength=256)
