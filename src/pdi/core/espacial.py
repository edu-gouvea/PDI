"""Vizinhanças, correlação e convolução em NumPy, compartilhadas com I3 (I2).

As bordas usam ``reflect`` por padrão: reflexão sem repetir o pixel da borda
(equivalente a BORDER_REFLECT_101). Também aceitam ``symmetric`` (repetindo a
borda), ``edge`` (replicação) e ``constant`` (zeros). O tamanho da saída é mantido.
"""

from collections.abc import Iterator
from numbers import Integral

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Limita o volume de cada bloco antes das reduções, principalmente da mediana,
# que pode copiar as vizinhanças. Nunca materializa H×W×janela² de uma vez.
_ELEMENTOS_POR_BLOCO = 1024 * 1024
_BORDAS = ("reflect", "symmetric", "edge", "constant")


def iterar_vizinhancas(
    img: np.ndarray, janela: int | tuple[int, int], *, borda: str = "reflect"
) -> Iterator[tuple[slice, np.ndarray]]:
    """Produz ``(linhas, vizinhancas)`` com shape ``(bloco, W, altura, largura)``.

    Aceita um canal 2D e uma janela ímpar positiva (inteiro ou par altura/largura).
    As vizinhanças são views somente leitura. Cada bloco contém no máximo cerca
    de 1 milhão de elementos, ou uma linha se ela já exceder esse limite.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.size == 0:
        raise ValueError("A imagem deve ser um array 2D não vazio.")
    dimensoes = (janela, janela) if isinstance(janela, Integral) else janela
    if (
        not isinstance(dimensoes, tuple)
        or len(dimensoes) != 2
        or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 1 or n % 2 == 0
            for n in dimensoes
        )
    ):
        raise ValueError("A janela deve ter dimensões inteiras, positivas e ímpares.")
    if borda not in _BORDAS:
        raise ValueError(f"Borda deve ser uma de: {', '.join(_BORDAS)}.")
    altura, largura = dimensoes
    margem = ((altura // 2, altura // 2), (largura // 2, largura // 2))
    expandida = np.pad(img, margem, mode=borda)
    linhas_por_bloco = max(1, _ELEMENTOS_POR_BLOCO // (img.shape[1] * altura * largura))
    for inicio in range(0, img.shape[0], linhas_por_bloco):
        fim = min(inicio + linhas_por_bloco, img.shape[0])
        faixa = expandida[inicio : fim + altura - 1]
        yield slice(inicio, fim), sliding_window_view(faixa, (altura, largura))


def correlacionar(img: np.ndarray, kernel: np.ndarray, *, borda: str = "reflect") -> np.ndarray:
    """Aplica um kernel 2D ímpar sem invertê-lo; devolve float64 sem clip.

    Preserva sinais e amplitudes para permitir Laplaciano e Sobel. Para RGB use
    ``aplicar_por_canal(correlacionar, img, kernel)``. A janela é o shape do kernel.
    """
    pesos = np.asarray(kernel, dtype=np.float64)
    if pesos.ndim != 2 or pesos.size == 0 or not np.all(np.isfinite(pesos)):
        raise ValueError("O kernel deve ser uma matriz 2D não vazia de valores finitos.")
    canal = np.asarray(img, dtype=np.float64)
    if not np.all(np.isfinite(canal)):
        raise ValueError("A imagem deve conter apenas valores finitos.")
    saida = np.empty(canal.shape, dtype=np.float64)
    for linhas, vizinhancas in iterar_vizinhancas(canal, pesos.shape, borda=borda):
        saida[linhas] = np.einsum("ijmn,mn->ij", vizinhancas, pesos)
    return saida


def convoluir(img: np.ndarray, kernel: np.ndarray, *, borda: str = "reflect") -> np.ndarray:
    """Convolução matemática: inverte os dois eixos do kernel antes de correlacionar.

    Devolve float64 do mesmo tamanho da imagem, sem normalização nem saturação.
    Kernels simétricos (como o gaussiano) dão o mesmo resultado nas duas funções.
    """
    pesos = np.asarray(kernel, dtype=np.float64)
    if pesos.ndim != 2:
        raise ValueError("O kernel deve ser uma matriz 2D.")
    return correlacionar(img, pesos[::-1, ::-1], borda=borda)
