"""FFT 2D centralizada, padding refletido e retorno ao domínio espacial (I3).

O padding dobra cada dimensão, com a imagem no centro. A reflexão evita criar
uma borda preta em imagens constantes. ``fftshift``/``ifftshift`` funcionam tanto
com dimensões pares quanto ímpares, inclusive sem padding.
"""

from numbers import Integral

import numpy as np


def _validar_shape(shape: tuple[int, int]) -> None:
    if len(shape) != 2 or any(
        isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in shape
    ):
        raise ValueError("As dimensões devem ser dois inteiros positivos.")


def _validar_matriz(arr: np.ndarray) -> None:
    if arr.ndim != 2 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("É necessária uma matriz 2D não vazia de valores finitos.")


def fft_centralizada(img: np.ndarray, *, padding: bool = True) -> np.ndarray:
    """Transforma um canal real 2D em espectro complexo com DC no centro.

    Com padding (padrão), devolve ``(2H, 2W)``; sem ele, ``(H, W)``. Para
    reconstruir e recortar, passe ``img.shape`` a ``inverter_fft``. Não altera img.
    """
    if np.iscomplexobj(img):
        raise ValueError("A imagem espacial deve conter valores reais.")
    canal = np.asarray(img, dtype=np.float64)
    _validar_matriz(canal)
    if padding:
        h, w = canal.shape
        canal = np.pad(canal, ((h // 2, h - h // 2), (w // 2, w - w // 2)), mode="reflect")
    return np.fft.fftshift(np.fft.fft2(canal))


def inverter_fft(espectro: np.ndarray, shape_original: tuple[int, int] | None = None) -> np.ndarray:
    """Inverte um espectro centralizado e recorta o centro, se solicitado.

    Devolve float64 com sinal, sem módulo, clip ou normalização. Os filtros
    reais e simétricos usados pelo projeto preservam a simetria conjugada, de
    modo que a parte imaginária da inversa é apenas erro de ponto flutuante.
    """
    matriz = np.asarray(espectro, dtype=np.complex128)
    _validar_matriz(matriz)
    espacial = np.fft.ifft2(np.fft.ifftshift(matriz)).real
    if shape_original is None:
        return espacial
    _validar_shape(shape_original)
    h, w = shape_original
    if h > matriz.shape[0] or w > matriz.shape[1]:
        raise ValueError("O recorte não pode ser maior que o espectro.")
    y, x = (matriz.shape[0] - h) // 2, (matriz.shape[1] - w) // 2
    return espacial[y : y + h, x : x + w].copy()


def grade_distancias(shape: tuple[int, int]) -> np.ndarray:
    """Distância euclidiana em bins da FFT ao DC central, em float64.

    ``D(u,v) = hypot(u − H//2, v − W//2)``. A unidade é um pixel do espectro
    com o padding já aplicado; D0 é especificado nessa mesma unidade.
    """
    _validar_shape(shape)
    h, w = shape
    y = np.arange(h, dtype=np.float64) - h // 2
    x = np.arange(w, dtype=np.float64) - w // 2
    return np.hypot(y[:, np.newaxis], x[np.newaxis, :])
