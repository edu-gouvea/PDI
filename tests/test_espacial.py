"""Convolução compartilhada: orientação, bordas, precisão e blocos limitados."""

import cv2
import numpy as np
import pytest

from pdi.core import espacial
from pdi.core.espacial import convoluir, correlacionar, iterar_vizinhancas


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (3, 1), (8, 11)])
@pytest.mark.parametrize("janela", [(1, 1), (3, 5), (7, 7)])
def test_correlacao_confere_com_opencv(shape, janela):
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, shape, dtype=np.uint8)
    kernel = rng.normal(size=janela)
    original = img.copy()
    img.flags.writeable = False
    esperado = cv2.filter2D(img.astype(np.float64), -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    obtido = correlacionar(img, kernel)
    np.testing.assert_allclose(obtido, esperado, atol=1e-10)
    np.testing.assert_array_equal(img, original)
    assert obtido.dtype == np.float64


def test_convolucao_inverte_kernel_assimetrico():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, 2] = 1
    kernel = np.arange(-4, 5, dtype=np.float64).reshape(3, 3)
    np.testing.assert_array_equal(convoluir(img, kernel)[1:4, 1:4], kernel)
    np.testing.assert_array_equal(correlacionar(img, kernel)[1:4, 1:4], kernel[::-1, ::-1])


@pytest.mark.parametrize(
    ("borda", "tipo_cv"),
    [
        ("reflect", cv2.BORDER_REFLECT_101),
        ("symmetric", cv2.BORDER_REFLECT),
        ("edge", cv2.BORDER_REPLICATE),
        ("constant", cv2.BORDER_CONSTANT),
    ],
)
def test_bordas(borda, tipo_cv):
    img = np.arange(20, dtype=np.float64).reshape(4, 5)
    kernel = np.ones((3, 3)) / 9
    esperado = cv2.filter2D(img, -1, kernel, borderType=tipo_cv)
    np.testing.assert_allclose(convoluir(img, kernel, borda=borda), esperado)


def test_blocos_nao_alteram_resultado(monkeypatch):
    img = np.arange(30, dtype=np.uint8).reshape(6, 5)
    esperado = correlacionar(img, np.ones((3, 3)))
    monkeypatch.setattr(espacial, "_ELEMENTOS_POR_BLOCO", 45)
    blocos = list(iterar_vizinhancas(img, 3))
    assert len(blocos) == 6
    assert all(v.shape == (1, 5, 3, 3) and not v.flags.writeable for _, v in blocos)
    np.testing.assert_array_equal(correlacionar(img, np.ones((3, 3))), esperado)


@pytest.mark.parametrize("kernel", [np.ones((2, 3)), np.ones((0, 0)), [1, 2], [[np.nan]]])
def test_recusa_kernel_invalido(kernel):
    with pytest.raises(ValueError):
        convoluir(np.ones((2, 3)), kernel)


@pytest.mark.parametrize("janela", [0, -1, 2, 3.0, True, (3, 2)])
def test_recusa_janela_invalida(janela):
    with pytest.raises(ValueError, match="janela"):
        list(iterar_vizinhancas(np.ones((2, 3)), janela))


def test_recusa_borda_invalida():
    with pytest.raises(ValueError, match="Borda"):
        convoluir(np.ones((2, 3)), [[1]], borda="circular")


@pytest.mark.parametrize("img", [np.ones((2, 3, 3)), np.empty((0, 3)), [[np.inf]]])
def test_recusa_imagem_invalida(img):
    with pytest.raises(ValueError):
        convoluir(img, [[1]])
