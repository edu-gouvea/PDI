"""FFT: DFT de referência, padding, centralização, recorte e preservação do sinal."""

import numpy as np
import pytest

from pdi.core.frequencia import fft_centralizada, grade_distancias, inverter_fft


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5), (4, 6)])
@pytest.mark.parametrize("padding", [False, True])
def test_fft_e_inversa_reconstroem_imagem(shape, padding):
    img = np.random.default_rng(3).normal(size=shape)
    original = img.copy()
    img.flags.writeable = False
    espectro = fft_centralizada(img, padding=padding)
    fator = 2 if padding else 1
    assert espectro.shape == (fator * shape[0], fator * shape[1])
    np.testing.assert_allclose(inverter_fft(espectro, shape), img, atol=1e-12)
    np.testing.assert_array_equal(img, original)


def test_fft_confere_com_dft_explicita():
    img = np.arange(15, dtype=np.float64).reshape(3, 5)
    esperado = np.zeros(img.shape, dtype=np.complex128)
    for u, v in np.ndindex(img.shape):
        for y, x in np.ndindex(img.shape):
            esperado[u, v] += img[y, x] * np.exp(-2j * np.pi * (u * y / 3 + v * x / 5))
    np.testing.assert_allclose(
        fft_centralizada(img, padding=False), np.roll(esperado, (1, 2), axis=(0, 1)), atol=1e-12
    )


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (4, 6)])
def test_padding_reflete_ao_redor_do_centro(shape):
    img = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    h, w = shape
    esperado = np.pad(img, ((h // 2, h - h // 2), (w // 2, w - w // 2)), mode="reflect")
    np.testing.assert_allclose(inverter_fft(fft_centralizada(img)), esperado, atol=1e-12)


@pytest.mark.parametrize("shape", [(3, 5), (4, 6)])
def test_dc_centralizado_e_constante_sem_borda_preta(shape):
    for padding in (False, True):
        espectro = fft_centralizada(np.full(shape, 73), padding=padding)
        h, w = espectro.shape
        esperado = np.zeros_like(espectro)
        esperado[h // 2, w // 2] = 73 * h * w
        np.testing.assert_allclose(espectro, esperado, atol=1e-10)


def test_inversa_preserva_valores_negativos():
    img = np.array([[-30.5, 0, 40.5]])
    np.testing.assert_allclose(inverter_fft(fft_centralizada(img), img.shape), img, atol=1e-12)


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (4, 6)])
def test_grade_distancias_conhecida(shape):
    d = grade_distancias(shape)
    for y, x in np.ndindex(shape):
        assert d[y, x] == pytest.approx(
            ((y - shape[0] // 2) ** 2 + (x - shape[1] // 2) ** 2) ** 0.5
        )
    assert d[shape[0] // 2, shape[1] // 2] == 0


@pytest.mark.parametrize("img", [np.empty((0, 2)), np.ones((2, 2, 3)), [[np.nan]], [[1j]]])
def test_fft_recusa_entrada_invalida(img):
    with pytest.raises(ValueError):
        fft_centralizada(img)


@pytest.mark.parametrize("shape", [(0, 2), (3, -1), (2.5, 3), (True, 3), (3,)])
def test_recusa_shape_invalido(shape):
    with pytest.raises(ValueError):
        grade_distancias(shape)
    with pytest.raises(ValueError):
        inverter_fft(np.ones((4, 4)), shape)


def test_recusa_recorte_maior_que_espectro():
    with pytest.raises(ValueError, match="recorte"):
        inverter_fft(np.ones((4, 4)), (5, 3))


@pytest.mark.parametrize("img", [np.empty((0, 2)), np.ones((2, 2, 3)), [[np.inf]]])
def test_inversa_recusa_espectro_invalido(img):
    with pytest.raises(ValueError):
        inverter_fft(img)
