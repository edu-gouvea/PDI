"""Realce: fórmulas, sinais, máscaras estendidas e referências OpenCV."""

import cv2
import numpy as np
import pytest

from pdi.core.registro import executar_processo, obter_processo
from pdi.processos.realce import _kernel_laplaciano, _kernels_sobel


def _uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _minmax(x):
    if x.max() == x.min():
        return np.zeros_like(x, dtype=np.uint8)
    return np.rint((x - x.min()) * 255 / (x.max() - x.min())).astype(np.uint8)


@pytest.mark.parametrize("ganho", [0, 1, 2.5, 10])
@pytest.mark.parametrize("janela", [3, 7, 31])
def test_agucamento_confere_com_media_opencv(ganho, janela, img_rgb):
    original = img_rgb.astype(np.float64)
    suave = cv2.blur(original, (janela, janela), borderType=cv2.BORDER_REFLECT_101)
    esperado = _uint8(original + ganho * (original - suave))
    obtido = executar_processo("mascara_agucamento", img_rgb, {"ganho": ganho, "janela": janela})
    np.testing.assert_allclose(obtido.saidas[0][1], esperado, atol=1)


def test_agucamento_ganho_zero_identidade_e_ganho_altera_saida():
    img = np.full((9, 9), 100, np.uint8)
    img[4, 4] = 120
    saidas = [
        executar_processo("mascara_agucamento", img, {"ganho": k}).saidas[0][1] for k in (0, 1, 2)
    ]
    np.testing.assert_array_equal(saidas[0], img)
    assert img[4, 4] < saidas[1][4, 4] < saidas[2][4, 4] < 255


def test_laplaciano_classico_tem_sinal_correto():
    img = np.full((5, 5), 100, np.uint8)
    img[2, 2] = 120
    res = executar_processo("realce_laplaciano", img)
    esperado = img.copy()
    esperado[2, 2] = 200  # Δf = 4·100 − 4·120 = −80; f − Δf = 200.
    esperado[1, 2] = esperado[2, 1] = esperado[2, 3] = esperado[3, 2] = 80
    np.testing.assert_array_equal(res.saidas[0][1], esperado)
    assert res.acessorios[0][0] == "Laplaciano ajustado"
    ajustado = res.acessorios[0][1]
    assert ajustado[2, 2] == 0 and ajustado[1, 2] == 255


@pytest.mark.parametrize("janela", [3, 5, 7, 15])
def test_laplaciano_estendido_confere_com_opencv(janela, img_rgb):
    # Suavização binomial gerada independentemente por convoluções sucessivas.
    suave = np.array([1.0])
    for _ in range(janela - 3):
        suave = np.convolve(suave, [0.5, 0.5])
    entrada = img_rgb.astype(np.float64)
    suavizada = cv2.sepFilter2D(entrada, -1, suave, suave, borderType=cv2.BORDER_REFLECT_101)
    lap = cv2.Laplacian(suavizada, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT_101)
    res = executar_processo("realce_laplaciano", img_rgb, {"janela": janela})
    np.testing.assert_allclose(res.saidas[0][1], _uint8(entrada - lap), atol=1)
    np.testing.assert_allclose(res.acessorios[0][1], _minmax(lap), atol=1)
    kernel = _kernel_laplaciano(janela)
    assert kernel.shape == (janela, janela) and kernel.sum() == 0


@pytest.mark.parametrize("janela", [3, 5, 7, 15])
def test_sobel_confere_com_opencv(janela, img_rgb):
    entrada = img_rgb.astype(np.float64)
    gx = cv2.Sobel(entrada, cv2.CV_64F, 1, 0, ksize=janela, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(entrada, cv2.CV_64F, 0, 1, ksize=janela, borderType=cv2.BORDER_REFLECT_101)
    res = executar_processo("gradiente_sobel", img_rgb, {"janela": janela})
    np.testing.assert_allclose(res.saidas[0][1], _minmax(np.hypot(gx, gy)), atol=1)
    dx, sx = cv2.getDerivKernels(1, 0, janela, ktype=cv2.CV_64F)
    kx, ky = _kernels_sobel(janela)
    np.testing.assert_array_equal(kx, sx @ dx.T)
    np.testing.assert_array_equal(ky, kx.T)


def test_sobel_detecta_borda_e_nao_converte_gradiente_em_realce():
    img = np.full((9, 9), 40, np.uint8)
    img[:, 5:] = 100
    saida = executar_processo("gradiente_sobel", img).saidas[0][1]
    assert np.all(saida[:, 4:6] == 255)
    assert np.all(saida[:, :4] == 0) and np.all(saida[:, 6:] == 0)


@pytest.mark.parametrize("processo", ["mascara_agucamento", "realce_laplaciano", "gradiente_sobel"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (3, 1), (2, 3, 3)])
def test_constantes_e_imagens_menores_que_janela(processo, shape):
    img = np.full(shape, 83, np.uint8)
    maximo = 31 if processo == "mascara_agucamento" else 15
    res = executar_processo(processo, img, {"janela": maximo})
    esperado = np.zeros_like(img) if processo == "gradiente_sobel" else img
    np.testing.assert_array_equal(res.saidas[0][1], esperado)
    if res.acessorios:
        np.testing.assert_array_equal(res.acessorios[0][1], np.zeros_like(img))


@pytest.mark.parametrize("processo", ["mascara_agucamento", "realce_laplaciano", "gradiente_sobel"])
def test_janela_altera_resposta(processo, img_rgb):
    a = executar_processo(processo, img_rgb, {"janela": 3}).saidas[0][1]
    b = executar_processo(processo, img_rgb, {"janela": 7}).saidas[0][1]
    assert np.any(a != b)


@pytest.mark.parametrize("processo", ["mascara_agucamento", "realce_laplaciano", "gradiente_sobel"])
def test_entrada_somente_leitura_e_nao_contigua(processo, img_rgb):
    img = img_rgb[::2, ::2]
    original = img.copy()
    img.flags.writeable = False
    spec = obter_processo(processo)
    res = spec.funcao(img, {p.nome: p.padrao for p in spec.parametros})
    np.testing.assert_array_equal(img, original)
    assert res.saidas[0][1].shape == img.shape and res.saidas[0][1].dtype == np.uint8
    assert not np.shares_memory(img, res.saidas[0][1])
