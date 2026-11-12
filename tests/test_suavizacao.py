"""Suavização: referências independentes, bordas, RGB e parâmetros extremos."""

import cv2
import numpy as np
import pytest

from pdi.core.espacial import convoluir
from pdi.core.registro import executar_processo, obter_processo


def _saida(processo, img, **parametros):
    return executar_processo(processo, img, parametros).saidas[0][1]


@pytest.mark.parametrize("janela", [3, 7, 31])
@pytest.mark.parametrize("sigma", [0.1, 1, 20])
def test_gaussiano_confere_com_opencv(janela, sigma, img_rgb):
    # CV_64F evita o arredondamento intermediário do caminho uint8 do OpenCV.
    esperado = cv2.GaussianBlur(
        img_rgb.astype(np.float64),
        (janela, janela),
        sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    esperado = np.clip(np.rint(esperado), 0, 255).astype(np.uint8)
    np.testing.assert_allclose(
        _saida("filtro_gaussiano", img_rgb, janela=janela, sigma=sigma), esperado, atol=1
    )


def test_gaussiano_separavel_equivale_a_kernel_2d(img_cinza):
    y, x = np.mgrid[-3:4, -3:4]
    kernel = np.exp(-(x**2 + y**2) / (2 * 1.4**2))
    kernel /= kernel.sum()
    esperado = np.rint(convoluir(img_cinza, kernel)).astype(np.uint8)
    np.testing.assert_array_equal(
        _saida("filtro_gaussiano", img_cinza, sigma=1.4, janela=7), esperado
    )


def test_sigma_e_janela_alteram_desfoque():
    img = np.zeros((33, 33), np.uint8)
    img[16, 16] = 255
    estreito = _saida("filtro_gaussiano", img, sigma=0.1, janela=7)
    largo = _saida("filtro_gaussiano", img, sigma=2, janela=7)
    curto = _saida("filtro_gaussiano", img, sigma=2, janela=3)
    np.testing.assert_array_equal(estreito, img)
    assert 0 < largo[16, 16] < curto[16, 16] < estreito[16, 16]
    assert np.count_nonzero(largo) > np.count_nonzero(curto)


def test_gaussiano_nao_estica_contraste():
    img = np.full((9, 9), 100, np.uint8)
    img[4, 4] = 150
    saida = _saida("filtro_gaussiano", img)
    assert saida.min() >= 100 and saida.max() < 150


@pytest.mark.parametrize("tipo", ["mediana", "mínimo", "máximo"])
@pytest.mark.parametrize("janela", [3, 7, 31])
def test_ordem_confere_com_opencv(tipo, janela, img_cinza):
    raio = janela // 2
    # Padding explícito mantém a mesma política na mediana do OpenCV, que usa
    # replicação internamente. O recorte fica longe da borda criada por ele.
    expandida = np.pad(img_cinza, raio, mode="reflect")
    if tipo == "mediana":
        referencia = cv2.medianBlur(expandida, janela)
    else:
        operacao = cv2.erode if tipo == "mínimo" else cv2.dilate
        referencia = operacao(expandida, np.ones((janela, janela), np.uint8))
    esperado = referencia[raio:-raio, raio:-raio]
    np.testing.assert_array_equal(
        _saida("filtro_ordem", img_cinza, tipo=tipo, janela=janela), esperado
    )


@pytest.mark.parametrize("tipo", ["mediana", "mínimo", "máximo"])
def test_ordem_imagem_aleatoria(tipo, img_rgb):
    esperado = []
    for c in range(3):
        expandida = np.pad(img_rgb[..., c], 1, mode="reflect")
        canal = np.empty(img_rgb.shape[:2], np.uint8)
        # Referência escalar independente do processamento vetorizado em blocos.
        for y, x in np.ndindex(canal.shape):
            niveis = sorted(expandida[y : y + 3, x : x + 3].ravel())
            canal[y, x] = niveis[{"mínimo": 0, "mediana": 4, "máximo": 8}[tipo]]
        esperado.append(canal)
    np.testing.assert_array_equal(
        _saida("filtro_ordem", img_rgb, tipo=tipo), np.stack(esperado, axis=-1)
    )


def test_ordem_remove_impulsos_e_janela_altera_resultado():
    img = np.full((9, 9), 100, np.uint8)
    img[4, 4] = 255
    assert _saida("filtro_ordem", img, tipo="mediana")[4, 4] == 100
    assert _saida("filtro_ordem", img, tipo="mínimo")[4, 4] == 100
    img[4, 4] = 0
    assert _saida("filtro_ordem", img, tipo="máximo")[4, 4] == 100
    img[3:6, 3:6] = 255
    assert _saida("filtro_ordem", img, janela=3)[4, 4] == 255
    assert _saida("filtro_ordem", img, janela=5)[4, 4] == 100


@pytest.mark.parametrize(
    ("processo", "parametros"),
    [
        ("filtro_gaussiano", {"sigma": 20}),
        ("filtro_ordem", {"tipo": "mediana"}),
        ("filtro_ordem", {"tipo": "mínimo"}),
        ("filtro_ordem", {"tipo": "máximo"}),
    ],
)
@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1), (2, 3, 3)])
@pytest.mark.parametrize("nivel", [0, 93, 255])
def test_constante_e_imagem_menor_que_janela(processo, parametros, shape, nivel):
    img = np.full(shape, nivel, np.uint8)
    np.testing.assert_array_equal(_saida(processo, img, janela=31, **parametros), img)


@pytest.mark.parametrize("processo", ["filtro_gaussiano", "filtro_ordem"])
def test_rgb_equivale_a_canais_independentes(processo, img_rgb):
    saida = _saida(processo, img_rgb)
    esperado = np.stack([_saida(processo, img_rgb[..., c]) for c in range(3)], axis=-1)
    np.testing.assert_array_equal(saida, esperado)


@pytest.mark.parametrize("processo", ["filtro_gaussiano", "filtro_ordem"])
def test_entrada_somente_leitura_e_view_nao_contigua(processo, img_rgb):
    img = img_rgb[::2, ::2]
    original = img.copy()
    img.flags.writeable = False
    spec = obter_processo(processo)
    res = spec.funcao(img, {p.nome: p.padrao for p in spec.parametros})
    np.testing.assert_array_equal(img, original)
    assert res.saidas[0][1].shape == img.shape and res.saidas[0][1].dtype == np.uint8
    assert not np.shares_memory(img, res.saidas[0][1])
