"""Testes de normalizar_para_uint8, clip_para_uint8 e aplicar_por_canal."""

import numpy as np
import pytest
from skimage.exposure import rescale_intensity

from pdi.core.util import aplicar_por_canal, clip_para_uint8, normalizar_para_uint8

# ------------------------------------------------------------------ normalizar_para_uint8


def test_normalizar_leva_min_para_0_e_max_para_255():
    lap = np.array([[-40.0, 0.0], [10.0, 60.0]])  # faixa típica de um Laplaciano
    s = normalizar_para_uint8(lap)
    assert s.dtype == np.uint8 and s.shape == lap.shape
    assert s.min() == 0 and s.max() == 255
    # 0 está a 40% do caminho entre -40 e 60: 0.4 · 255 = 102
    assert s[0, 1] == 102


def test_normalizar_igual_a_referencia_skimage():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 50, size=(64, 64))
    ref = np.rint(rescale_intensity(x, in_range="image", out_range=(0, 255))).astype(np.uint8)
    np.testing.assert_array_equal(normalizar_para_uint8(x), ref)


def test_normalizar_preserva_ordem():
    x = np.array([3.0, -1.0, 7.5, 2.0])
    s = normalizar_para_uint8(x)
    np.testing.assert_array_equal(np.argsort(s, kind="stable"), np.argsort(x, kind="stable"))


def test_normalizar_imagem_constante_vira_zeros():
    s = normalizar_para_uint8(np.full((5, 5), 42.0))
    assert s.dtype == np.uint8 and not s.any()


def test_normalizar_ignora_o_ganho():
    # Motivo de NÃO usar min-max em processos com ganho c (ver docstring do módulo).
    r = np.linspace(0, 255, 50)
    np.testing.assert_array_equal(
        normalizar_para_uint8(1.0 * np.log1p(r)), normalizar_para_uint8(30.0 * np.log1p(r))
    )


def test_normalizar_rgb_usa_min_max_global():
    x = np.zeros((2, 2, 3))
    x[..., 0] = 100.0  # canal R mais forte
    x[..., 1] = 50.0
    s = normalizar_para_uint8(x)
    assert (s[..., 0] == 255).all() and (s[..., 1] == 128).all() and (s[..., 2] == 0).all()


def test_normalizar_nao_modifica_entrada():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    copia = x.copy()
    normalizar_para_uint8(x)
    np.testing.assert_array_equal(x, copia)


# ------------------------------------------------------------------------ clip_para_uint8


def test_clip_satura_e_arredonda():
    x = np.array([-30.0, 0.4, 0.6, 127.6, 254.4, 300.0])
    np.testing.assert_array_equal(clip_para_uint8(x), [0, 0, 1, 128, 254, 255])
    assert clip_para_uint8(x).dtype == np.uint8


def test_clip_nao_reescala_valores_validos():
    x = np.arange(256, dtype=np.float64).reshape(16, 16)
    np.testing.assert_array_equal(clip_para_uint8(x), x.astype(np.uint8))


@pytest.mark.parametrize("func", [clip_para_uint8, normalizar_para_uint8])
@pytest.mark.parametrize("ruim", [np.nan, np.inf, -np.inf])
def test_valores_invalidos_sao_recusados(func, ruim):
    with pytest.raises(ValueError, match="NaN ou infinito"):
        func(np.array([1.0, ruim]))


# ---------------------------------------------------------------------- aplicar_por_canal


def _dobrar(canal, fator=2):
    return canal.astype(np.float64) * fator


def test_aplicar_por_canal_cinza(img_cinza):
    s = aplicar_por_canal(_dobrar, img_cinza)
    np.testing.assert_array_equal(s, img_cinza * 2.0)


def test_aplicar_por_canal_rgb_independente_e_em_ordem(img_rgb):
    s = aplicar_por_canal(_dobrar, img_rgb, fator=3)
    assert s.shape == img_rgb.shape
    for c in range(3):
        np.testing.assert_array_equal(s[..., c], img_rgb[..., c] * 3.0)


def test_aplicar_por_canal_chama_func_com_2d(img_rgb):
    formatos = []

    def espiao(canal):
        formatos.append(canal.shape)
        return canal

    aplicar_por_canal(espiao, img_rgb)
    assert formatos == [img_rgb.shape[:2]] * 3


def test_aplicar_por_canal_recusa_saida_com_formato_errado(img_rgb):
    with pytest.raises(ValueError, match="cortar"):

        def cortar(canal):
            return canal[1:, :]

        aplicar_por_canal(cortar, img_rgb)
