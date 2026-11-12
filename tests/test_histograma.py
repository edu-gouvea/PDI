"""Contagem exata, níveis ausentes e validação do histograma compartilhado."""

import numpy as np
import pytest

from pdi.core.histograma import calcular_histograma


def test_contagem_inclui_extremos_e_niveis_ausentes():
    img = np.array([[0, 255, 128], [128, 255, 255]], dtype=np.uint8)
    img.flags.writeable = False
    hist = calcular_histograma(img)
    assert hist.shape == (256,)
    assert np.issubdtype(hist.dtype, np.integer)
    assert (hist[0], hist[128], hist[255], hist[1]) == (1, 2, 3, 0)
    assert hist.sum() == img.size


def test_histograma_de_view_nao_contigua(img_rgb):
    canal = img_rgb[..., 1]
    hist = calcular_histograma(canal)
    for nivel in range(256):
        assert hist[nivel] == np.count_nonzero(canal == nivel)


@pytest.mark.parametrize(
    "img",
    [np.zeros((2, 2, 3), np.uint8), np.zeros((0, 1), np.uint8), np.ones((2, 2))],
)
def test_recusa_imagens_invalidas(img):
    with pytest.raises(ValueError):
        calcular_histograma(img)
