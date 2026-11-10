"""Testes dos processos de cor: decomposição RGB (e HSV, na próxima etapa)."""

import cv2
import numpy as np
import pytest

from pdi.core.registro import executar_processo

# ------------------------------------------------------------------ decomposição RGB


def test_rgb_tres_saidas_em_cinza_na_ordem(img_rgb):
    res = executar_processo("decomposicao_rgb", img_rgb)  # valida o contrato também
    assert [t for t, _ in res.saidas] == ["R (vermelho)", "G (verde)", "B (azul)"]
    for i, (_, canal) in enumerate(res.saidas):
        assert canal.dtype == np.uint8 and canal.shape == img_rgb.shape[:2]
        np.testing.assert_array_equal(canal, img_rgb[..., i])


def test_rgb_igual_ao_opencv(img_rgb):
    # cv2.split devolve na ordem B, G, R (o OpenCV trabalha em BGR).
    b, g, r = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    saidas = [c for _, c in executar_processo("decomposicao_rgb", img_rgb).saidas]
    for nosso, referencia in zip(saidas, (r, g, b), strict=True):
        np.testing.assert_array_equal(nosso, referencia)


@pytest.mark.parametrize(
    ("cor", "esperado"),
    [
        ((255, 0, 0), (255, 0, 0)),  # vermelho puro: claro só no canal R
        ((0, 255, 0), (0, 255, 0)),
        ((0, 0, 255), (0, 0, 255)),
        ((255, 255, 0), (255, 255, 0)),  # amarelo = vermelho + verde
        ((255, 255, 255), (255, 255, 255)),  # branco: claro nos três
        ((0, 0, 0), (0, 0, 0)),
    ],
)
def test_rgb_cores_puras(cor, esperado):
    img = np.empty((4, 4, 3), np.uint8)
    img[...] = cor
    saidas = [c for _, c in executar_processo("decomposicao_rgb", img).saidas]
    assert tuple(int(c[0, 0]) for c in saidas) == esperado


def test_rgb_info_com_medias():
    img = np.zeros((2, 2, 3), np.uint8)
    img[..., 0] = 200
    img[0, 0, 2] = 100  # média do B = 100/4 = 25
    info = executar_processo("decomposicao_rgb", img).info
    assert info == {"média R": 200.0, "média G": 0.0, "média B": 25.0}


def test_rgb_saidas_independentes_da_entrada(img_rgb):
    original = img_rgb.copy()
    res = executar_processo("decomposicao_rgb", img_rgb)
    res.saidas[0][1][:] = 0  # mexer na saída não pode afetar a entrada
    np.testing.assert_array_equal(img_rgb, original)
    assert all(c.flags.c_contiguous for _, c in res.saidas)


def test_rgb_recusa_imagem_cinza(img_cinza):
    with pytest.raises(ValueError, match="coloridas"):
        executar_processo("decomposicao_rgb", img_cinza)


# ------------------------------------------------------------------ decomposição HSV


def _cubo_rgb(passo=5):
    """Muitas cores cobrindo o cubo RGB inteiro, inclusive cinzas, preto e branco."""
    v = np.arange(0, 256, passo, dtype=np.uint8)
    r, g, b = np.meshgrid(v, v, v, indexing="ij")
    return np.stack([r, g, b], axis=-1).reshape(-1, len(v), 3)


def test_hsv_igual_ao_skimage_em_todo_o_cubo_rgb():
    from skimage.color import rgb2hsv

    from pdi.processos.cor import rgb_para_hsv

    img = _cubo_rgb()
    h, s, v = rgb_para_hsv(img)
    ref = rgb2hsv(img)  # H, S, V em [0, 1]
    np.testing.assert_allclose(h / 360.0, ref[..., 0], atol=1e-12)
    np.testing.assert_allclose(s, ref[..., 1], atol=1e-12)
    np.testing.assert_allclose(v, ref[..., 2], atol=1e-12)


def test_hsv_saidas_uint8_batem_com_skimage(img_rgb):
    from skimage.color import rgb2hsv

    res = executar_processo("decomposicao_hsv", img_rgb)
    assert [t for t, _ in res.saidas] == ["H (matiz)", "S (saturação)", "V (valor)"]
    ref = rgb2hsv(img_rgb) * 255.0
    for (_, nosso), canal in zip(res.saidas, range(3), strict=True):
        assert nosso.dtype == np.uint8 and nosso.shape == img_rgb.shape[:2]
        diferenca = np.abs(nosso.astype(int) - np.rint(ref[..., canal]).astype(int))
        assert diferenca.max() <= 1  # só arredondamento de empates (x,5)


@pytest.mark.parametrize(
    ("cor", "h_graus", "s", "v"),
    [
        ((255, 0, 0), 0, 1, 1),  # vermelho
        ((255, 255, 0), 60, 1, 1),  # amarelo
        ((0, 255, 0), 120, 1, 1),  # verde
        ((0, 255, 255), 180, 1, 1),  # ciano
        ((0, 0, 255), 240, 1, 1),  # azul
        ((255, 0, 255), 300, 1, 1),  # magenta
        ((128, 0, 0), 0, 1, 128 / 255),  # vermelho escuro: mesma H e S, V menor
        ((255, 128, 128), 0, 127 / 255, 1),  # rosa: vermelho "desbotado", S menor
        ((128, 128, 128), 0, 0, 128 / 255),  # cinza: sem matiz, S = 0
        ((255, 255, 255), 0, 0, 1),  # branco
        ((0, 0, 0), 0, 0, 0),  # preto: V = 0 e S = 0 (sem dividir por zero)
    ],
)
def test_hsv_cores_conhecidas(cor, h_graus, s, v):
    from pdi.processos.cor import rgb_para_hsv

    img = np.empty((1, 1, 3), np.uint8)
    img[...] = cor
    h_calc, s_calc, v_calc = (x[0, 0] for x in rgb_para_hsv(img))
    assert h_calc == pytest.approx(h_graus)
    assert s_calc == pytest.approx(s)
    assert v_calc == pytest.approx(v)


def test_hsv_escala_de_exibicao():
    img = np.array([[[0, 0, 255], [0, 255, 255]]], np.uint8)  # azul (240°), ciano (180°)
    h, s, v = (c for _, c in executar_processo("decomposicao_hsv", img).saidas)
    assert h[0].tolist() == [170, 128]  # 240·255/360 = 170; 180·255/360 = 127,5 → 128
    assert s[0].tolist() == [255, 255] and v[0].tolist() == [255, 255]


def test_hsv_v_e_o_maior_canal(img_rgb):
    _, _, v = (c for _, c in executar_processo("decomposicao_hsv", img_rgb).saidas)
    np.testing.assert_array_equal(v, img_rgb.max(axis=-1))


def test_hsv_h_sempre_menor_que_360():
    from pdi.processos.cor import rgb_para_hsv

    h, _, _ = rgb_para_hsv(_cubo_rgb(passo=3))
    assert h.min() >= 0 and h.max() < 360


def test_hsv_sem_avisos_de_divisao_por_zero():
    # Preto (V = 0) e cinzas (C = 0) não podem gerar NaN nem avisos do NumPy.
    from pdi.processos.cor import rgb_para_hsv

    img = np.array([[[0, 0, 0], [77, 77, 77]]], np.uint8)
    with np.errstate(all="raise"):
        h, s, v = rgb_para_hsv(img)
    assert np.isfinite(h).all() and np.isfinite(s).all()


def test_hsv_info():
    img = np.array([[[255, 0, 0], [128, 128, 128]]], np.uint8)
    info = executar_processo("decomposicao_hsv", img).info
    assert info["pixels sem matiz (cinza, S = 0)"] == "50.0%"
    assert info["S médio"] == 0.5


def test_hsv_recusa_imagem_cinza(img_cinza):
    with pytest.raises(ValueError, match="coloridas"):
        executar_processo("decomposicao_hsv", img_cinza)
