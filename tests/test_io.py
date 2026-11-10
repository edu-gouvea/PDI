"""Testes de carregar_png e salvar_png (core/imagem_io.py)."""

import cv2
import numpy as np
import pytest
from PIL import Image

from pdi.core.imagem_io import (
    AVISO_1_BIT,
    AVISO_16_BITS,
    AVISO_ALFA,
    AVISO_RGB_CINZA,
    carregar_png,
    salvar_png,
)
from pdi.core.util import TIPO_CINZA, TIPO_RGB


def _salvar_pil(tmp_path, nome, im: Image.Image, **kwargs):
    caminho = tmp_path / nome
    im.save(caminho, format=kwargs.pop("format", "PNG"), **kwargs)
    return caminho


# ------------------------------------------------------------ ida e volta sem perda


def test_ida_e_volta_cinza(tmp_path, img_cinza):
    caminho = salvar_png(img_cinza, tmp_path / "cinza.png")
    lida = carregar_png(caminho)
    assert lida.tipo == TIPO_CINZA and lida.avisos == []
    assert lida.pixels.dtype == np.uint8
    np.testing.assert_array_equal(lida.pixels, img_cinza)


def test_ida_e_volta_rgb(tmp_path, img_rgb):
    caminho = salvar_png(img_rgb, tmp_path / "rgb.png")
    lida = carregar_png(caminho)
    assert lida.tipo == TIPO_RGB and lida.avisos == []
    np.testing.assert_array_equal(lida.pixels, img_rgb)


def test_ordem_dos_canais_igual_ao_opencv(tmp_path, img_rgb):
    # OpenCV lê em BGR; invertendo os canais, tem de bater com o nosso RGB.
    caminho = salvar_png(img_rgb, tmp_path / "rgb.png")
    ref = cv2.cvtColor(cv2.imread(str(caminho), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(carregar_png(caminho).pixels, ref)


def test_pixel_vermelho_fica_no_canal_r(tmp_path):
    img = np.zeros((2, 2, 3), np.uint8)
    img[..., 0] = 255
    lida = carregar_png(salvar_png(img, tmp_path / "vermelho.png"))
    assert lida.pixels[0, 0].tolist() == [255, 0, 0]


def test_resumo_para_barra_de_status(tmp_path):
    img = np.zeros((384, 512, 3), np.uint8)
    img[0, 0] = (1, 2, 3)  # canais diferentes, para não virar cinza
    assert carregar_png(salvar_png(img, tmp_path / "x.png")).resumo == "RGB 24 bits · 512×384"


# --------------------------------------------------------------------- transparência


def test_rgba_composto_sobre_branco(tmp_path):
    rgba = np.zeros((1, 3, 4), np.uint8)
    rgba[..., 0] = 200  # cor = (200, 0, 0)
    rgba[0, :, 3] = [0, 255, 51]  # alfa: transparente, opaco, 20%
    lida = carregar_png(_salvar_pil(tmp_path, "rgba.png", Image.fromarray(rgba)))
    assert lida.tipo == TIPO_RGB and AVISO_ALFA in lida.avisos
    assert lida.pixels[0, 0].tolist() == [255, 255, 255]  # transparente → branco
    assert lida.pixels[0, 1].tolist() == [200, 0, 0]  # opaco → a própria cor
    # 20%: 0,2·200 + 0,8·255 = 244 e 0,2·0 + 0,8·255 = 204
    assert lida.pixels[0, 2].tolist() == [244, 204, 204]


def test_la_vira_cinza_composto(tmp_path):
    im = Image.new("LA", (1, 1), (100, 128))
    lida = carregar_png(_salvar_pil(tmp_path, "la.png", im))
    assert lida.tipo == TIPO_CINZA and AVISO_ALFA in lida.avisos
    # a = 128/255: a·100 + (1 − a)·255 = 177,2 → 177
    assert lida.pixels[0, 0] == 177


# --------------------------------------------------------------------------- paleta


def test_paleta_colorida_vira_rgb(tmp_path):
    im = Image.new("P", (2, 1))
    im.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
    im.putpixel((1, 0), 1)
    lida = carregar_png(_salvar_pil(tmp_path, "p.png", im))
    assert lida.tipo == TIPO_RGB and lida.avisos == []
    assert lida.pixels.tolist() == [[[255, 0, 0], [0, 0, 255]]]


def test_paleta_de_cinzas_vira_cinza(tmp_path):
    im = Image.new("P", (2, 1))
    im.putpalette([10, 10, 10, 200, 200, 200] + [0] * 762)
    im.putpixel((1, 0), 1)
    lida = carregar_png(_salvar_pil(tmp_path, "pg.png", im))
    assert lida.tipo == TIPO_CINZA
    assert lida.pixels.tolist() == [[10, 200]]


def test_paleta_com_transparencia(tmp_path):
    im = Image.new("P", (2, 1))
    im.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
    im.putpixel((1, 0), 1)
    lida = carregar_png(_salvar_pil(tmp_path, "pt.png", im, transparency=0))
    assert AVISO_ALFA in lida.avisos
    assert lida.pixels.tolist() == [[[255, 255, 255], [0, 0, 255]]]


# ------------------------------------------------------------ outras profundidades


def test_cinza_16_bits_reescalado(tmp_path):
    a16 = np.array([[0, 257 * 100, 65535, 128]], dtype=np.uint16)
    lida = carregar_png(_salvar_pil(tmp_path, "g16.png", Image.fromarray(a16)))
    assert lida.tipo == TIPO_CINZA and AVISO_16_BITS in lida.avisos
    assert lida.pixels.tolist() == [[0, 100, 255, 0]]  # 128/257 = 0,498 → 0


def test_rgb_16_bits_avisa(tmp_path):
    # O Pillow converte RGB 16 bits para 8 sem avisar; detectamos pelo cabeçalho.
    a16 = np.zeros((2, 2, 3), np.uint16)
    a16[..., 0] = 65535
    caminho = tmp_path / "rgb16.png"
    cv2.imwrite(str(caminho), a16[..., ::-1])  # OpenCV grava BGR
    lida = carregar_png(caminho)
    assert lida.tipo == TIPO_RGB and AVISO_16_BITS in lida.avisos
    assert lida.pixels[0, 0].tolist() == [255, 0, 0]


def test_1_bit_vira_0_e_255(tmp_path):
    im = Image.fromarray(np.array([[False, True]]))
    lida = carregar_png(_salvar_pil(tmp_path, "b1.png", im))
    assert lida.tipo == TIPO_CINZA and AVISO_1_BIT in lida.avisos
    assert lida.pixels.tolist() == [[0, 255]]


# --------------------------------------------------------------- RGB com canais iguais


def test_rgb_com_canais_iguais_vira_cinza(tmp_path, img_cinza):
    rgb = np.dstack([img_cinza] * 3)
    lida = carregar_png(salvar_png(rgb, tmp_path / "falso_rgb.png"))
    assert lida.tipo == TIPO_CINZA and AVISO_RGB_CINZA in lida.avisos
    np.testing.assert_array_equal(lida.pixels, img_cinza)


def test_rgb_com_canais_iguais_pode_continuar_rgb(tmp_path, img_cinza):
    rgb = np.dstack([img_cinza] * 3)
    lida = carregar_png(salvar_png(rgb, tmp_path / "x.png"), rgb_igual_vira_cinza=False)
    assert lida.tipo == TIPO_RGB and lida.avisos == []


# ---------------------------------------------------------------------- recusas


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ValueError, match="não encontrado"):
        carregar_png(tmp_path / "nao_existe.png")


def test_jpeg_renomeado_para_png(tmp_path):
    caminho = _salvar_pil(tmp_path, "falso.png", Image.new("RGB", (4, 4)), format="JPEG")
    with pytest.raises(ValueError, match="não é um arquivo PNG"):
        carregar_png(caminho)


def test_arquivo_de_texto(tmp_path):
    caminho = tmp_path / "texto.png"
    caminho.write_text("isto não é uma imagem")
    with pytest.raises(ValueError, match="não é um arquivo PNG"):
        carregar_png(caminho)


def test_png_corrompido(tmp_path, img_rgb):
    bytes_png = salvar_png(img_rgb, tmp_path / "ok.png").read_bytes()
    caminho = tmp_path / "cortado.png"
    caminho.write_bytes(bytes_png[: len(bytes_png) // 2])  # metade do arquivo
    with pytest.raises(ValueError, match="corrompido"):
        carregar_png(caminho)


def test_pasta_em_vez_de_arquivo(tmp_path):
    with pytest.raises(ValueError, match="não encontrado"):
        carregar_png(tmp_path)


# ------------------------------------------------------------------------- salvar


def test_salvar_acrescenta_extensao(tmp_path, img_cinza):
    assert salvar_png(img_cinza, tmp_path / "saida").name == "saida.png"
    assert salvar_png(img_cinza, tmp_path / "saida.PNG").name == "saida.PNG"
    assert salvar_png(img_cinza, tmp_path / "foto.jpg").name == "foto.jpg.png"


def test_salvar_grava_png_de_8_bits_no_modo_certo(tmp_path, img_cinza, img_rgb):
    with Image.open(salvar_png(img_cinza, tmp_path / "c.png")) as im:
        assert (im.format, im.mode) == ("PNG", "L")
    with Image.open(salvar_png(img_rgb, tmp_path / "r.png")) as im:
        assert (im.format, im.mode) == ("PNG", "RGB")


def test_salvar_aceita_array_nao_contiguo(tmp_path, img_rgb):
    canal = img_rgb[..., 1]  # fatia: não contígua na memória
    np.testing.assert_array_equal(carregar_png(salvar_png(canal, tmp_path / "g.png")).pixels, canal)


@pytest.mark.parametrize(
    "ruim",
    [np.zeros((4, 4), np.float64), np.zeros((4, 4, 4), np.uint8), np.zeros((0, 4), np.uint8)],
)
def test_salvar_recusa_formato_invalido(tmp_path, ruim):
    with pytest.raises(ValueError):
        salvar_png(ruim, tmp_path / "x.png")


def test_salvar_em_pasta_inexistente(tmp_path, img_cinza):
    with pytest.raises(ValueError, match="Não foi possível salvar"):
        salvar_png(img_cinza, tmp_path / "nao" / "existe" / "x.png")
