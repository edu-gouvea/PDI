"""Testes do visualizador e da janela principal (sem abrir janelas: modo offscreen)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pdi.core.imagem_io import AVISO_ALFA

IMAGENS = Path(__file__).resolve().parents[1] / "imagens_teste"


# ---------------------------------------------------------------- numpy → QImage


def test_qimage_cinza_preserva_pixels(qapp, img_cinza):
    from pdi.ui.visualizador import numpy_para_qimage

    q = numpy_para_qimage(img_cinza)
    assert (q.width(), q.height()) == (48, 32)
    for y, x in [(0, 0), (5, 20), (31, 47)]:
        v = int(img_cinza[y, x])
        assert q.pixelColor(x, y).getRgb()[:3] == (v, v, v)


def test_qimage_rgb_preserva_canais(qapp, img_rgb):
    from pdi.ui.visualizador import numpy_para_qimage

    q = numpy_para_qimage(img_rgb)
    for y, x in [(0, 0), (10, 30), (31, 47)]:
        assert q.pixelColor(x, y).getRgb()[:3] == tuple(int(c) for c in img_rgb[y, x])


def test_qimage_aceita_fatia_nao_contigua(qapp, img_rgb):
    from pdi.ui.visualizador import numpy_para_qimage

    canal = img_rgb[..., 2]
    q = numpy_para_qimage(canal)
    assert q.pixelColor(7, 3).red() == canal[3, 7]


def test_qimage_independente_do_array(qapp, img_cinza):
    from pdi.ui.visualizador import numpy_para_qimage

    arr = img_cinza.copy()
    q = numpy_para_qimage(arr)
    arr[:] = 0  # mudar o array depois não pode afetar a imagem exibida
    assert q.pixelColor(47, 0).red() == 255


# ------------------------------------------------------------------- visualizador


def test_visualizador_ajusta_mantendo_proporcao(qapp):
    from pdi.ui.visualizador import Visualizador

    vis = Visualizador("Teste")
    vis.resize(800, 400)
    vis.definir_imagem(np.zeros((100, 200), np.uint8))  # proporção 2:1
    vis.show()
    area = vis._tela.area_da_imagem()
    assert area.width() / area.height() == pytest.approx(2.0)
    assert area.width() <= vis._tela.width() and area.height() <= vis._tela.height()
    vis.grab()  # desenha sem erro
    vis.definir_imagem(None)
    assert vis.imagem is None
    vis.grab()


# ------------------------------------------------------------------ janela principal


@pytest.fixture
def janela(qapp, monkeypatch):
    from pdi.ui.janela_principal import JanelaPrincipal

    j = JanelaPrincipal()
    erros: list[str] = []
    monkeypatch.setattr(j, "_mostrar_erro", erros.append)  # não abre caixa de diálogo
    j.erros = erros
    yield j
    j.close()


def test_estado_inicial(janela):
    assert janela.entrada is None and janela.saida is None
    assert not janela.acao_salvar.isEnabled()
    assert not janela.acao_encadear.isEnabled()
    assert janela.vis_entrada.imagem is None


def test_abrir_imagem_cinza(janela):
    assert janela.abrir_arquivo(IMAGENS / "cameraman.png")
    assert janela.entrada.tipo == "cinza"
    assert janela.vis_entrada.imagem.shape == (512, 512)
    assert "Cinza 8 bits · 512×512" in janela.statusBar().currentMessage()
    assert "cameraman.png" in janela.windowTitle()
    assert janela.erros == []


def test_abrir_mostra_avisos_na_barra_de_status(janela, tmp_path):
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 128
    caminho = tmp_path / "transparente.png"
    Image.fromarray(rgba).save(caminho)
    assert janela.abrir_arquivo(caminho)
    assert AVISO_ALFA in janela.statusBar().currentMessage()


def test_abrir_arquivo_invalido_mostra_erro_e_mantem_estado(janela, tmp_path):
    janela.abrir_arquivo(IMAGENS / "astronauta.png")
    falso = tmp_path / "falso.png"
    falso.write_text("não é imagem")
    assert not janela.abrir_arquivo(falso)
    assert len(janela.erros) == 1 and "não é um arquivo PNG" in janela.erros[0]
    assert janela.entrada.caminho.name == "astronauta.png"  # nada mudou


def test_saida_habilita_salvar_e_salva(janela, tmp_path, img_rgb):
    janela.abrir_arquivo(IMAGENS / "astronauta.png")
    janela.definir_saida(img_rgb)
    assert janela.acao_salvar.isEnabled() and janela.acao_encadear.isEnabled()
    usado = janela.salvar_saida(tmp_path / "resultado")
    assert usado == tmp_path / "resultado.png"
    with Image.open(usado) as im:
        np.testing.assert_array_equal(np.asarray(im), img_rgb)


def test_abrir_nova_imagem_limpa_saida(janela, img_rgb):
    janela.abrir_arquivo(IMAGENS / "astronauta.png")
    janela.definir_saida(img_rgb)
    janela.abrir_arquivo(IMAGENS / "cafe.png")
    assert janela.saida is None and not janela.acao_salvar.isEnabled()


def test_usar_saida_como_entrada(janela, img_cinza):
    janela.abrir_arquivo(IMAGENS / "astronauta.png")
    janela.definir_saida(img_cinza)  # ex.: um canal da decomposição RGB
    janela.usar_saida_como_entrada()
    assert janela.entrada.tipo == "cinza"  # o tipo acompanha a nova entrada
    np.testing.assert_array_equal(janela.entrada.pixels, img_cinza)
    assert janela.saida is None and not janela.acao_salvar.isEnabled()


def test_salvar_com_erro_mostra_mensagem(janela, tmp_path, img_cinza):
    janela.abrir_arquivo(IMAGENS / "cameraman.png")
    janela.definir_saida(img_cinza)
    assert janela.salvar_saida(tmp_path / "nao" / "existe.png") is None
    assert "Não foi possível salvar" in janela.erros[0]


def test_atalhos_valem_em_qualquer_sistema(janela):
    # Os atalhos do README têm de existir mesmo onde o Qt não define um padrão (Windows).
    atalhos = {
        a.text(): a.shortcut().toString() for a in janela.findChildren(type(janela.acao_abrir))
    }
    assert atalhos["Sair"] == "Ctrl+Q"
    assert janela.acao_abrir.shortcut().toString() == "Ctrl+O"
    assert janela.acao_salvar.shortcut().toString() == "Ctrl+S"
    assert [s.toString() for s in janela.acao_aplicar.shortcuts()] == ["Ctrl+Return", "F5"]
