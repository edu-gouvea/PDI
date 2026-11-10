"""Testes da lista de processos (habilitação por tipo de imagem) e da integração na janela."""

from pathlib import Path

import pytest

from pdi.core.registro import GRUPOS, listar_processos, processos_por_grupo

IMAGENS = Path(__file__).resolve().parents[1] / "imagens_teste"
SO_RGB = {"decomposicao_rgb", "decomposicao_hsv"}
SO_CINZA = {"limiarizacao", "equalizacao_histograma", "fatiamento_intensidade"}


@pytest.fixture
def painel(qapp):
    from pdi.ui.painel_processos import PainelProcessos

    p = PainelProcessos(processos_por_grupo())
    p.emitidos = []
    p.processo_selecionado.connect(p.emitidos.append)
    return p


def _habilitados(painel) -> set[str]:
    return {s.id for s in listar_processos() if painel.esta_habilitado(s.id)}


def test_arvore_tem_grupos_e_processos_na_ordem(painel):
    arvore = painel.arvore
    grupos = [arvore.topLevelItem(i).text(0) for i in range(arvore.topLevelItemCount())]
    assert grupos == [g for g in GRUPOS if g in processos_por_grupo()]
    nomes = [
        arvore.topLevelItem(i).child(j).text(0)
        for i in range(arvore.topLevelItemCount())
        for j in range(arvore.topLevelItem(i).childCount())
    ]
    assert nomes == [s.nome for s in listar_processos()]  # os 17, na ordem do PDF


def test_sem_imagem_todos_habilitados(painel):
    assert _habilitados(painel) == {s.id for s in listar_processos()}


def test_imagem_rgb_desabilita_os_so_cinza(painel):
    painel.definir_tipo_imagem("rgb")
    desabilitados = {s.id for s in listar_processos()} - _habilitados(painel)
    assert desabilitados == SO_CINZA
    dica = painel.item_do_processo("equalizacao_histograma").toolTip(0)
    assert "apenas para imagens em escala de cinza" in dica


def test_imagem_cinza_desabilita_os_so_rgb(painel):
    painel.definir_tipo_imagem("cinza")
    desabilitados = {s.id for s in listar_processos()} - _habilitados(painel)
    assert desabilitados == SO_RGB
    assert "coloridas" in painel.item_do_processo("decomposicao_rgb").toolTip(0)


def test_tooltip_de_habilitado_e_a_descricao(painel):
    spec = next(s for s in listar_processos() if s.id == "filtro_gaussiano")
    assert painel.item_do_processo("filtro_gaussiano").toolTip(0) == spec.descricao


def test_selecionar_emite_o_spec(painel):
    assert painel.selecionar("filtro_ordem")
    assert painel.processo_atual.id == "filtro_ordem"
    assert painel.emitidos[-1].id == "filtro_ordem"


def test_nao_seleciona_desabilitado(painel):
    painel.definir_tipo_imagem("rgb")
    assert not painel.selecionar("equalizacao_histograma")
    assert painel.processo_atual is None


def test_selecao_limpa_quando_deixa_de_se_aplicar(painel):
    painel.selecionar("equalizacao_histograma")
    painel.definir_tipo_imagem("rgb")  # abriu uma colorida
    assert painel.processo_atual is None
    assert painel.emitidos[-1] is None


def test_selecao_mantida_quando_continua_valida(painel):
    painel.selecionar("filtro_gaussiano")
    painel.definir_tipo_imagem("rgb")
    painel.definir_tipo_imagem("cinza")
    assert painel.processo_atual.id == "filtro_gaussiano"


def test_clicar_no_grupo_mantem_o_processo(painel):
    painel.selecionar("limiarizacao")
    cabecalho = painel.arvore.topLevelItem(0)
    painel.arvore.setCurrentItem(cabecalho)
    assert painel.processo_atual.id == "limiarizacao"
    assert painel.emitidos[-1].id == "limiarizacao"


# --------------------------------------------------------------- integração na janela


@pytest.fixture
def janela(qapp, monkeypatch):
    from pdi.ui.janela_principal import JanelaPrincipal

    j = JanelaPrincipal()
    monkeypatch.setattr(j, "_mostrar_erro", lambda msg: pytest.fail(msg))
    yield j
    j.close()


def test_abrir_rgb_desabilita_equalizacao(janela):
    janela.abrir_arquivo(IMAGENS / "astronauta.png")
    assert not janela.painel_processos.esta_habilitado("equalizacao_histograma")
    assert janela.painel_processos.esta_habilitado("decomposicao_hsv")


def test_selecao_na_lista_chega_na_janela(janela):
    janela.painel_processos.selecionar("filtro_butterworth")
    assert janela.processo.id == "filtro_butterworth"
    assert "Butterworth" in janela._rotulo_parametros.text()


def test_encadear_canal_cinza_habilita_equalizacao(janela):
    janela.abrir_arquivo(IMAGENS / "astronauta.png")
    janela.definir_saida(janela.entrada.pixels[..., 0].copy())  # um canal: cinza
    janela.usar_saida_como_entrada()
    assert janela.painel_processos.esta_habilitado("equalizacao_histograma")
    assert not janela.painel_processos.esta_habilitado("decomposicao_rgb")
