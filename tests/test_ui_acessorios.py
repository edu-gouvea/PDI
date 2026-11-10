"""Testes do painel de acessórios: info, imagens e histogramas."""

import subprocess
import sys

import numpy as np
import pytest

from pdi.core.contrato import Histograma, Resultado


@pytest.fixture
def painel(qapp):
    from pdi.ui.painel_acessorios import PainelAcessorios

    p = PainelAcessorios()
    p.resize(900, 220)
    p.show()
    yield p
    p.close()


def _resultado(acessorios=(), info=None):
    img = np.zeros((4, 4), np.uint8)
    return Resultado(saidas=[("s", img)], acessorios=list(acessorios), info=info or {})


def _hist(titulo="Antes", valores=None):
    if valores is None:
        valores = np.bincount(np.arange(256).repeat(3), minlength=256)
    return Histograma(titulo, np.asarray(valores))


def test_vazio(painel):
    painel.definir(None)
    assert painel.texto_info == "Nada a exibir" and painel.cartoes == []


def test_so_info(painel):
    painel.definir(_resultado(info={"limiar usado": 127, "semente usada": 3}))
    assert painel.texto_info == "limiar usado: 127   ·   semente usada: 3"
    assert painel.cartoes == []


def test_sem_info_nem_acessorios(painel):
    painel.definir(_resultado())
    assert painel.texto_info == "Este processo não tem dados extras."


def test_histogramas_e_imagem_viram_cartoes(painel):
    from pdi.ui.painel_acessorios import CartaoHistograma, CartaoImagem

    lap = np.tile(np.arange(64, dtype=np.uint8), (32, 1))
    painel.definir(_resultado([_hist("Antes"), _hist("Depois"), ("Laplaciano ajustado", lap)]))
    tipos = [type(c) for c in painel.cartoes]
    assert tipos == [CartaoHistograma, CartaoHistograma, CartaoImagem]
    assert [c.titulo for c in painel.cartoes] == ["Antes", "Depois", "Laplaciano ajustado"]
    np.testing.assert_array_equal(painel.cartoes[2].imagem, lap)
    for c in painel.cartoes:
        assert c.isVisible()
        c.grab()  # desenha sem erro (inclui o gráfico do Matplotlib)


def test_tooltip_do_histograma_mostra_valor_exato(painel):
    valores = np.zeros(256, dtype=np.int64)
    valores[128] = 1234
    painel.definir(_resultado([_hist("H", valores)]))
    cartao = painel.cartoes[0]
    assert cartao.texto_do_nivel(128) == "Nível 128: 1234 pixels"
    assert cartao.texto_do_nivel(300) == "Nível 255: 0 pixels"  # fora da faixa: limita


def test_histograma_de_probabilidades(painel):
    painel.definir(_resultado([_hist("p(r)", np.full(256, 1 / 256))]))
    assert painel.cartoes[0].texto_do_nivel(0) == "Nível 0: 0.0039"


def test_novo_resultado_remove_cartoes_antigos(painel):
    painel.definir(_resultado([_hist("Antes"), _hist("Depois")]))
    antigos = list(painel.cartoes)
    painel.definir(_resultado([("Lap", np.zeros((8, 8), np.uint8))]))
    assert len(painel.cartoes) == 1
    assert all(not c.isVisible() for c in antigos)


def test_formatacao_do_eixo():
    from pdi.ui.painel_acessorios import _formatar_eixo

    assert _formatar_eixo(12000) == "12k" and _formatar_eixo(1500) == "1.5k"
    assert _formatar_eixo(250) == "250"


def test_matplotlib_nao_carrega_na_abertura():
    # Importar a janela não pode puxar o Matplotlib (só o primeiro histograma puxa).
    codigo = "import sys, pdi.ui.janela_principal; print('matplotlib' in sys.modules)"
    saida = subprocess.run(
        [sys.executable, "-c", codigo], capture_output=True, text=True, check=True
    ).stdout
    assert saida.strip() == "False"


# ------------------------------------------------------------- altura na janela


def test_area_de_dados_cresce_com_cartoes_e_encolhe_sem(qapp, monkeypatch):
    import time
    from pathlib import Path

    from pdi.core import registro
    from pdi.core.contrato import ProcessoSpec
    from pdi.ui.janela_principal import ALTURA_ACESSORIOS_COM_CARTOES, JanelaPrincipal

    def com_hist(img, p):
        return _resultado([_hist()])

    spec = ProcessoSpec("teste_hist", "Teste hist", "Cor", 1, True, True, (), com_hist)
    monkeypatch.setitem(registro._PROCESSOS, spec.id, spec)
    j = JanelaPrincipal()
    j.resize(1280, 860)
    j.show()
    qapp.processEvents()

    def aplicar(id_processo):
        j.painel_processos.selecionar(id_processo)
        j.aplicar()
        while j.executor.ocupado:
            qapp.processEvents()
            time.sleep(0.005)
        qapp.processEvents()

    try:
        j.abrir_arquivo(Path(__file__).resolve().parents[1] / "imagens_teste" / "cameraman.png")
        baixa = j.caixa_acessorios.height()
        aplicar("teste_hist")
        assert j.caixa_acessorios.height() == ALTURA_ACESSORIOS_COM_CARTOES
        aplicar("limiarizacao")  # stub: só info, sem cartões
        assert j.caixa_acessorios.height() <= baixa + 5
    finally:
        j.close()  # encerra a thread de trabalho mesmo se uma verificação falhar


def test_cartoes_cabem_na_area_visivel(qapp):
    # Nada pode ficar cortado: a faixa de cartões não pode pedir mais altura que a área.
    from pdi.ui.painel_acessorios import PainelAcessorios

    painel = PainelAcessorios()
    painel.resize(1000, 190)
    painel.show()
    painel.definir(_resultado([_hist(), ("Lap", np.zeros((64, 64), np.uint8))]))
    qapp.processEvents()
    visivel = painel._rolagem.viewport().height()
    assert painel._faixa.minimumSizeHint().height() <= visivel
    assert painel._faixa.height() <= visivel
    painel.close()
