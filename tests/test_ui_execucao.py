"""Testes da execução em segundo plano (Aplicar), saídas múltiplas, erros e tempo."""

import time
from pathlib import Path

import numpy as np
import pytest

from pdi.core import registro
from pdi.core.contrato import Histograma, ProcessoSpec, Resultado
from pdi.core.registro import obter_processo

IMAGENS = Path(__file__).resolve().parents[1] / "imagens_teste"


def esperar(qapp, condicao, limite=5.0):
    """Processa eventos do Qt até ``condicao()`` ser verdadeira (ou estourar o limite)."""
    fim = time.monotonic() + limite
    while not condicao():
        if time.monotonic() > fim:
            raise TimeoutError("condição não ocorreu a tempo")
        qapp.processEvents()
        time.sleep(0.005)


# ------------------------------------------------------- processos falsos de teste


def _lento(img, p):
    time.sleep(0.3)
    return Resultado(saidas=[("Saída", img.copy())])


def _tres_saidas(img, p):
    canais = [(c, img[..., i].copy()) for i, c in enumerate("RGB")]
    hist = Histograma("Histograma R", np.bincount(img[..., 0].ravel(), minlength=256))
    return Resultado(saidas=canais, acessorios=[hist], info={"canais": 3})


def _erro_de_uso(img, p):
    raise ValueError("A deve ser menor ou igual a B.")


def _com_bug(img, p):
    return 1 / 0


def _spec(id_, funcao):
    return ProcessoSpec(id_, id_.replace("_", " ").title(), "Cor", 2, True, True, (), funcao)


FALSOS = {
    s.id: s
    for s in (
        _spec("teste_lento", _lento),
        _spec("teste_tres_saidas", _tres_saidas),
        _spec("teste_erro_de_uso", _erro_de_uso),
        _spec("teste_com_bug", _com_bug),
    )
}


# ------------------------------------------------------------- funções auxiliares


def test_formatar_tempo():
    from pdi.ui.janela_principal import formatar_tempo

    assert formatar_tempo(0.0123) == "12 ms"
    assert formatar_tempo(1.5) == "1,50 s"


def test_nome_sugerido():
    from pdi.ui.janela_principal import nome_sugerido

    ordem = obter_processo("filtro_ordem")
    assert (
        nome_sugerido("cameraman", ordem, {"tipo": "mínimo", "janela": 5})
        == "cameraman_filtro-ordem_minimo_janela5.png"
    )
    gauss = obter_processo("filtro_gaussiano")
    assert (
        nome_sugerido("x", gauss, {"sigma": 1.5, "janela": 7})
        == "x_filtro-gaussiano_sigma1.5_janela7.png"
    )
    rgb = obter_processo("decomposicao_rgb")
    assert nome_sugerido("astronauta", rgb, {}, "G") == "astronauta_decomposicao-rgb_g.png"


# ---------------------------------------------------------------------- executor


@pytest.fixture
def executor(qapp):
    from pdi.ui.worker import ExecutorDeProcessos

    ex = ExecutorDeProcessos()
    ex.resultados, ex.erros = [], []
    ex.concluido.connect(lambda r, s: ex.resultados.append((r, s)))
    ex.falhou.connect(ex.erros.append)
    yield ex
    ex.encerrar()


def test_executor_entrega_resultado_e_tempo(qapp, executor, img_cinza):
    assert executor.executar(obter_processo("limiarizacao"), img_cinza, {})
    esperar(qapp, lambda: executor.resultados)
    resultado, segundos = executor.resultados[0]
    assert isinstance(resultado, Resultado) and segundos >= 0
    assert not executor.ocupado


def test_executor_nao_bloqueia_a_interface(qapp, executor, img_cinza):
    inicio = time.monotonic()
    executor.executar(FALSOS["teste_lento"], img_cinza, {})
    assert time.monotonic() - inicio < 0.1  # voltou na hora; o trabalho está na outra thread
    assert executor.ocupado
    assert not executor.executar(FALSOS["teste_lento"], img_cinza, {})  # um por vez
    esperar(qapp, lambda: executor.resultados)
    assert len(executor.resultados) == 1


def test_executor_erro_de_uso_chega_como_veio(qapp, executor, img_cinza):
    executor.executar(FALSOS["teste_erro_de_uso"], img_cinza, {})
    esperar(qapp, lambda: executor.erros)
    assert executor.erros == ["A deve ser menor ou igual a B."]


def test_executor_bug_vira_mensagem_com_responsavel(qapp, executor, img_cinza, capsys):
    executor.executar(FALSOS["teste_com_bug"], img_cinza, {})
    esperar(qapp, lambda: executor.erros)
    assert "Erro inesperado" in executor.erros[0]
    assert "Integrante 2" in executor.erros[0] and "ZeroDivisionError" in executor.erros[0]
    assert "Traceback" in capsys.readouterr().err  # detalhes no terminal
    assert not executor.ocupado


def test_executor_descarta_resultado_antigo(qapp, executor, img_cinza):
    executor.executar(FALSOS["teste_lento"], img_cinza, {})
    executor.descartar_pendente()
    esperar(qapp, lambda: not executor.ocupado)
    qapp.processEvents()
    assert executor.resultados == []


# ------------------------------------------------------------------------ janela


@pytest.fixture
def janela(qapp, monkeypatch):
    for spec in FALSOS.values():
        monkeypatch.setitem(registro._PROCESSOS, spec.id, spec)
    from pdi.ui.janela_principal import JanelaPrincipal

    j = JanelaPrincipal()
    j.erros = []
    monkeypatch.setattr(j, "_mostrar_erro", j.erros.append)
    yield j
    j.close()


def _aplicar_e_esperar(qapp, janela, id_processo):
    assert janela.painel_processos.selecionar(id_processo)
    assert janela.aplicar()
    esperar(qapp, lambda: not janela.executor.ocupado)
    qapp.processEvents()


def test_aplicar_desabilitado_sem_imagem_ou_processo(janela):
    assert not janela.acao_aplicar.isEnabled() and not janela.aplicar()
    janela.abrir_arquivo(IMAGENS / "cameraman.png")
    assert not janela.acao_aplicar.isEnabled()  # ainda sem processo
    janela.painel_processos.selecionar("limiarizacao")
    assert janela.acao_aplicar.isEnabled()


def test_aplicar_mostra_saida_tempo_e_info(qapp, janela):
    janela.abrir_arquivo(IMAGENS / "cameraman.png")
    _aplicar_e_esperar(qapp, janela, "limiarizacao")  # stub: devolve a própria entrada
    np.testing.assert_array_equal(janela.saida, janela.entrada.pixels)
    assert "aplicado em" in janela.statusBar().currentMessage()
    assert janela.rotulo_tempo.text().startswith("Tempo: ")
    assert "stub" in janela._rotulo_acessorios.text()
    assert janela.acao_salvar.isEnabled() and not janela.progresso.isVisible()


def test_saidas_multiplas_viram_abas(qapp, janela, tmp_path):
    janela.abrir_arquivo(IMAGENS / "astronauta.png")
    _aplicar_e_esperar(qapp, janela, "teste_tres_saidas")
    assert janela.vis_saida.quantidade == 3
    np.testing.assert_array_equal(janela.saida, janela.entrada.pixels[..., 0])
    janela.vis_saida.selecionar(1)  # aba G
    np.testing.assert_array_equal(janela.saida, janela.entrada.pixels[..., 1])
    assert janela._sugestao_para_salvar().endswith("_g.png")
    usado = janela.salvar_saida(tmp_path / "g.png")
    from pdi.core.imagem_io import carregar_png

    np.testing.assert_array_equal(carregar_png(usado).pixels, janela.entrada.pixels[..., 1])
    assert "histograma: Histograma R" in janela._rotulo_acessorios.text()


def test_erro_mostra_mensagem_e_mantem_saida_anterior(qapp, janela):
    janela.abrir_arquivo(IMAGENS / "cameraman.png")
    _aplicar_e_esperar(qapp, janela, "limiarizacao")
    anterior = janela.saida
    _aplicar_e_esperar(qapp, janela, "teste_erro_de_uso")
    assert janela.erros == ["A deve ser menor ou igual a B."]
    assert janela.saida is anterior
    assert "limiarizacao" in janela._sugestao_para_salvar()  # nome ainda é o da saída exibida


def test_aplicar_bloqueado_durante_processamento(qapp, janela):
    janela.abrir_arquivo(IMAGENS / "cameraman.png")
    janela.painel_processos.selecionar("teste_lento")
    janela.aplicar()
    assert not janela.acao_aplicar.isEnabled() and janela.progresso.isVisibleTo(janela)
    esperar(qapp, lambda: not janela.executor.ocupado)
    assert janela.acao_aplicar.isEnabled() and not janela.progresso.isVisibleTo(janela)


def test_abrir_outra_imagem_descarta_resultado_em_andamento(qapp, janela):
    janela.abrir_arquivo(IMAGENS / "astronauta.png")
    janela.painel_processos.selecionar("teste_lento")
    janela.aplicar()
    janela.abrir_arquivo(IMAGENS / "cameraman.png")
    esperar(qapp, lambda: not janela.executor.ocupado)
    qapp.processEvents()
    assert janela.saida is None  # o resultado da astronauta não aparece sobre a cameraman


def test_entrada_e_saida_na_mesma_escala_com_abas(qapp, janela):
    janela.resize(1280, 820)
    janela.show()
    janela.abrir_arquivo(IMAGENS / "astronauta.png")  # quadrada: limitada pela altura
    _aplicar_e_esperar(qapp, janela, "teste_tres_saidas")
    qapp.processEvents()
    area_entrada = janela.vis_entrada._tela.area_da_imagem()
    area_saida = janela.vis_saida._tela.area_da_imagem()
    assert area_saida.width() == pytest.approx(area_entrada.width())
