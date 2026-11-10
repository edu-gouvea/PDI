"""Testes da pré-visualização automática (debounce, último valor vence, lentos, erros)."""

import time
from pathlib import Path

import numpy as np
import pytest

from pdi.core import registro
from pdi.core.contrato import Param, ProcessoSpec, Resultado, TipoParam

IMAGENS = Path(__file__).resolve().parents[1] / "imagens_teste"
K = Param("k", "k", TipoParam.INT, 0, minimo=0, maximo=100, passo=1, ajuda="teste")


def esperar(qapp, condicao, limite=5.0):
    fim = time.monotonic() + limite
    while not condicao():
        if time.monotonic() > fim:
            raise TimeoutError("condição não ocorreu a tempo")
        qapp.processEvents()
        time.sleep(0.005)


def ocioso(qapp, segundos):
    """Processa eventos por um tempo (deixa timers dispararem)."""
    fim = time.monotonic() + segundos
    while time.monotonic() < fim:
        qapp.processEvents()
        time.sleep(0.005)


class Contador:
    """Processo falso que registra cada execução e o k recebido."""

    def __init__(self, atraso=0.0, erro=False):
        self.ks: list[int] = []
        self.atraso, self.erro = atraso, erro

    def __call__(self, img, p):
        time.sleep(self.atraso)
        self.ks.append(p["k"])
        if self.erro:
            raise ValueError("k inválido para este teste")
        return Resultado(saidas=[("s", img.copy())], info={"k": p["k"]})


@pytest.fixture
def janela(qapp, monkeypatch):
    rapido, lento, com_erro = Contador(), Contador(atraso=0.25), Contador(erro=True)
    muito_lento = Contador(atraso=0.6)
    processos = (
        ("auto_rapido", rapido),
        ("auto_lento", lento),
        ("auto_muito_lento", muito_lento),
        ("auto_erro", com_erro),
    )
    for id_, f in processos:
        spec = ProcessoSpec(id_, id_, "Cor", 1, True, True, (K,), f)
        monkeypatch.setitem(registro._PROCESSOS, id_, spec)
    from pdi.ui.janela_principal import JanelaPrincipal

    j = JanelaPrincipal()
    j.erros = []
    monkeypatch.setattr(j, "_mostrar_erro", j.erros.append)
    j.contadores = {"rapido": rapido, "lento": lento, "muito_lento": muito_lento, "erro": com_erro}
    j.abrir_arquivo(IMAGENS / "cameraman.png")
    yield j
    j.close()


def _mudar_k(janela, valor):
    janela.painel_parametros.controle("k").definir(valor)


def _rodar_ate_quieto(qapp, janela, limite=10.0):
    """Espera o debounce e TODAS as execuções (inclusive a reaplicação pendente) acabarem.

    "Livre" por um instante não basta: a reaplicação pendente é agendada para o ciclo
    seguinte (singleShot(0)). Por isso exige 150 ms seguidos sem nada acontecendo.
    """
    ocioso(qapp, 0.35)
    fim = time.monotonic() + limite
    quieto_desde = None
    while time.monotonic() < fim:
        qapp.processEvents()
        ativo = (
            janela.executor.ocupado
            or janela._timer_automatico.isActive()
            or janela._automatico_pendente
        )
        agora = time.monotonic()
        if ativo:
            quieto_desde = None
        elif quieto_desde is None:
            quieto_desde = agora
        elif agora - quieto_desde > 0.15:
            return
        time.sleep(0.005)
    raise TimeoutError("o processamento não sossegou a tempo")


def test_desligado_nao_aplica_sozinho(qapp, janela):
    janela.painel_processos.selecionar("auto_rapido")
    _mudar_k(janela, 5)
    ocioso(qapp, 0.4)
    assert janela.contadores["rapido"].ks == [] and janela.saida is None


def test_ligar_aplica_e_mudanca_reaplica(qapp, janela):
    janela.painel_processos.selecionar("auto_rapido")
    janela.caixa_automatico.setChecked(True)
    _rodar_ate_quieto(qapp, janela)
    assert janela.contadores["rapido"].ks == [0] and janela.saida is not None
    _mudar_k(janela, 7)
    _rodar_ate_quieto(qapp, janela)
    assert janela.contadores["rapido"].ks == [0, 7]


def test_debounce_varias_mudancas_rapidas_rodam_uma_vez(qapp, janela):
    janela.caixa_automatico.setChecked(True)
    janela.painel_processos.selecionar("auto_rapido")
    for k in range(1, 30):  # como arrastar um slider
        _mudar_k(janela, k)
    _rodar_ate_quieto(qapp, janela)
    assert janela.contadores["rapido"].ks == [29]  # só o valor final


def test_ultimo_valor_vence_durante_processamento(qapp, janela):
    janela.caixa_automatico.setChecked(True)
    janela.painel_processos.selecionar("auto_muito_lento")  # 0,6 s por execução
    ocioso(qapp, 0.3)  # debounce dispara: começa a rodar com k = 0
    assert janela.executor.ocupado
    for k in (3, 6, 9):  # três mudanças rápidas, todas DURANTE a execução
        _mudar_k(janela, k)
        ocioso(qapp, 0.05)
    _rodar_ate_quieto(qapp, janela)
    # Exatamente 2 execuções: a inicial e UMA com o valor final (3 e 6 nunca rodam).
    assert janela.contadores["muito_lento"].ks == [0, 9]
    assert janela.painel_acessorios.texto_info == "k: 9"


def test_processo_lento_pausa_o_automatico(qapp, janela, monkeypatch):
    import pdi.ui.janela_principal as jp

    monkeypatch.setattr(jp, "LIMITE_AUTOMATICO_S", 0.1)
    janela.caixa_automatico.setChecked(True)
    janela.painel_processos.selecionar("auto_lento")
    _rodar_ate_quieto(qapp, janela)  # 1ª execução: tempo ainda desconhecido, roda
    _mudar_k(janela, 4)
    _rodar_ate_quieto(qapp, janela)
    assert janela.contadores["lento"].ks == [0]  # a 2ª não rodou sozinha
    assert "pausada" in janela.statusBar().currentMessage()
    janela.aplicar()  # o Aplicar manual continua funcionando
    _rodar_ate_quieto(qapp, janela)
    assert janela.contadores["lento"].ks == [0, 4]


def test_erro_no_automatico_nao_abre_caixa_de_aviso(qapp, janela):
    janela.caixa_automatico.setChecked(True)
    janela.painel_processos.selecionar("auto_erro")
    _rodar_ate_quieto(qapp, janela)
    assert janela.erros == []
    assert "k inválido" in janela.statusBar().currentMessage()
    janela.caixa_automatico.setChecked(False)
    janela.aplicar()  # no Aplicar manual, o aviso aparece
    _rodar_ate_quieto(qapp, janela)
    assert janela.erros == ["k inválido para este teste"]


def test_trocar_imagem_reaplica(qapp, janela):
    janela.caixa_automatico.setChecked(True)
    janela.painel_processos.selecionar("auto_rapido")
    _rodar_ate_quieto(qapp, janela)
    janela.abrir_arquivo(IMAGENS / "astronauta.png")
    _rodar_ate_quieto(qapp, janela)
    assert len(janela.contadores["rapido"].ks) == 2
    np.testing.assert_array_equal(janela.saida, janela.entrada.pixels)


def test_trocar_imagem_durante_processamento_reaplica_na_nova(qapp, janela):
    # O resultado da imagem antiga é descartado; a pendência tem de rodar na nova.
    janela.caixa_automatico.setChecked(True)
    janela.painel_processos.selecionar("auto_lento")
    ocioso(qapp, 0.3)
    assert janela.executor.ocupado
    janela.abrir_arquivo(IMAGENS / "astronauta.png")
    _rodar_ate_quieto(qapp, janela)
    assert janela.saida is not None
    np.testing.assert_array_equal(janela.saida, janela.entrada.pixels)
