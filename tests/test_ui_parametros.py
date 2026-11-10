"""Testes do painel de parâmetros gerado a partir do ProcessoSpec."""

import pytest

from pdi.core.registro import listar_processos, obter_processo, validar_parametros


@pytest.fixture
def painel(qapp):
    from pdi.ui.painel_parametros import PainelParametros

    p = PainelParametros()
    p.sinais = []
    p.valores_alterados.connect(lambda: p.sinais.append(1))
    return p


def _abrir(painel, id_processo):
    painel.definir_processo(obter_processo(id_processo))
    return painel


def _digitar(controle, texto):
    """Simula o usuário apagando o campo, digitando e apertando Enter."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    controle.spin.lineEdit().selectAll()
    QTest.keyClicks(controle.spin, texto)
    QTest.keyClick(controle.spin, Qt.Key.Key_Return)


# ------------------------------------------------------- todos os processos reais


@pytest.mark.parametrize("spec", listar_processos(), ids=lambda s: s.id)
def test_padroes_gerados_passam_na_validacao_do_registro(painel, spec):
    painel.definir_processo(spec)
    valores = painel.valores()
    assert set(valores) == {p.nome for p in spec.parametros}
    # Os padrões que a interface monta são exatamente os declarados no spec...
    assert valores == {p.nome: p.padrao for p in spec.parametros}
    # ...e passam na mesma validação que executar_processo faz.
    assert validar_parametros(spec, valores) == valores


def test_tipo_de_controle_por_tipo_de_parametro(painel):
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox

    _abrir(painel, "filtro_ordem")  # ESCOLHA + INT
    assert isinstance(painel.controle("tipo").combo, QComboBox)
    assert isinstance(painel.controle("janela").spin, QSpinBox)
    _abrir(painel, "filtro_gaussiano")  # FLOAT
    assert isinstance(painel.controle("sigma").spin, QDoubleSpinBox)


def test_processo_sem_parametros(painel):
    _abrir(painel, "equalizacao_histograma")
    assert painel.valores() == {}
    assert not painel.botao_padroes.isEnabled()


def test_sem_processo(painel):
    painel.definir_processo(None)
    assert painel.processo is None and painel.valores() == {}


# --------------------------------------------------------- precisão dos valores


def test_valor_fora_da_grade_do_slider_e_mantido(painel):
    _abrir(painel, "transformacao_potencia")
    painel.controle("gama").definir(0.04)  # slider anda de 0,05 em 0,05
    assert painel.valores()["gama"] == pytest.approx(0.04)


def test_aceita_virgula_e_ponto(painel):
    _abrir(painel, "filtro_gaussiano")
    sigma = painel.controle("sigma")
    _digitar(sigma, "0,25")
    assert painel.valores()["sigma"] == pytest.approx(0.25)  # 2 casas: não vira 0,3
    _digitar(sigma, "2.75")
    assert painel.valores()["sigma"] == pytest.approx(2.75)
    assert sigma.spin.text() == "2.75"  # sempre mostra ponto


def test_padrao_com_duas_casas_aparece_exato(painel):
    _abrir(painel, "transformacao_log")
    assert painel.valores()["c"] == pytest.approx(45.99)


def test_valor_digitado_sem_enter_e_considerado(painel):
    from PySide6.QtTest import QTest

    _abrir(painel, "limiarizacao")
    k = painel.controle("k")
    k.spin.lineEdit().selectAll()
    QTest.keyClicks(k.spin, "200")  # sem Enter: ainda não confirmado pelo Qt
    assert painel.valores()["k"] == 200


# ------------------------------------------------------------------ janela ímpar


@pytest.mark.parametrize(("digitado", "esperado"), [(4, 5), (7, 7), (30, 31), (3, 3)])
def test_janela_par_vira_impar(painel, digitado, esperado):
    _abrir(painel, "filtro_ordem")
    _digitar(painel.controle("janela"), str(digitado))
    assert painel.valores()["janela"] == esperado


def test_slider_da_janela_so_passa_por_impares(painel):
    _abrir(painel, "filtro_ordem")
    janela = painel.controle("janela")
    janela.slider.setValue(2)  # posição 0 = 3, 1 = 5, 2 = 7
    assert painel.valores()["janela"] == 7
    assert janela.spin.singleStep() == 2


# ------------------------------------------------------------- slider ↔ caixa


def test_slider_e_caixa_sincronizados(painel):
    _abrir(painel, "limiarizacao")
    k = painel.controle("k")
    k.spin.setValue(200)
    assert k.slider.value() == 200
    k.slider.setValue(50)
    assert painel.valores()["k"] == 50


def test_slider_de_gama_passa_por_1_exato(painel):
    _abrir(painel, "transformacao_potencia")
    gama = painel.controle("gama")
    gama.slider.setValue(20)  # 20 × 0,05
    assert painel.valores()["gama"] == 1.0


def test_semente_nao_tem_slider(painel):
    _abrir(painel, "ruido_gaussiano")
    assert painel.controle("semente").slider is None


def test_tooltip_mostra_ajuda_e_faixa(painel):
    _abrir(painel, "filtro_ordem")
    dica = painel.controle("janela").spin.toolTip()
    assert "Faixa: 3 a 31 (somente ímpares)" in dica


# ------------------------------------------------------- memória e padrões


def test_lembra_valores_ao_trocar_de_processo(painel):
    _abrir(painel, "filtro_gaussiano").controle("sigma").definir(3.5)
    _abrir(painel, "limiarizacao")
    _abrir(painel, "filtro_gaussiano")
    assert painel.valores()["sigma"] == pytest.approx(3.5)


def test_restaurar_padroes(painel):
    _abrir(painel, "ruido_sal_pimenta")
    painel.definir_valores({"tipo": "sal", "probabilidade": 0.3})
    painel.botao_padroes.click()
    assert painel.valores()["tipo"] == "sal e pimenta"
    assert painel.valores()["probabilidade"] == pytest.approx(0.05)


def test_emite_sinal_ao_mudar(painel):
    _abrir(painel, "fatiamento_intensidade")
    painel.sinais.clear()
    painel.controle("fundo").definir("zerar")
    painel.controle("a").definir(10)
    assert len(painel.sinais) >= 2


def test_grade_antiga_some_ao_trocar_de_processo(painel):
    from PySide6.QtWidgets import QWidget

    painel.show()
    _abrir(painel, "ruido_gaussiano")
    antigos = list(painel.controle("media").widgets)
    _abrir(painel, "filtro_butterworth")
    # Os controles do processo anterior não podem continuar visíveis por baixo dos novos.
    assert all(not w.isVisible() for w in antigos)
    visiveis = [w for w in painel.findChildren(QWidget) if w.isVisible()]
    assert painel.controle("ordem").spin in visiveis
