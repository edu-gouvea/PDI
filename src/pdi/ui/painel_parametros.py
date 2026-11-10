"""Painel de parâmetros gerado automaticamente a partir de ProcessoSpec.parametros (I1).

Cada ``Param`` vira um controle, sem nenhum código específico por processo:

- INT / FLOAT → caixa numérica (valor exato, digitável) + slider sincronizado
- BOOL → caixa de seleção
- ESCOLHA → lista suspensa

Regras pensadas para a apresentação, em que o professor dita os valores:

- A caixa numérica é a fonte da verdade; o slider é só atalho. Um valor digitado fora
  da grade do slider (γ = 0,04 com passo 0,05) é mantido exatamente.
- Decimais aceitam vírgula ou ponto ("0,5" e "0.5") e sempre mostram ponto. Num Qt em
  português, "0.5" poderia ser lido como separador de milhar.
- FLOAT tem pelo menos 2 casas: com 1 casa, σ = 0,25 viraria 0,3.
- Janelas (``somente_impar``) andam de 2 em 2; um valor par digitado vira o ímpar seguinte.
- O valor só é confirmado ao apertar Enter, sair do campo ou usar as setas (digitar
  "15" não passa por "1" no meio do caminho).
"""

import math
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pdi.core.contrato import Param, ProcessoSpec, TipoParam

_MAX_POSICOES_SLIDER = 1000  # acima disso (ex.: semente 0–999999) o slider não ajuda


def _casas_decimais(p: Param) -> int:
    """Casas necessárias para mostrar exatamente passo, limites e padrão (entre 2 e 4)."""

    def casas(x: float | None) -> int:
        if x is None:
            return 0
        texto = f"{x:.6f}".rstrip("0")
        return len(texto.split(".")[1])

    return min(4, max(2, *(casas(x) for x in (p.passo, p.minimo, p.maximo, p.padrao))))


def _faixa_texto(p: Param) -> str:
    def fmt(x: float) -> str:
        return f"{x:g}"

    texto = f"Faixa: {fmt(p.minimo)} a {fmt(p.maximo)}"
    return texto + " (somente ímpares)" if p.somente_impar else texto


class _SpinDecimal(QDoubleSpinBox):
    """QDoubleSpinBox que aceita vírgula ou ponto e sempre mostra ponto."""

    def __init__(self) -> None:
        super().__init__()
        self.setLocale(QLocale.c())  # ponto decimal, sem separador de milhar

    def validate(self, texto: str, pos: int):
        return super().validate(texto.replace(",", "."), pos)

    def valueFromText(self, texto: str) -> float:  # noqa: N802 (nome do Qt)
        return super().valueFromText(texto.replace(",", "."))


# ------------------------------------------------------------------------ controles


class _Controle:
    """Interface comum: ``widgets`` (para o layout), ``valor()`` e ``definir(v)``."""

    widgets: list[QWidget]

    def valor(self) -> Any: ...

    def definir(self, valor: Any) -> None: ...


class _ControleNumerico(_Controle):
    def __init__(self, p: Param, ao_mudar: Callable[[], None]) -> None:
        self._p = p
        self._ao_mudar = ao_mudar
        self._inteiro = p.tipo is TipoParam.INT

        if self._inteiro:
            self.spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            self.spin.setRange(int(p.minimo), int(p.maximo))
            passo: float = 2 if p.somente_impar else int(p.passo or 1)
        else:
            self.spin = _SpinDecimal()
            self.spin.setDecimals(_casas_decimais(p))
            self.spin.setRange(p.minimo, p.maximo)
            passo = p.passo or (p.maximo - p.minimo) / 100
        self.spin.setSingleStep(passo)
        self.spin.setKeyboardTracking(False)
        self.spin.setAccelerated(True)
        self.spin.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
        self.spin.setMinimumWidth(90)

        # Grade do slider. Janela: ímpares a partir do mínimo. Demais: múltiplos do passo
        # (assim γ passa por 1,00 exato, e não por 1,01).
        if p.somente_impar:
            self._base, self._passo_slider = (p.minimo if p.minimo % 2 else p.minimo + 1), 2
            indices = (0, int((p.maximo - self._base) // 2))
        else:
            self._base, self._passo_slider = 0.0, passo
            indices = (math.ceil(p.minimo / passo - 1e-9), math.floor(p.maximo / passo + 1e-9))

        self.slider: QSlider | None = None
        if indices[1] - indices[0] <= _MAX_POSICOES_SLIDER:
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(*indices)
            self.slider.setMinimumWidth(120)
            self.slider.valueChanged.connect(self._do_slider)

        self.spin.valueChanged.connect(self._do_spin)
        self.widgets = [self.spin] + ([self.slider] if self.slider else [])
        self.definir(p.padrao)

    def _indice(self, v: float) -> int:
        return round((v - self._base) / self._passo_slider)

    def _do_spin(self, v: float) -> None:
        if self._p.somente_impar and int(v) % 2 == 0:
            # Par digitado: vai para o ímpar seguinte (ou anterior, se passar do máximo).
            corrigido = int(v) + 1 if int(v) + 1 <= self._p.maximo else int(v) - 1
            self.spin.setValue(corrigido)  # dispara _do_spin de novo, já ímpar
            return
        if self.slider is not None:
            self.slider.blockSignals(True)  # evita o slider "arredondar" o valor digitado
            self.slider.setValue(self._indice(v))
            self.slider.blockSignals(False)
        self._ao_mudar()

    def _do_slider(self, indice: int) -> None:
        v = self._base + indice * self._passo_slider
        v = min(max(v, self._p.minimo), self._p.maximo)
        self.spin.setValue(int(round(v)) if self._inteiro else v)

    def valor(self) -> int | float:
        self.spin.interpretText()  # confirma o que estiver digitado e não confirmado
        return int(self.spin.value()) if self._inteiro else float(self.spin.value())

    def definir(self, valor: Any) -> None:
        self.spin.setValue(valor)
        self._do_spin(self.spin.value())  # garante slider sincronizado mesmo sem mudança


class _ControleBool(_Controle):
    def __init__(self, p: Param, ao_mudar: Callable[[], None]) -> None:
        self.caixa = QCheckBox()
        self.caixa.setChecked(bool(p.padrao))
        self.caixa.toggled.connect(lambda _: ao_mudar())
        self.widgets = [self.caixa]

    def valor(self) -> bool:
        return self.caixa.isChecked()

    def definir(self, valor: Any) -> None:
        self.caixa.setChecked(bool(valor))


class _ControleEscolha(_Controle):
    def __init__(self, p: Param, ao_mudar: Callable[[], None]) -> None:
        self.combo = QComboBox()
        self.combo.addItems(list(p.opcoes))
        self.combo.setCurrentText(p.padrao)
        self.combo.currentIndexChanged.connect(lambda _: ao_mudar())
        self.widgets = [self.combo]

    def valor(self) -> str:
        return self.combo.currentText()

    def definir(self, valor: Any) -> None:
        self.combo.setCurrentText(valor)


_CONTROLE_POR_TIPO = {
    TipoParam.INT: _ControleNumerico,
    TipoParam.FLOAT: _ControleNumerico,
    TipoParam.BOOL: _ControleBool,
    TipoParam.ESCOLHA: _ControleEscolha,
}


# --------------------------------------------------------------------------- painel


class PainelParametros(QWidget):
    """Controles do processo atual. Emite ``valores_alterados`` a cada mudança."""

    valores_alterados = Signal()
    _COLUNAS = 2  # parâmetros por linha (economiza altura para as imagens)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec: ProcessoSpec | None = None
        self._controles: dict[str, _Controle] = {}
        self._memoria: dict[str, dict[str, Any]] = {}  # últimos valores usados, por processo

        self._descricao = QLabel()
        self._descricao.setWordWrap(True)
        self._descricao.setTextFormat(Qt.TextFormat.PlainText)

        self._grade_widget = QWidget()

        self.botao_padroes = QPushButton("Restaurar padrões")
        self.botao_padroes.setToolTip("Volta todos os parâmetros deste processo ao valor padrão.")
        self.botao_padroes.clicked.connect(self.restaurar_padroes)

        self._rodape = QHBoxLayout()
        self._rodape.addStretch(1)
        self._rodape.addWidget(self.botao_padroes)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._descricao)
        self._layout.addWidget(self._grade_widget)
        self._layout.addLayout(self._rodape)
        self.definir_processo(None)

    # ------------------------------------------------------------------ consulta

    @property
    def processo(self) -> ProcessoSpec | None:
        return self._spec

    def controle(self, nome: str) -> _Controle:
        return self._controles[nome]

    def valores(self) -> dict[str, Any]:
        """Valores atuais, prontos para ``executar_processo``."""
        return {nome: c.valor() for nome, c in self._controles.items()}

    # ------------------------------------------------------------------- ações

    def definir_processo(self, spec: ProcessoSpec | None) -> None:
        """Troca os controles para os parâmetros de ``spec`` (lembrando os últimos valores)."""
        if self._spec is not None:
            self._memoria[self._spec.id] = self.valores()
        self._spec = spec
        self._controles = {}

        # Troca a grade inteira por uma nova (mais simples que remover linha a linha).
        nova = QWidget()
        antiga = self._grade_widget
        self._layout.replaceWidget(antiga, nova)
        # replaceWidget só tira do layout: sem hide(), a grade antiga continua desenhada
        # por baixo da nova até o deleteLater acontecer.
        antiga.hide()
        antiga.deleteLater()
        self._grade_widget = nova
        grade = QGridLayout(nova)
        grade.setContentsMargins(0, 4, 0, 0)
        grade.setHorizontalSpacing(10)
        try:
            self._preencher_grade(spec, grade)
        finally:
            # Widgets criados numa janela já visível ficam ocultos até o próximo ciclo de
            # eventos, e ocultos não contam no tamanho pedido. Mostrar já, para a janela
            # conseguir dar à caixa a altura certa logo em seguida.
            nova.show()

    def _preencher_grade(self, spec: ProcessoSpec | None, grade: QGridLayout) -> None:
        if spec is None:
            self._descricao.setText("Nenhum processo selecionado.")
            self.botao_padroes.setEnabled(False)
            return

        self._descricao.setText(spec.descricao)
        self.botao_padroes.setEnabled(bool(spec.parametros))
        if not spec.parametros:
            grade.addWidget(QLabel("Este processo não tem parâmetros."), 0, 0)

        for i, p in enumerate(spec.parametros):
            controle = _CONTROLE_POR_TIPO[p.tipo](p, self.valores_alterados.emit)
            self._controles[p.nome] = controle
            linha, coluna = divmod(i, self._COLUNAS)
            base = coluna * 3  # cada parâmetro ocupa: rótulo | caixa | slider

            dica = p.ajuda
            if p.tipo in (TipoParam.INT, TipoParam.FLOAT):
                dica = f"{p.ajuda}\n{_faixa_texto(p)}".strip()
            rotulo = QLabel(p.rotulo)
            rotulo.setToolTip(dica)
            grade.addWidget(rotulo, linha, base)

            if len(controle.widgets) == 2:  # caixa + slider
                grade.addWidget(controle.widgets[0], linha, base + 1)
                grade.addWidget(controle.widgets[1], linha, base + 2)
            else:  # ocupa caixa e slider
                grade.addWidget(controle.widgets[0], linha, base + 1, 1, 2)
            for w in controle.widgets:
                w.setToolTip(dica)
            rotulo.setBuddy(controle.widgets[0])  # clicar no rótulo foca o campo

        for c in range(self._COLUNAS):
            grade.setColumnStretch(c * 3 + 2, 1)  # os sliders ocupam a sobra

        lembrados = self._memoria.get(spec.id)
        if lembrados:
            self.definir_valores(lembrados)

    def adicionar_ao_rodape(self, widget: QWidget) -> None:
        """Coloca ``widget`` no rodapé, à direita de "Restaurar padrões" (ex.: Aplicar)."""
        self._rodape.addWidget(widget)

    def definir_valores(self, valores: dict[str, Any]) -> None:
        for nome, valor in valores.items():
            if nome in self._controles:
                self._controles[nome].definir(valor)

    def restaurar_padroes(self) -> None:
        if self._spec is not None:
            self.definir_valores({p.nome: p.padrao for p in self._spec.parametros})
