"""Grade de imagens acessórias e histogramas (Matplotlib) (I1).

Mostra o que vem em ``Resultado.acessorios`` e ``Resultado.info``:

- imagem ``(titulo, array)`` → cartão com ``Visualizador`` (mesmo ajuste de escala)
- ``Histograma`` → cartão com gráfico de barras dos 256 níveis, valor exato no tooltip
- ``info`` → uma linha de texto no topo (chave: valor · chave: valor …)

Os cartões ficam lado a lado numa faixa com rolagem horizontal: a área de dados é
baixa e larga, e a equalização (2 histogramas) e o Laplaciano (1 imagem) cabem sem
rolar. Quem precisar ver maior arrasta a divisória acima da área.

O Matplotlib só é importado quando o primeiro histograma aparece (o import custa
algumas centenas de milissegundos e atrasaria a abertura do programa).
"""

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QPalette
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from pdi.core.contrato import Histograma, Resultado
from pdi.ui.visualizador import Visualizador

# Cor única dos histogramas (uma série só, sem legenda: o título já diz o que é).
# Validada (faixa de luminosidade, croma e contraste ≥ 3:1) nos modos claro e escuro.
_COR_HISTOGRAMA = {"claro": "#2a78d6", "escuro": "#3987e5"}
_LARGURA_MINIMA_CARTAO = 260


def _modo_escuro(widget: QWidget) -> bool:
    return widget.palette().color(QPalette.ColorRole.Window).lightness() < 128


def _formatar_valor(v: float) -> str:
    return f"{int(v)}" if float(v).is_integer() else f"{v:.4f}"


def _formatar_eixo(v: float, _pos: Any = None) -> str:
    """Contagens grandes encurtadas: 12000 → '12k'."""
    if abs(v) >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


class CartaoHistograma(QWidget):
    """Título + gráfico de barras de 256 níveis. Passe o mouse para ver o valor exato."""

    def __init__(self, histograma: Histograma, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Import tardio: só paga o custo do Matplotlib quem mostra um histograma.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        self.titulo = histograma.titulo
        self.valores = np.asarray(histograma.valores, dtype=np.float64)
        unidade_inteira = np.all(np.mod(self.valores, 1) == 0)
        self._unidade = "pixels" if unidade_inteira else ""

        paleta = self.palette()
        fundo = paleta.color(QPalette.ColorRole.Window).name()
        texto = paleta.color(QPalette.ColorRole.WindowText).name()
        texto_suave = paleta.color(QPalette.ColorRole.PlaceholderText).name()
        cor = _COR_HISTOGRAMA["escuro" if _modo_escuro(self) else "claro"]

        figura = Figure(figsize=(3.2, 1.6), dpi=100, facecolor=fundo, layout="constrained")
        eixo = figura.add_subplot()
        eixo.set_facecolor(fundo)
        # stairs com bordas 0..256: cada nível k é uma barra de largura 1 em [k, k+1).
        eixo.stairs(self.valores, np.arange(257), fill=True, color=cor, linewidth=0)
        eixo.set_xlim(0, 256)
        eixo.set_ylim(bottom=0)
        eixo.set_xticks([0, 64, 128, 192, 255])
        eixo.yaxis.set_major_locator(MaxNLocator(nbins=3))
        eixo.yaxis.set_major_formatter(FuncFormatter(_formatar_eixo))
        eixo.tick_params(labelsize=7, colors=texto_suave, length=2)
        eixo.grid(axis="y", color=texto_suave, alpha=0.25, linewidth=0.5)
        eixo.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            eixo.spines[lado].set_visible(False)
        eixo.spines["bottom"].set_color(texto_suave)

        self.canvas = FigureCanvasQTAgg(figura)
        self.canvas.setMinimumSize(200, 90)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.mpl_connect("motion_notify_event", self._ao_mover_mouse)
        self._eixo = eixo

        rotulo = QLabel(self.titulo)
        rotulo.setStyleSheet(f"font-weight: 600; color: {texto};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(rotulo)
        layout.addWidget(self.canvas, stretch=1)

    def texto_do_nivel(self, nivel: int) -> str:
        nivel = int(np.clip(nivel, 0, 255))
        valor = _formatar_valor(self.valores[nivel])
        return f"Nível {nivel}: {valor} {self._unidade}".rstrip()

    def _ao_mover_mouse(self, evento: Any) -> None:
        if evento.inaxes is not self._eixo or evento.xdata is None:
            QToolTip.hideText()
            return
        QToolTip.showText(QCursor.pos(), self.texto_do_nivel(int(evento.xdata)), self.canvas)


class CartaoImagem(Visualizador):
    """Imagem acessória (ex.: Laplaciano ajustado), com o mesmo ajuste de escala."""

    def __init__(self, titulo: str, imagem: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(titulo, parent=parent)
        self.titulo = titulo
        # Num cartão nunca há abas: sem reservar o espaço delas, e com altura mínima
        # menor, o cartão cabe na área baixa de dados sem empurrar os outros para fora.
        self._abas.setFixedHeight(0)
        self._tela.setMinimumSize(120, 60)
        self.definir_imagem(imagem)


class PainelAcessorios(QWidget):
    """Mostra ``info`` e ``acessorios`` de um ``Resultado`` (ou nada, com None)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cartoes: list[QWidget] = []

        self._info = QLabel()
        self._info.setWordWrap(True)
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._rolagem = QScrollArea()
        self._rolagem.setWidgetResizable(True)
        self._rolagem.setFrameShape(QScrollArea.Shape.NoFrame)
        self._rolagem.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._faixa = QWidget()
        self._rolagem.setWidget(self._faixa)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._info)
        layout.addWidget(self._rolagem, stretch=1)
        self.definir(None)

    @property
    def texto_info(self) -> str:
        return self._info.text()

    def definir(self, resultado: Resultado | None) -> None:
        # Faixa nova a cada resultado (a antiga sai inteira, com seus gráficos).
        antiga = self._rolagem.takeWidget()
        if antiga is not None:
            antiga.hide()
            antiga.deleteLater()
        self._faixa = QWidget()
        linha = QHBoxLayout(self._faixa)
        linha.setContentsMargins(0, 0, 0, 0)
        linha.setSpacing(12)
        self.cartoes = []

        if resultado is None:
            self._info.setText("Nada a exibir")
            self._info.setVisible(True)
            self._rolagem.setWidget(self._faixa)
            self._rolagem.setVisible(False)
            return

        linhas = [f"{chave}: {valor}" for chave, valor in resultado.info.items()]
        self._info.setText("   ·   ".join(linhas))
        self._info.setVisible(bool(linhas) or not resultado.acessorios)
        if not linhas and not resultado.acessorios:
            self._info.setText("Este processo não tem dados extras.")

        for item in resultado.acessorios:
            if isinstance(item, Histograma):
                cartao: QWidget = CartaoHistograma(item)
            else:
                cartao = CartaoImagem(item[0], item[1])
            cartao.setMinimumWidth(_LARGURA_MINIMA_CARTAO)
            linha.addWidget(cartao, stretch=1)
            self.cartoes.append(cartao)

        self._rolagem.setWidget(self._faixa)
        self._rolagem.setVisible(bool(self.cartoes))
