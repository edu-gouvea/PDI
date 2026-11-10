"""Exibição de imagens com ajuste à janela (I1).

``Visualizador`` mostra um título e uma ou mais imagens NumPy (uint8, cinza ou RGB)
ajustadas ao espaço disponível, sem distorcer a proporção (várias imagens = abas).

Detalhe de PDI na exibição: ao REDUZIR, a imagem é suavizada (senão aparecem
serrilhados que não existem na imagem); ao AMPLIAR, usa-se o vizinho mais próximo,
para que cada pixel apareça como um quadrado nítido e o usuário veja os valores
reais, sem uma interpolação que "inventa" tons intermediários.
"""

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPalette
from PySide6.QtWidgets import QLabel, QSizePolicy, QTabBar, QVBoxLayout, QWidget


def numpy_para_qimage(img: np.ndarray) -> QImage:
    """Converte uint8 ``(H, W)`` ou ``(H, W, 3)`` RGB em ``QImage`` (cópia independente)."""
    img = np.ascontiguousarray(img)  # o QImage exige linhas contíguas na memória
    altura, largura = img.shape[:2]
    if img.ndim == 2:
        formato, bytes_por_linha = QImage.Format.Format_Grayscale8, largura
    else:
        formato, bytes_por_linha = QImage.Format.Format_RGB888, 3 * largura
    # O QImage criado assim só aponta para a memória do array; .copy() faz uma cópia
    # própria, para a imagem continuar válida mesmo depois que o array for liberado.
    return QImage(img.data, largura, altura, bytes_por_linha, formato).copy()


class _Tela(QWidget):
    """Área que desenha a imagem centralizada e ajustada."""

    def __init__(self, texto_vazio: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._qimage: QImage | None = None
        self._texto_vazio = texto_vazio
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.ColorRole.Base)

    def definir(self, qimage: QImage | None) -> None:
        self._qimage = qimage
        self.update()

    def area_da_imagem(self) -> QRectF:
        """Retângulo onde a imagem é desenhada: maior possível, mesma proporção, centrado."""
        if self._qimage is None:
            return QRectF()
        iw, ih = self._qimage.width(), self._qimage.height()
        escala = min(self.width() / iw, self.height() / ih)
        w, h = iw * escala, ih * escala
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, evento) -> None:  # noqa: N802 (nome exigido pelo Qt)
        p = QPainter(self)
        if self._qimage is None:
            p.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._texto_vazio)
            return
        area = self.area_da_imagem()
        reduzindo = area.width() < self._qimage.width()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, reduzindo)
        p.drawImage(area, self._qimage)


class Visualizador(QWidget):
    """Título + uma ou mais imagens ajustadas.

    - ``definir_imagem(array ou None)``: uma imagem só.
    - ``definir_imagens([(titulo, array), ...])``: várias (ex.: canais R, G, B); aparece
      uma aba por imagem e ``imagem`` é a da aba selecionada.
    """

    imagem_trocada = Signal()

    def __init__(
        self, titulo: str, texto_vazio: str = "Nenhuma imagem", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._imagens: list[tuple[str, np.ndarray]] = []
        self._titulo = QLabel(titulo)
        self._titulo.setStyleSheet("font-weight: 600;")
        self._abas = QTabBar()
        self._abas.setVisible(False)  # só aparece com mais de uma imagem
        # O espaço das abas fica reservado mesmo quando ocultas: assim entrada e saída
        # têm sempre a mesma área útil e as imagens aparecem na mesma escala.
        politica = self._abas.sizePolicy()
        politica.setRetainSizeWhenHidden(True)
        self._abas.setSizePolicy(politica)
        # Uma barra de abas vazia pede altura zero; fixa a altura que ela tem com abas.
        self._abas.addTab("R")
        self._abas.setFixedHeight(self._abas.sizeHint().height())
        self._abas.removeTab(0)
        self._abas.setExpanding(False)
        self._abas.currentChanged.connect(self._ao_trocar_aba)
        self._tela = _Tela(texto_vazio)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._titulo)
        layout.addWidget(self._abas)
        layout.addWidget(self._tela, stretch=1)

    # ------------------------------------------------------------------ consulta

    @property
    def imagem(self) -> np.ndarray | None:
        """Imagem exibida agora (a da aba selecionada)."""
        i = self._abas.currentIndex() if len(self._imagens) > 1 else 0
        return self._imagens[i][1] if self._imagens else None

    @property
    def titulo_atual(self) -> str:
        i = self._abas.currentIndex() if len(self._imagens) > 1 else 0
        return self._imagens[i][0] if self._imagens else ""

    @property
    def quantidade(self) -> int:
        return len(self._imagens)

    # ------------------------------------------------------------------- ações

    def definir_titulo(self, titulo: str) -> None:
        self._titulo.setText(titulo)

    def definir_imagem(self, img: np.ndarray | None) -> None:
        self.definir_imagens([] if img is None else [("", img)])

    def definir_imagens(self, imagens: list[tuple[str, np.ndarray]]) -> None:
        self._imagens = list(imagens)
        self._abas.blockSignals(True)
        while self._abas.count():
            self._abas.removeTab(0)
        for titulo, _ in self._imagens:
            self._abas.addTab(titulo)
        self._abas.setCurrentIndex(0)
        self._abas.blockSignals(False)
        self._abas.setVisible(len(self._imagens) > 1)
        self._mostrar_atual()

    def selecionar(self, indice: int) -> None:
        self._abas.setCurrentIndex(indice)

    def _ao_trocar_aba(self, _indice: int) -> None:
        self._mostrar_atual()

    def _mostrar_atual(self) -> None:
        img = self.imagem
        self._tela.definir(None if img is None else numpy_para_qimage(img))
        self.imagem_trocada.emit()
