"""Exibição de imagens com ajuste à janela (I1).

``Visualizador`` mostra um título e uma imagem NumPy (uint8, cinza ou RGB) ajustada ao
espaço disponível, sem distorcer a proporção.

Detalhe de PDI na exibição: ao REDUZIR, a imagem é suavizada (senão aparecem
serrilhados que não existem na imagem); ao AMPLIAR, usa-se o vizinho mais próximo,
para que cada pixel apareça como um quadrado nítido e o usuário veja os valores
reais, sem uma interpolação que "inventa" tons intermediários.
"""

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter, QPalette
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


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
    """Título + imagem ajustada. Use ``definir_imagem(array)`` ou ``definir_imagem(None)``."""

    def __init__(
        self, titulo: str, texto_vazio: str = "Nenhuma imagem", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._imagem: np.ndarray | None = None
        self._titulo = QLabel(titulo)
        self._titulo.setStyleSheet("font-weight: 600;")
        self._tela = _Tela(texto_vazio)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._titulo)
        layout.addWidget(self._tela, stretch=1)

    @property
    def imagem(self) -> np.ndarray | None:
        return self._imagem

    def definir_titulo(self, titulo: str) -> None:
        self._titulo.setText(titulo)

    def definir_imagem(self, img: np.ndarray | None) -> None:
        self._imagem = img
        self._tela.definir(None if img is None else numpy_para_qimage(img))
