"""Janela principal: seleção | entrada | saída | parâmetros | acessórios (I1).

Layout, seguindo o esboço do PDF do professor:

    ┌ Arquivo: [Abrir] [Salvar saída] [Usar saída como entrada] ──────────┐
    │ ┌─────────┐ ┌──────────────────┐ ┌──────────────────┐               │
    │ │Processos│ │ Imagem de entrada│ │ Imagem de saída  │               │
    │ │         │ └──────────────────┘ └──────────────────┘               │
    │ │         │ [ Parâmetros do processo selecionado      ]             │
    │ │         │ [ Dados / imagens acessórias              ]             │
    │ └─────────┘                                                         │
    └ barra de status: tipo da imagem, dimensões, avisos, tempo ──────────┘

Neste passo: esqueleto, abrir imagem e salvar a saída. A lista de processos, os
parâmetros e os acessórios entram nos próximos passos.
"""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pdi.core.contrato import ProcessoSpec
from pdi.core.imagem_io import ImagemCarregada, carregar_png, salvar_png
from pdi.core.registro import carregar_processos, processos_por_grupo
from pdi.core.util import tipo_da_imagem
from pdi.ui.painel_processos import PainelProcessos
from pdi.ui.visualizador import Visualizador

TITULO_APP = "PDI: Processamento Digital de Imagens"

# Pasta inicial do diálogo "Abrir": imagens_teste/ do projeto, se existir.
_PASTA_IMAGENS_TESTE = Path(__file__).resolve().parents[3] / "imagens_teste"


def _caixa(titulo: str, conteudo: QWidget) -> QGroupBox:
    caixa = QGroupBox(titulo)
    layout = QVBoxLayout(caixa)
    layout.addWidget(conteudo)
    return caixa


def _aviso_provisorio(texto: str) -> QLabel:
    rotulo = QLabel(texto)
    rotulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
    rotulo.setEnabled(False)  # aparece acinzentado
    return rotulo


class JanelaPrincipal(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._entrada: ImagemCarregada | None = None
        self._saida: np.ndarray | None = None
        self._processo: ProcessoSpec | None = None
        self._ultima_pasta = _PASTA_IMAGENS_TESTE if _PASTA_IMAGENS_TESTE.is_dir() else Path.home()

        # Importa os módulos de processo (dispara os @registrar_processo). A interface
        # só conhece os processos por meio do registro.
        carregar_processos()

        self.setWindowTitle(TITULO_APP)
        self.resize(1280, 820)
        self._criar_acoes()
        self._criar_layout()
        self._atualizar_estado()
        self.statusBar().showMessage("Abra uma imagem PNG (Ctrl+O) para começar.")

    # ------------------------------------------------------------------ construção

    def _criar_acoes(self) -> None:
        self.acao_abrir = QAction("Abrir imagem…", self)
        self.acao_abrir.setShortcut(QKeySequence.StandardKey.Open)
        self.acao_abrir.triggered.connect(self._dialogo_abrir)

        self.acao_salvar = QAction("Salvar saída…", self)
        self.acao_salvar.setShortcut(QKeySequence.StandardKey.Save)
        self.acao_salvar.triggered.connect(self._dialogo_salvar)

        self.acao_encadear = QAction("Usar saída como entrada", self)
        self.acao_encadear.setToolTip(
            "Transforma o resultado em nova entrada, para aplicar outro processo em seguida "
            "(ex.: aplicar ruído e depois filtrar)."
        )
        self.acao_encadear.triggered.connect(self.usar_saida_como_entrada)

        acao_sair = QAction("Sair", self)
        acao_sair.setShortcut(QKeySequence.StandardKey.Quit)
        acao_sair.triggered.connect(self.close)

        menu = self.menuBar().addMenu("&Arquivo")
        menu.addActions([self.acao_abrir, self.acao_salvar, self.acao_encadear])
        menu.addSeparator()
        menu.addAction(acao_sair)

        barra = self.addToolBar("Arquivo")
        barra.setMovable(False)
        barra.addActions([self.acao_abrir, self.acao_salvar, self.acao_encadear])

    def _criar_layout(self) -> None:
        self.vis_entrada = Visualizador("Imagem de entrada", "Nenhuma imagem aberta")
        self.vis_saida = Visualizador("Imagem de saída", "Escolha um processo e clique em Aplicar")

        imagens = QWidget()
        lado_a_lado = QHBoxLayout(imagens)
        lado_a_lado.setContentsMargins(0, 0, 0, 0)
        lado_a_lado.addWidget(self.vis_entrada, stretch=1)  # mesmo stretch = mesmo tamanho
        lado_a_lado.addWidget(self.vis_saida, stretch=1)

        # Provisório: mostra a descrição do processo até o painel de parâmetros existir.
        self._rotulo_parametros = _aviso_provisorio("Nenhum processo selecionado")
        self._rotulo_parametros.setWordWrap(True)
        self.caixa_parametros = _caixa(
            "Parâmetros do processo selecionado", self._rotulo_parametros
        )
        self.caixa_acessorios = _caixa(
            "Dados e imagens acessórias", _aviso_provisorio("Nada a exibir")
        )

        direita = QSplitter(Qt.Orientation.Vertical)
        direita.addWidget(imagens)
        direita.addWidget(self.caixa_parametros)
        direita.addWidget(self.caixa_acessorios)
        direita.setStretchFactor(0, 5)
        direita.setStretchFactor(1, 1)
        direita.setStretchFactor(2, 2)
        direita.setChildrenCollapsible(False)

        self.painel_processos = PainelProcessos(processos_por_grupo())
        self.painel_processos.processo_selecionado.connect(self._ao_selecionar_processo)
        self.caixa_processos = _caixa("Processos", self.painel_processos)
        self.caixa_processos.setMinimumWidth(260)

        principal = QSplitter(Qt.Orientation.Horizontal)
        principal.addWidget(self.caixa_processos)
        principal.addWidget(direita)
        principal.setStretchFactor(1, 1)
        principal.setChildrenCollapsible(False)

        central = QWidget()
        margem = QVBoxLayout(central)
        margem.setContentsMargins(8, 8, 8, 4)
        margem.addWidget(principal)
        self.setCentralWidget(central)

    # -------------------------------------------------------------------- estado

    @property
    def entrada(self) -> ImagemCarregada | None:
        return self._entrada

    @property
    def saida(self) -> np.ndarray | None:
        return self._saida

    @property
    def processo(self) -> ProcessoSpec | None:
        return self._processo

    def _ao_selecionar_processo(self, spec: ProcessoSpec | None) -> None:
        self._processo = spec
        if spec is None:
            self._rotulo_parametros.setText("Nenhum processo selecionado")
        else:
            self._rotulo_parametros.setText(f"{spec.nome}: {spec.descricao}")
        self._atualizar_estado()

    def _definir_entrada(self, imagem: ImagemCarregada) -> None:
        """Troca a imagem de entrada e reavalia quais processos se aplicam a ela."""
        self._entrada = imagem
        self.vis_entrada.definir_imagem(imagem.pixels)
        self.painel_processos.definir_tipo_imagem(imagem.tipo)
        self.definir_saida(None)

    def _atualizar_estado(self) -> None:
        """Habilita só o que faz sentido agora (ex.: não dá para salvar sem saída)."""
        tem_saida = self._saida is not None
        self.acao_salvar.setEnabled(tem_saida)
        self.acao_encadear.setEnabled(tem_saida)
        nome = f" · {self._entrada.caminho.name}" if self._entrada else ""
        self.setWindowTitle(TITULO_APP + nome)

    def _mostrar_status_da_entrada(self) -> None:
        assert self._entrada is not None
        partes = [f"{self._entrada.caminho.name}: {self._entrada.resumo}", *self._entrada.avisos]
        self.statusBar().showMessage("  |  ".join(partes))

    def _mostrar_erro(self, mensagem: str) -> None:
        QMessageBox.warning(self, TITULO_APP, mensagem)

    # ------------------------------------------------------------------- arquivo

    def abrir_arquivo(self, caminho: str | Path) -> bool:
        """Carrega o PNG como nova entrada. Devolve False (e mostra o erro) se falhar."""
        try:
            imagem = carregar_png(caminho)
        except ValueError as e:
            self._mostrar_erro(str(e))
            return False

        self._ultima_pasta = imagem.caminho.parent
        self._definir_entrada(imagem)
        self._mostrar_status_da_entrada()
        return True

    def definir_saida(self, img: np.ndarray | None) -> None:
        """Mostra ``img`` como saída principal (usado pela execução de processos)."""
        self._saida = img
        self.vis_saida.definir_imagem(img)
        self._atualizar_estado()

    def salvar_saida(self, caminho: str | Path) -> Path | None:
        if self._saida is None:
            return None
        try:
            usado = salvar_png(self._saida, caminho)
        except ValueError as e:
            self._mostrar_erro(str(e))
            return None
        self._ultima_pasta = usado.parent
        self.statusBar().showMessage(f"Saída salva em {usado}")
        return usado

    def usar_saida_como_entrada(self) -> None:
        if self._saida is None or self._entrada is None:
            return
        # Mesmo caminho de origem (para nomes sugeridos), pixels da saída, sem avisos.
        # O tipo pode mudar: a decomposição RGB de uma colorida devolve uma saída cinza.
        nova = ImagemCarregada(
            self._saida.copy(), tipo_da_imagem(self._saida), self._entrada.caminho
        )
        self._definir_entrada(nova)
        self.statusBar().showMessage(
            f"Saída transformada em entrada: {self._entrada.resumo}. Escolha o próximo processo."
        )

    def _dialogo_abrir(self) -> None:
        caminho, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagem", str(self._ultima_pasta), "Imagens PNG (*.png)"
        )
        if caminho:
            self.abrir_arquivo(caminho)

    def _dialogo_salvar(self) -> None:
        if self._saida is None or self._entrada is None:
            return
        sugestao = self._ultima_pasta / f"{self._entrada.caminho.stem}_saida.png"
        caminho, _ = QFileDialog.getSaveFileName(
            self, "Salvar saída", str(sugestao), "Imagens PNG (*.png)"
        )
        if caminho:
            self.salvar_saida(caminho)
