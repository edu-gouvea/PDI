"""Janela principal: seleção | entrada | saída | parâmetros | acessórios (I1).

Layout, seguindo o esboço do PDF do professor:

    ┌ Arquivo: [Abrir] [Salvar saída] [Usar saída como entrada] [Aplicar] ┐
    │ ┌─────────┐ ┌──────────────────┐ ┌──────────────────┐               │
    │ │Processos│ │ Imagem de entrada│ │ Imagem de saída  │ (abas se >1)  │
    │ │         │ └──────────────────┘ └──────────────────┘               │
    │ │         │ [ Parâmetros do processo selecionado      ] [Aplicar]   │
    │ │         │ [ Dados / imagens acessórias              ]             │
    │ └─────────┘                                                         │
    └ barra de status: tipo da imagem, avisos, erros      [progresso] [tempo] ┘

O processamento roda em outra thread (``ExecutorDeProcessos``): a janela continua
respondendo e mostra o tempo gasto ao final.
"""

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pdi.core.contrato import ProcessoSpec, Resultado
from pdi.core.imagem_io import ImagemCarregada, carregar_png, salvar_png
from pdi.core.registro import carregar_processos, processos_por_grupo
from pdi.core.util import tipo_da_imagem
from pdi.ui.painel_acessorios import PainelAcessorios
from pdi.ui.painel_parametros import PainelParametros
from pdi.ui.painel_processos import PainelProcessos
from pdi.ui.visualizador import Visualizador
from pdi.ui.worker import ExecutorDeProcessos

TITULO_APP = "PDI: Processamento Digital de Imagens"
# Altura da área de dados quando há histogramas ou imagens acessórias.
ALTURA_ACESSORIOS_COM_CARTOES = 230

# Pasta inicial do diálogo "Abrir": imagens_teste/ do projeto, se existir.
_PASTA_IMAGENS_TESTE = Path(__file__).resolve().parents[3] / "imagens_teste"


# ------------------------------------------------------------------ funções auxiliares


def formatar_tempo(segundos: float) -> str:
    """0.0123 → '12 ms'; 1.5 → '1,50 s'."""
    if segundos < 1:
        return f"{segundos * 1000:.0f} ms"
    return f"{segundos:.2f} s".replace(".", ",")


def _slug(texto: str) -> str:
    """'Mínimo' → 'minimo'; 'sal e pimenta' → 'sal-e-pimenta' (seguro para nome de arquivo)."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9.]+", "-", sem_acento.lower()).strip("-")


def nome_sugerido(
    original: str, spec: ProcessoSpec, parametros: dict[str, Any], titulo_saida: str = ""
) -> str:
    """Nome de arquivo que diz o que foi feito: 'cameraman_filtro-ordem_mediana_janela5.png'.

    Escolhas entram só pelo valor; números entram como nome+valor. Com várias saídas, o
    título da selecionada vai no fim (ex.: '..._r.png' para o canal R).
    """
    partes = [original, _slug(spec.id)]
    for nome, valor in parametros.items():
        if isinstance(valor, str):
            partes.append(_slug(valor))
        elif isinstance(valor, bool):
            partes.append(f"{_slug(nome)}-{'sim' if valor else 'nao'}")
        else:
            partes.append(f"{_slug(nome)}{valor:g}")
    if titulo_saida:
        partes.append(_slug(titulo_saida))
    return "_".join(p for p in partes if p) + ".png"


def _caixa(titulo: str, conteudo: QWidget) -> QGroupBox:
    caixa = QGroupBox(titulo)
    layout = QVBoxLayout(caixa)
    layout.addWidget(conteudo)
    return caixa


# --------------------------------------------------------------------------- janela


class JanelaPrincipal(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._entrada: ImagemCarregada | None = None
        self._processo: ProcessoSpec | None = None
        # O que gerou a saída exibida (para o nome sugerido ao salvar).
        self._origem_saida: tuple[ProcessoSpec, dict[str, Any]] | None = None
        self._origem_pendente: tuple[ProcessoSpec, dict[str, Any]] | None = None
        self._acessorios_expandidos = False
        self._ultima_pasta = _PASTA_IMAGENS_TESTE if _PASTA_IMAGENS_TESTE.is_dir() else Path.home()

        # Importa os módulos de processo (dispara os @registrar_processo). A interface
        # só conhece os processos por meio do registro.
        carregar_processos()

        self.executor = ExecutorDeProcessos(self)
        self.executor.concluido.connect(self._ao_concluir)
        self.executor.falhou.connect(self._ao_falhar)
        self.executor.ocupado_mudou.connect(self._ao_mudar_ocupado)

        self.setWindowTitle(TITULO_APP)
        self.resize(1280, 820)
        self._criar_acoes()
        self._criar_layout()
        self._criar_barra_de_status()
        self._atualizar_estado()
        self.statusBar().showMessage("Abra uma imagem PNG (Ctrl+O) para começar.")

    # ------------------------------------------------------------------ construção

    def _criar_acoes(self) -> None:
        self.acao_abrir = QAction("Abrir imagem…", self)
        self.acao_abrir.setShortcut(QKeySequence.StandardKey.Open)
        self.acao_abrir.triggered.connect(self._dialogo_abrir)

        self.acao_salvar = QAction("Salvar saída…", self)
        self.acao_salvar.setShortcut(QKeySequence.StandardKey.Save)
        self.acao_salvar.setToolTip("Salva a saída exibida (a aba selecionada) como PNG.")
        self.acao_salvar.triggered.connect(self._dialogo_salvar)

        self.acao_encadear = QAction("Usar saída como entrada", self)
        self.acao_encadear.setToolTip(
            "Transforma o resultado em nova entrada, para aplicar outro processo em seguida "
            "(ex.: aplicar ruído e depois filtrar)."
        )
        self.acao_encadear.triggered.connect(self.usar_saida_como_entrada)

        self.acao_aplicar = QAction("Aplicar", self)
        self.acao_aplicar.setShortcuts([QKeySequence("Ctrl+Return"), QKeySequence("F5")])
        self.acao_aplicar.setToolTip("Aplica o processo selecionado à entrada (Ctrl+Enter ou F5).")
        self.acao_aplicar.triggered.connect(self.aplicar)

        acao_sair = QAction("Sair", self)
        acao_sair.setShortcut(QKeySequence.StandardKey.Quit)
        acao_sair.triggered.connect(self.close)

        menu = self.menuBar().addMenu("&Arquivo")
        menu.addActions([self.acao_abrir, self.acao_salvar, self.acao_encadear])
        menu.addSeparator()
        menu.addAction(acao_sair)
        self.menuBar().addMenu("&Processo").addAction(self.acao_aplicar)

        barra = self.addToolBar("Arquivo")
        barra.setMovable(False)
        barra.addActions([self.acao_abrir, self.acao_salvar, self.acao_encadear])
        barra.addSeparator()
        barra.addAction(self.acao_aplicar)

    def _criar_layout(self) -> None:
        self.vis_entrada = Visualizador("Imagem de entrada", "Nenhuma imagem aberta")
        self.vis_saida = Visualizador("Imagem de saída", "Escolha um processo e clique em Aplicar")
        self.vis_saida.imagem_trocada.connect(self._atualizar_estado)

        imagens = QWidget()
        lado_a_lado = QHBoxLayout(imagens)
        lado_a_lado.setContentsMargins(0, 0, 0, 0)
        lado_a_lado.addWidget(self.vis_entrada, stretch=1)  # mesmo stretch = mesmo tamanho
        lado_a_lado.addWidget(self.vis_saida, stretch=1)

        self.painel_parametros = PainelParametros()
        # Botão grande ao lado de "Restaurar padrões"; segue a ação (texto, habilitado).
        self.botao_aplicar = QToolButton()
        self.botao_aplicar.setDefaultAction(self.acao_aplicar)
        self.botao_aplicar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.botao_aplicar.setMinimumWidth(110)
        self.botao_aplicar.setStyleSheet("font-weight: 600; padding: 4px 16px;")
        self.painel_parametros.adicionar_ao_rodape(self.botao_aplicar)
        self.caixa_parametros = _caixa("Parâmetros do processo selecionado", self.painel_parametros)

        self.painel_acessorios = PainelAcessorios()
        self.caixa_acessorios = _caixa("Dados e imagens acessórias", self.painel_acessorios)

        direita = QSplitter(Qt.Orientation.Vertical)
        self._divisor = direita
        direita.addWidget(imagens)
        direita.addWidget(self.caixa_parametros)
        direita.addWidget(self.caixa_acessorios)
        # Parâmetros não esticam: ficam com a altura dos controles; o resto vai às imagens.
        # A área de dados começa baixa e cresce só quando há gráficos ou imagens nela
        # (_ajustar_altura_acessorios); todo o resto vai para as imagens.
        direita.setStretchFactor(0, 1)
        direita.setStretchFactor(1, 0)
        direita.setStretchFactor(2, 0)
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
        self._ajustar_altura_parametros()

    def _criar_barra_de_status(self) -> None:
        self.progresso = QProgressBar()
        self.progresso.setRange(0, 0)  # sem duração conhecida: barra "indo e voltando"
        self.progresso.setMaximumWidth(140)
        self.progresso.setTextVisible(False)
        self.progresso.setVisible(False)
        self.rotulo_tempo = QLabel()
        self.rotulo_tempo.setToolTip("Tempo de execução do último processo")
        self.statusBar().addPermanentWidget(self.progresso)
        self.statusBar().addPermanentWidget(self.rotulo_tempo)

    # -------------------------------------------------------------------- estado

    @property
    def entrada(self) -> ImagemCarregada | None:
        return self._entrada

    @property
    def saida(self) -> np.ndarray | None:
        """A saída exibida agora (com várias saídas, a da aba selecionada)."""
        return self.vis_saida.imagem

    @property
    def processo(self) -> ProcessoSpec | None:
        return self._processo

    def _pode_aplicar(self) -> bool:
        return (
            self._entrada is not None
            and self._processo is not None
            and self.painel_processos.esta_habilitado(self._processo.id)
            and not self.executor.ocupado
        )

    def _atualizar_estado(self) -> None:
        """Habilita só o que faz sentido agora (ex.: não dá para salvar sem saída)."""
        tem_saida = self.saida is not None
        livre = not self.executor.ocupado
        self.acao_salvar.setEnabled(tem_saida)
        self.acao_encadear.setEnabled(tem_saida and livre)
        self.acao_aplicar.setEnabled(self._pode_aplicar())
        nome = f" · {self._entrada.caminho.name}" if self._entrada else ""
        self.setWindowTitle(TITULO_APP + nome)

    def _ao_selecionar_processo(self, spec: ProcessoSpec | None) -> None:
        self._processo = spec
        self.painel_parametros.definir_processo(spec)
        titulo = (
            "Parâmetros do processo selecionado" if spec is None else f"Parâmetros: {spec.nome}"
        )
        self.caixa_parametros.setTitle(titulo)
        self._ajustar_altura_parametros()
        self._atualizar_estado()

    def _ajustar_altura_parametros(self) -> None:
        """Dá à caixa de parâmetros exatamente a altura que os controles pedem.

        O QSplitter só calcula os tamanhos na criação; quando os controles mudam (ex.: 1
        parâmetro → 4), ele não abre espaço sozinho e as linhas ficam cortadas.
        """
        self.painel_parametros.layout().activate()
        self.caixa_parametros.layout().activate()
        self.caixa_parametros.setFixedHeight(self.caixa_parametros.sizeHint().height())

    def _ajustar_altura_acessorios(self) -> None:
        """Dá altura à área de dados só quando há cartões (histogramas, imagens) nela.

        Com só texto, ela fica com uma linha e as imagens ganham o espaço. O ajuste só
        acontece quando o estado muda, para não desfazer a cada Aplicar uma divisória
        que o usuário arrastou à mão.
        """
        expandir = bool(self.painel_acessorios.cartoes)
        if expandir == self._acessorios_expandidos:
            return
        tamanhos = self._divisor.sizes()
        total = sum(tamanhos)
        if total == 0:  # janela ainda não exibida: tenta de novo no próximo resultado
            return
        self._acessorios_expandidos = expandir
        # Atualiza o layout antes de medir: a faixa de cartões acabou de ser escondida e,
        # sem isso, ainda entraria na conta da altura mínima.
        self.painel_acessorios.layout().activate()
        self.caixa_acessorios.layout().activate()
        desejado = (
            ALTURA_ACESSORIOS_COM_CARTOES
            if expandir
            else self.caixa_acessorios.minimumSizeHint().height()
        )
        tamanhos[2] = desejado
        tamanhos[0] = max(total - tamanhos[1] - desejado, 0)
        self._divisor.setSizes(tamanhos)

    def _definir_entrada(self, imagem: ImagemCarregada) -> None:
        """Troca a imagem de entrada e reavalia quais processos se aplicam a ela."""
        self.executor.descartar_pendente()  # um resultado da imagem antiga não serve mais
        self._entrada = imagem
        self.vis_entrada.definir_imagem(imagem.pixels)
        self.painel_processos.definir_tipo_imagem(imagem.tipo)
        self.definir_saida(None)

    def _mostrar_status_da_entrada(self) -> None:
        assert self._entrada is not None
        partes = [f"{self._entrada.caminho.name}: {self._entrada.resumo}", *self._entrada.avisos]
        self.statusBar().showMessage("  |  ".join(partes))

    def _mostrar_erro(self, mensagem: str) -> None:
        QMessageBox.warning(self, TITULO_APP, mensagem)

    # ------------------------------------------------------------------ execução

    def aplicar(self) -> bool:
        """Dispara o processo selecionado em segundo plano. Devolve se conseguiu disparar."""
        if not self._pode_aplicar():
            return False
        assert self._entrada is not None and self._processo is not None
        parametros = self.painel_parametros.valores()
        # Só vira a origem da saída quando o resultado chegar (se falhar, a saída antiga
        # continua na tela e o nome sugerido continua sendo o dela).
        self._origem_pendente = (self._processo, parametros)
        self.statusBar().showMessage(f"Aplicando {self._processo.nome}…")
        return self.executor.executar(self._processo, self._entrada.pixels, parametros)

    def _ao_mudar_ocupado(self, ocupado: bool) -> None:
        self.progresso.setVisible(ocupado)
        self._atualizar_estado()

    def _ao_concluir(self, resultado: Resultado, segundos: float) -> None:
        self._origem_saida = self._origem_pendente
        self.vis_saida.definir_imagens(resultado.saidas)
        self.painel_acessorios.definir(resultado)
        self._ajustar_altura_acessorios()
        tempo = formatar_tempo(segundos)
        self.rotulo_tempo.setText(f"Tempo: {tempo}")
        nome = self._origem_saida[0].nome if self._origem_saida else "Processo"
        self.statusBar().showMessage(f"{nome} aplicado em {tempo}.")
        self._atualizar_estado()

    def _ao_falhar(self, mensagem: str) -> None:
        self.statusBar().showMessage(f"Erro: {mensagem}")
        self._mostrar_erro(mensagem)

    def closeEvent(self, evento: QCloseEvent) -> None:  # noqa: N802 (nome do Qt)
        self.executor.encerrar()  # sem isso o Qt aborta ao destruir uma thread ativa
        super().closeEvent(evento)

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
        """Mostra ``img`` como única saída (ou limpa, com None)."""
        self.vis_saida.definir_imagem(img)
        if img is None:
            self._origem_saida = None
            self.painel_acessorios.definir(None)
            self._ajustar_altura_acessorios()
        self._atualizar_estado()

    def salvar_saida(self, caminho: str | Path) -> Path | None:
        if self.saida is None:
            return None
        try:
            usado = salvar_png(self.saida, caminho)
        except ValueError as e:
            self._mostrar_erro(str(e))
            return None
        self._ultima_pasta = usado.parent
        self.statusBar().showMessage(f"Saída salva em {usado}")
        return usado

    def usar_saida_como_entrada(self) -> None:
        saida = self.saida
        if saida is None or self._entrada is None or self.executor.ocupado:
            return
        # Mesmo caminho de origem (para nomes sugeridos), pixels da saída, sem avisos.
        # O tipo pode mudar: a decomposição RGB de uma colorida devolve uma saída cinza.
        nova = ImagemCarregada(saida.copy(), tipo_da_imagem(saida), self._entrada.caminho)
        self._definir_entrada(nova)
        self.statusBar().showMessage(
            f"Saída transformada em entrada: {self._entrada.resumo}. Escolha o próximo processo."
        )

    def _sugestao_para_salvar(self) -> str:
        assert self._entrada is not None
        original = self._entrada.caminho.stem
        if self._origem_saida is None:
            return f"{original}_saida.png"
        spec, parametros = self._origem_saida
        titulo = self.vis_saida.titulo_atual if self.vis_saida.quantidade > 1 else ""
        return nome_sugerido(original, spec, parametros, titulo)

    def _dialogo_abrir(self) -> None:
        caminho, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagem", str(self._ultima_pasta), "Imagens PNG (*.png)"
        )
        if caminho:
            self.abrir_arquivo(caminho)

    def _dialogo_salvar(self) -> None:
        if self.saida is None or self._entrada is None:
            return
        sugestao = self._ultima_pasta / self._sugestao_para_salvar()
        caminho, _ = QFileDialog.getSaveFileName(
            self, "Salvar saída", str(sugestao), "Imagens PNG (*.png)"
        )
        if caminho:
            self.salvar_saida(caminho)
