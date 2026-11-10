"""Lista de processos agrupada, habilitada conforme o tipo da imagem (I1).

Os processos vêm do registro (``processos_por_grupo``); este painel não conhece nenhum
processo pelo nome. Processos que não se aplicam à imagem aberta aparecem
acinzentados, com tooltip explicando o motivo (melhor que escondê-los: o usuário
entende por que a equalização "sumiu" com uma imagem colorida).
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from pdi.core.contrato import ProcessoSpec
from pdi.core.registro import motivo_desabilitado

_PAPEL_SPEC = Qt.ItemDataRole.UserRole  # onde cada item guarda o seu ProcessoSpec


class PainelProcessos(QWidget):
    """Árvore Grupo → Processo. Emite ``processo_selecionado(spec ou None)``."""

    processo_selecionado = Signal(object)

    def __init__(
        self, grupos: dict[str, list[ProcessoSpec]], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._tipo: str | None = None
        self._itens: dict[str, QTreeWidgetItem] = {}

        self.arvore = QTreeWidget()
        self.arvore.setHeaderHidden(True)
        self.arvore.setRootIsDecorated(False)  # sem setinhas: os grupos ficam sempre abertos
        self.arvore.setIndentation(12)
        self.arvore.setItemsExpandable(False)
        self.arvore.setUniformRowHeights(True)
        self.arvore.currentItemChanged.connect(self._ao_mudar_item)

        for grupo, specs in grupos.items():
            cabecalho = QTreeWidgetItem([grupo])
            # Só "habilitado": aparece normal, mas não pode ser selecionado.
            cabecalho.setFlags(Qt.ItemFlag.ItemIsEnabled)
            fonte = cabecalho.font(0)
            fonte.setBold(True)
            cabecalho.setFont(0, fonte)
            self.arvore.addTopLevelItem(cabecalho)
            for spec in specs:
                item = QTreeWidgetItem(cabecalho, [spec.nome])
                item.setData(0, _PAPEL_SPEC, spec)
                self._itens[spec.id] = item
            cabecalho.setExpanded(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.arvore)
        self.definir_tipo_imagem(None)

    # ------------------------------------------------------------------ consulta

    @property
    def processo_atual(self) -> ProcessoSpec | None:
        item = self.arvore.currentItem()
        return None if item is None else item.data(0, _PAPEL_SPEC)

    def item_do_processo(self, id_processo: str) -> QTreeWidgetItem:
        return self._itens[id_processo]

    def esta_habilitado(self, id_processo: str) -> bool:
        return bool(self._itens[id_processo].flags() & Qt.ItemFlag.ItemIsEnabled)

    # ------------------------------------------------------------------- ações

    def selecionar(self, id_processo: str) -> bool:
        """Seleciona o processo, se estiver habilitado. Devolve se conseguiu."""
        if not self.esta_habilitado(id_processo):
            return False
        self.arvore.setCurrentItem(self._itens[id_processo])
        return True

    def definir_tipo_imagem(self, tipo: str | None) -> None:
        """Habilita/desabilita os processos para ``tipo`` ("cinza", "rgb" ou None = sem imagem)."""
        self._tipo = tipo
        for item in self._itens.values():
            spec: ProcessoSpec = item.data(0, _PAPEL_SPEC)
            motivo = None if tipo is None else motivo_desabilitado(spec, tipo)
            if motivo is None:
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setToolTip(0, spec.descricao)
            else:
                item.setFlags(Qt.ItemFlag.NoItemFlags)  # acinzentado e não selecionável
                item.setToolTip(0, f"{motivo}\n\n{spec.descricao}")

        # O processo selecionado pode ter deixado de se aplicar (ex.: abriu uma colorida
        # com a equalização selecionada): limpa a seleção para não aplicá-lo por engano.
        atual = self.processo_atual
        if atual is not None and not self.esta_habilitado(atual.id):
            self.arvore.setCurrentItem(None)

    def _ao_mudar_item(
        self, atual: QTreeWidgetItem | None, anterior: QTreeWidgetItem | None
    ) -> None:
        if atual is not None and atual.data(0, _PAPEL_SPEC) is None:
            # Clicou no nome de um grupo: mantém o processo que estava selecionado.
            self.arvore.blockSignals(True)
            self.arvore.setCurrentItem(anterior)
            self.arvore.blockSignals(False)
            return
        self.processo_selecionado.emit(None if atual is None else atual.data(0, _PAPEL_SPEC))
