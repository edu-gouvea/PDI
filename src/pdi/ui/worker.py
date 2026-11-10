"""QThread que executa o processo sem travar a interface (I1).

Como funciona:

- ``ExecutorDeProcessos`` vive na thread da interface e é o que a janela usa.
- Ele mantém uma ``QThread`` de trabalho com um ``_Trabalhador`` dentro. O pedido vai
  por sinal (o Qt entrega na outra thread) e o resultado volta por sinal (o Qt entrega
  de volta na thread da interface). Assim nenhum widget é tocado fora da thread da
  interface, o que o Qt exige.
- Cada pedido tem um número. ``descartar_pendente()`` (ex.: o usuário abriu outra
  imagem no meio do processamento) faz o resultado que ainda vai chegar ser ignorado.
- Um processo por vez: enquanto ``ocupado``, a janela desabilita o "Aplicar".

O NumPy libera o GIL nas operações pesadas, então a interface continua respondendo; até
laços em Python puro cedem a vez à interface a cada poucos milissegundos.
"""

import traceback
from time import perf_counter
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from pdi.core.contrato import ProcessoSpec
from pdi.core.registro import ErroDeImplementacao, executar_processo


class _Trabalhador(QObject):
    """Roda dentro da QThread de trabalho. Não toca em nenhum widget."""

    concluido = Signal(int, object, float)  # id, Resultado, segundos
    falhou = Signal(int, str)  # id, mensagem

    @Slot(int, object, object, object)
    def executar(
        self, id_tarefa: int, spec: ProcessoSpec, img: np.ndarray, parametros: dict[str, Any]
    ) -> None:
        inicio = perf_counter()
        try:
            resultado = executar_processo(spec, img, parametros)
        except (ValueError, ErroDeImplementacao) as e:
            # ValueError: erro de uso (ex.: A > B), mensagem já escrita para o usuário.
            # ErroDeImplementacao: saída fora do contrato, mensagem já diz o responsável.
            self.falhou.emit(id_tarefa, str(e))
            return
        except Exception as e:  # bug no processo: não pode derrubar o programa
            traceback.print_exc()  # detalhes no terminal, para quem for corrigir
            self.falhou.emit(
                id_tarefa,
                f"Erro inesperado em '{spec.nome}' (Integrante {spec.responsavel}): "
                f"{type(e).__name__}: {e}",
            )
            return
        self.concluido.emit(id_tarefa, resultado, perf_counter() - inicio)


class ExecutorDeProcessos(QObject):
    """Executa um processo por vez em segundo plano. Use a partir da thread da interface."""

    concluido = Signal(object, float)  # Resultado, segundos
    falhou = Signal(str)
    ocupado_mudou = Signal(bool)
    _pedir = Signal(int, object, object, object)  # interno: interface → trabalhador

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._id_atual = 0
        self._ocupado = False

        self._thread = QThread(self)
        self._thread.setObjectName("pdi-processamento")
        self._trabalhador = _Trabalhador()
        self._trabalhador.moveToThread(self._thread)
        self._pedir.connect(self._trabalhador.executar)
        self._trabalhador.concluido.connect(self._ao_concluir)
        self._trabalhador.falhou.connect(self._ao_falhar)
        self._thread.finished.connect(self._trabalhador.deleteLater)
        self._thread.start()

    @property
    def ocupado(self) -> bool:
        return self._ocupado

    def executar(self, spec: ProcessoSpec, img: np.ndarray, parametros: dict[str, Any]) -> bool:
        """Envia o pedido e volta na hora. Devolve False se já houver um em andamento."""
        if self._ocupado:
            return False
        self._id_atual += 1
        self._definir_ocupado(True)
        self._pedir.emit(self._id_atual, spec, img, dict(parametros))
        return True

    def descartar_pendente(self) -> None:
        """O resultado do pedido em andamento, quando chegar, será ignorado."""
        self._id_atual += 1

    def encerrar(self) -> None:
        """Para a thread de trabalho (espera o processo atual terminar)."""
        self._thread.quit()
        self._thread.wait()

    def _definir_ocupado(self, ocupado: bool) -> None:
        self._ocupado = ocupado
        self.ocupado_mudou.emit(ocupado)

    def _ao_concluir(self, id_tarefa: int, resultado: object, segundos: float) -> None:
        self._definir_ocupado(False)
        if id_tarefa == self._id_atual:
            self.concluido.emit(resultado, segundos)

    def _ao_falhar(self, id_tarefa: int, mensagem: str) -> None:
        self._definir_ocupado(False)
        if id_tarefa == self._id_atual:
            self.falhou.emit(mensagem)
