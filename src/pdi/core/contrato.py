"""Contrato de processos: Param, ProcessoSpec, Histograma e Resultado (I1).

ATENÇÃO: os três integrantes dependem deste arquivo. Não altere sem combinar com o grupo.

Fluxo:
1. Cada processo é uma função ``f(img, p) -> Resultado``, registrada com
   ``@registrar_processo`` (ver ``pdi.core.registro``).
2. ``img`` é ``np.ndarray`` uint8, cinza ``(H, W)`` ou RGB ``(H, W, 3)``, e chega
   **somente leitura**: o processo não pode modificá-la.
3. ``p`` é um dict ``{Param.nome: valor}``, já validado e completo (faltantes recebem
   ``Param.padrao``).
4. A interface lê os ``Param`` para gerar os controles e exibe o ``Resultado``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TipoParam(Enum):
    """Tipo de um parâmetro; define o controle gerado na interface."""

    INT = "int"  # spinbox inteira + slider
    FLOAT = "float"  # spinbox decimal + slider
    BOOL = "bool"  # caixa de seleção
    ESCOLHA = "escolha"  # lista de opções (ex.: "passa-alta" / "passa-baixa")


@dataclass(frozen=True)
class Param:
    """Descrição de um parâmetro ajustável de um processo."""

    nome: str  # chave usada no dict de parâmetros (ex.: "sigma")
    rotulo: str  # texto exibido na interface (ex.: "Desvio padrão σ")
    tipo: TipoParam
    padrao: Any
    minimo: float | None = None  # obrigatório para INT e FLOAT
    maximo: float | None = None  # obrigatório para INT e FLOAT
    passo: float | None = None
    opcoes: tuple[str, ...] = ()  # obrigatório para ESCOLHA
    ajuda: str = ""  # tooltip curto explicando o efeito do parâmetro
    somente_impar: bool = False  # tamanho de janela: 3, 5, 7...


@dataclass
class Histograma:
    """Histograma para a área de acessórios."""

    titulo: str
    valores: np.ndarray  # shape (256,), contagens ou probabilidades


@dataclass
class Resultado:
    """O que um processo devolve para a interface."""

    # [(titulo, imagem uint8)]; a 1ª é a "principal", que é salva por padrão.
    saidas: list[tuple[str, np.ndarray]]
    # Imagens extras (ex.: Laplaciano ajustado) e/ou histogramas.
    acessorios: list[tuple[str, np.ndarray] | Histograma] = field(default_factory=list)
    # Dados exibidos em texto (ex.: {"limiar usado": 127}).
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessoSpec:
    """Especificação completa de um processo, criada pelo decorador de registro."""

    id: str  # "decomposicao_rgb"
    nome: str  # "Decomposição RGB"
    grupo: str  # "Cor", "Intensidade", "Espacial", "Frequência", "Ruído", "Restauração"
    responsavel: int  # 1, 2 ou 3
    aceita_cinza: bool
    aceita_rgb: bool
    parametros: tuple[Param, ...]
    funcao: Callable[[np.ndarray, dict[str, Any]], Resultado]
    descricao: str = ""  # 1–2 frases, exibida na interface
