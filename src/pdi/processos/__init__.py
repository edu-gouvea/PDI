"""Processos de PDI.

Importar este pacote importa todos os módulos abaixo, o que executa os decoradores
``@registrar_processo`` e faz cada processo aparecer na interface. Ao criar um módulo
novo de processos, adicione-o aqui.

A ORDEM DOS IMPORTS IMPORTA: ela define a ordem dos processos dentro de cada grupo
na interface, que segue a ordem da especificação do professor. Por isso o isort está
desligado neste arquivo.
"""

# ruff: noqa: I001
from pdi.processos import cor  # 1-2
from pdi.processos import intensidade  # 3-7
from pdi.processos import suavizacao  # 8-9
from pdi.processos import realce  # 10-12
from pdi.processos import frequencia_filtros  # 13-14
from pdi.processos import restauracao  # 15
from pdi.processos import ruido  # 16-17

__all__ = [
    "cor",
    "intensidade",
    "suavizacao",
    "realce",
    "frequencia_filtros",
    "restauracao",
    "ruido",
]
