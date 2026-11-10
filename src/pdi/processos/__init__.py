"""Processos de PDI.

Importar este pacote importa todos os módulos abaixo, o que executa os decoradores
``@registrar_processo`` e faz cada processo aparecer na interface. Ao criar um módulo
novo de processos, adicione-o aqui.
"""

from pdi.processos import (
    cor,
    frequencia_filtros,
    intensidade,
    realce,
    restauracao,
    ruido,
    suavizacao,
)

__all__ = [
    "cor",
    "frequencia_filtros",
    "intensidade",
    "realce",
    "restauracao",
    "ruido",
    "suavizacao",
]
