"""Teste de fumaça da estrutura: todos os módulos do pacote importam sem erro.

Se alguém criar um módulo com erro de sintaxe ou import quebrado, este teste acusa.
Os módulos de UI ficam de fora porque dependem de PySide6/display e terão testes próprios.
"""

import importlib
import pkgutil

import pytest

import pdi


def _modulos_do_pacote() -> list[str]:
    nomes = []
    for info in pkgutil.walk_packages(pdi.__path__, prefix="pdi."):
        if info.name.startswith("pdi.ui") or info.name == "pdi.__main__":
            continue
        nomes.append(info.name)
    return sorted(nomes)


@pytest.mark.parametrize("nome", _modulos_do_pacote())
def test_modulo_importa(nome: str) -> None:
    importlib.import_module(nome)


def test_pacotes_principais_existem() -> None:
    nomes = set(_modulos_do_pacote())
    for esperado in ("pdi.core.contrato", "pdi.core.registro", "pdi.processos.cor"):
        assert esperado in nomes
