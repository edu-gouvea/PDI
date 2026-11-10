"""Fixtures compartilhadas pelos testes de todos os integrantes."""

import numpy as np
import pytest

# Importa todos os processos ANTES de qualquer teste. Se um módulo de processo fosse
# importado pela primeira vez durante um teste com `registro_limpo`, os registros dele
# iriam para o dicionário temporário e se perderiam depois.
import pdi.processos  # noqa: F401
from pdi.core import registro


@pytest.fixture
def registro_limpo(monkeypatch):
    """Substitui o registro global por um vazio durante o teste."""
    vazio: dict = {}
    monkeypatch.setattr(registro, "_PROCESSOS", vazio)
    return vazio


@pytest.fixture
def img_cinza() -> np.ndarray:
    """Imagem cinza 32x48 com gradiente horizontal (valores de 0 a 255)."""
    linha = np.linspace(0, 255, 48).round().astype(np.uint8)
    return np.tile(linha, (32, 1))


@pytest.fixture
def img_rgb() -> np.ndarray:
    """Imagem RGB 32x48 com canais diferentes entre si."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
