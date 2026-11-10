"""Testes do contrato (dataclasses) e de util.tipo_da_imagem."""

import dataclasses

import numpy as np
import pytest

from pdi.core.contrato import Histograma, Param, ProcessoSpec, Resultado, TipoParam
from pdi.core.util import TIPO_CINZA, TIPO_RGB, tipo_da_imagem


def test_param_e_imutavel():
    p = Param("k", "Limiar k", TipoParam.INT, 127, 0, 255)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.padrao = 10  # type: ignore[misc]


def test_param_valores_padrao():
    p = Param("modo", "Modo", TipoParam.BOOL, True)
    assert p.minimo is None and p.maximo is None and p.passo is None
    assert p.opcoes == () and p.ajuda == "" and p.somente_impar is False


def test_processo_spec_e_imutavel():
    spec = ProcessoSpec("x", "X", "Cor", 1, True, True, (), lambda img, p: None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.nome = "Y"  # type: ignore[misc]


def test_resultado_listas_nao_sao_compartilhadas():
    # field(default_factory=list): cada Resultado tem as próprias listas.
    a = Resultado(saidas=[])
    b = Resultado(saidas=[])
    a.acessorios.append(Histograma("h", np.zeros(256)))
    a.info["x"] = 1
    assert b.acessorios == [] and b.info == {}


def test_tipo_da_imagem(img_cinza, img_rgb):
    assert tipo_da_imagem(img_cinza) == TIPO_CINZA
    assert tipo_da_imagem(img_rgb) == TIPO_RGB


@pytest.mark.parametrize(
    "img",
    [
        np.zeros((4, 4), dtype=np.float64),  # dtype errado
        np.zeros((4, 4, 4), dtype=np.uint8),  # RGBA
        np.zeros((4, 4, 1), dtype=np.uint8),  # canal único em 3D
        np.zeros((0, 4), dtype=np.uint8),  # vazia
        np.zeros(4, dtype=np.uint8),  # 1D
        [[0, 1], [2, 3]],  # não é ndarray
    ],
)
def test_tipo_da_imagem_invalida(img):
    with pytest.raises(ValueError):
        tipo_da_imagem(img)
