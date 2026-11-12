"""Fórmulas de intensidade, ganhos, limites e CDF discreta do Gonzalez."""

import math

import numpy as np
import pytest
from skimage.exposure import equalize_hist

from pdi.core.contrato import Histograma
from pdi.core.registro import executar_processo, listar_processos


def _saida(id_processo, img, **parametros):
    return executar_processo(id_processo, img, parametros).saidas[0][1]


@pytest.mark.parametrize("k", [0, 127, 255])
def test_limiar_usa_desigualdade_estrita(k):
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    saida = _saida("limiarizacao", img, k=k)
    assert np.all(saida[img <= k] == 0)
    assert np.all(saida[img > k] == 255)


@pytest.mark.parametrize("c", [0, 10, 50, 200])
def test_log_confere_com_formula_escalar(c):
    img = np.array([[0, 1, 2, 10, 64, 128, 255]], dtype=np.uint8)
    esperado = [[min(255, round(c * math.log(1 + int(r)))) for r in img[0]]]
    np.testing.assert_array_equal(_saida("transformacao_log", img, c=c), esperado)


def test_log_padrao_mapeia_extremos_sem_min_max():
    img = np.array([[0, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(_saida("transformacao_log", img), [[0, 255]])
    # Uma imagem constante intermediária precisa manter o seu valor absoluto.
    np.testing.assert_array_equal(
        _saida("transformacao_log", np.full((2, 2), 15, np.uint8), c=10), np.full((2, 2), 28)
    )


@pytest.mark.parametrize("gama", [0.01, 0.5, 1, 2, 25])
@pytest.mark.parametrize("c", [0, 1, 10])
def test_potencia_confere_com_formula_escalar(c, gama):
    img = np.array([[0, 1, 64, 128, 255]], dtype=np.uint8)
    esperado = [[min(255, round(255 * c * (int(r) / 255) ** gama)) for r in img[0]]]
    np.testing.assert_array_equal(_saida("transformacao_potencia", img, c=c, gama=gama), esperado)


def test_gama_um_e_identidade_e_gama_muda_brilho(img_cinza):
    np.testing.assert_array_equal(_saida("transformacao_potencia", img_cinza), img_cinza)
    clara = _saida("transformacao_potencia", img_cinza, gama=0.5)
    escura = _saida("transformacao_potencia", img_cinza, gama=2)
    assert np.all(clara >= img_cinza) and np.any(clara > img_cinza)
    assert np.all(escura <= img_cinza) and np.any(escura < img_cinza)


@pytest.mark.parametrize("processo", ["transformacao_log", "transformacao_potencia"])
def test_ganho_altera_saida_sem_normalizacao(processo):
    img = np.array([[0, 16, 64, 128, 255]], dtype=np.uint8)
    c1, c2 = (10, 50) if processo == "transformacao_log" else (0.5, 1)
    menor = _saida(processo, img, c=c1)
    maior = _saida(processo, img, c=c2)
    assert np.all(maior >= menor) and np.any(maior > menor)


@pytest.mark.parametrize("processo", ["transformacao_log", "transformacao_potencia"])
def test_transformacao_rgb_equivale_a_canais_independentes(processo, img_rgb):
    obtido = _saida(processo, img_rgb, c=2)
    esperado = np.stack([_saida(processo, img_rgb[..., c], c=2) for c in range(3)], axis=-1)
    np.testing.assert_array_equal(obtido, esperado)


def test_equalizacao_cdf_conhecida_e_acessorios():
    img = np.array([[0, 0, 1, 3]], dtype=np.uint8)
    res = executar_processo("equalizacao_histograma", img)
    np.testing.assert_array_equal(res.saidas[0][1], [[128, 128, 191, 255]])
    assert len(res.acessorios) == 2
    for acessorio, titulo, imagem in zip(
        res.acessorios, ["Antes", "Depois"], [img, res.saidas[0][1]], strict=True
    ):
        assert isinstance(acessorio, Histograma) and acessorio.titulo == titulo
        assert acessorio.valores.shape == (256,) and acessorio.valores.sum() == img.size
        for nivel in range(256):
            assert acessorio.valores[nivel] == np.count_nonzero(imagem == nivel)


def test_equalizacao_confere_com_scikit_image(img_cinza):
    esperado = np.rint(255 * equalize_hist(img_cinza)).astype(np.uint8)
    np.testing.assert_allclose(_saida("equalizacao_histograma", img_cinza), esperado, atol=1)


@pytest.mark.parametrize("nivel", [0, 100, 255])
def test_equalizacao_constante_segue_cdf_sem_subtrair_minimo(nivel):
    img = np.full((3, 4), nivel, np.uint8)
    np.testing.assert_array_equal(_saida("equalizacao_histograma", img), np.full((3, 4), 255))


def test_equalizacao_rampa_uniforme():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    esperado = np.rint(np.arange(1, 257) * 255 / 256).astype(np.uint8).reshape(16, 16)
    np.testing.assert_array_equal(_saida("equalizacao_histograma", img), esperado)


@pytest.mark.parametrize("fundo", ["preservar", "zerar"])
def test_fatiamento_inclui_os_limites(fundo):
    img = np.array([[0, 99, 100, 150, 180, 181, 255]], dtype=np.uint8)
    fora = [0, 99, 181, 255] if fundo == "preservar" else [0, 0, 0, 0]
    esperado = [[fora[0], fora[1], 255, 255, 255, fora[2], fora[3]]]
    np.testing.assert_array_equal(_saida("fatiamento_intensidade", img, fundo=fundo), esperado)


@pytest.mark.parametrize("nivel", [0, 128, 255])
def test_fatiamento_faixa_unitaria(nivel):
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    saida = _saida("fatiamento_intensidade", img, a=nivel, b=nivel, fundo="zerar")
    assert np.count_nonzero(saida) == 1 and saida.ravel()[nivel] == 255


def test_fatiamento_recusa_faixa_invertida(img_cinza):
    with pytest.raises(ValueError, match="A deve ser menor ou igual a B"):
        executar_processo("fatiamento_intensidade", img_cinza, {"a": 181, "b": 100})


@pytest.mark.parametrize("spec", [s for s in listar_processos() if s.grupo == "Intensidade"])
def test_intensidade_aceita_entrada_somente_leitura_sem_alterar(spec, img_cinza):
    original = img_cinza.copy()
    img_cinza.flags.writeable = False
    res = spec.funcao(img_cinza, {p.nome: p.padrao for p in spec.parametros})
    assert res.saidas[0][1].dtype == np.uint8
    assert not np.shares_memory(res.saidas[0][1], img_cinza)
    np.testing.assert_array_equal(img_cinza, original)
