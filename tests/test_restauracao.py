"""Mediana adaptativa: estágios A/B, crescimento, fallback e referência escalar."""

import numpy as np
import pytest

from pdi.core.registro import executar_processo, obter_processo


def _referencia_escalar(canal, janela_max):
    raio = janela_max // 2
    borda = np.pad(canal, raio, mode="reflect")
    saida = np.empty_like(canal)
    for y, x in np.ndindex(canal.shape):
        for tamanho in range(3, janela_max + 1, 2):
            r = tamanho // 2
            viz = borda[y + raio - r : y + raio + r + 1, x + raio - r : x + raio + r + 1]
            ordenados = sorted(int(v) for v in viz.ravel())
            minimo, maximo, mediana = ordenados[0], ordenados[-1], ordenados[len(ordenados) // 2]
            pixel = int(canal[y, x])
            if minimo < mediana < maximo:
                saida[y, x] = pixel if minimo < pixel < maximo else mediana
                break
            if tamanho == janela_max:
                saida[y, x] = mediana
    return saida


@pytest.mark.parametrize("shape", [(1, 1), (1, 6), (4, 1), (7, 9)])
@pytest.mark.parametrize("janela_max", [3, 7, 31])
def test_confere_com_referencia_escalar(shape, janela_max):
    rng = np.random.default_rng(43)
    img = rng.choice(np.array([0, 75, 120, 180, 255], np.uint8), size=shape)
    res = executar_processo("mediana_adaptativa", img, {"janela_max": janela_max})
    np.testing.assert_array_equal(res.saidas[0][1], _referencia_escalar(img, janela_max))


def test_estagio_a_cresce_ate_encontrar_mediana_nao_impulsiva():
    img = np.full((5, 5), 100, np.uint8)
    img[1:4, 1:4] = 0
    img[0, 0] = 255
    curta = executar_processo("mediana_adaptativa", img, {"janela_max": 3}).saidas[0][1]
    longa = executar_processo("mediana_adaptativa", img, {"janela_max": 5}).saidas[0][1]
    assert curta[2, 2] == 0 and longa[2, 2] == 100


def test_estagio_b_preserva_pixel_intermediario_em_vez_de_substituir_por_mediana():
    img = np.array([[0, 100, 255], [100, 128, 100], [100, 100, 100]], np.uint8)
    saida = executar_processo("mediana_adaptativa", img).saidas[0][1]
    assert np.median(img) == 100 and saida[1, 1] == 128


@pytest.mark.parametrize("impulso", [0, 255])
def test_estagio_b_substitui_extremo_por_mediana(impulso):
    img = np.array([[0, 100, 255], [100, impulso, 100], [100, 100, 100]], np.uint8)
    saida = executar_processo("mediana_adaptativa", img).saidas[0][1]
    assert saida[1, 1] == 100


@pytest.mark.parametrize("janela_max", [3, 7, 31])
def test_imagem_de_dois_niveis_usa_fallback_correto(janela_max):
    img = np.random.default_rng(4).choice(np.array([20, 230], np.uint8), size=(9, 11))
    saida = executar_processo("mediana_adaptativa", img, {"janela_max": janela_max}).saidas[0][1]
    np.testing.assert_array_equal(saida, _referencia_escalar(img, janela_max))


@pytest.mark.parametrize("nivel", [0, 83, 255])
def test_constante_e_janela_maxima(nivel):
    img = np.full((2, 3, 3), nivel, np.uint8)
    saida = executar_processo("mediana_adaptativa", img, {"janela_max": 31}).saidas[0][1]
    np.testing.assert_array_equal(saida, img)


def test_reduz_ruido_impulsivo_denso():
    rng = np.random.default_rng(5)
    ruidosa = np.full((32, 48), 128, np.uint8)
    amostra = rng.random(ruidosa.shape)
    ruidosa[amostra < 0.3] = 0
    ruidosa[amostra > 0.7] = 255
    saida = executar_processo("mediana_adaptativa", ruidosa).saidas[0][1]
    antes = np.mean(np.abs(ruidosa.astype(float) - 128))
    depois = np.mean(np.abs(saida.astype(float) - 128))
    assert depois < antes / 10


def test_rgb_equivale_a_canais_independentes(img_rgb):
    obtido = executar_processo("mediana_adaptativa", img_rgb).saidas[0][1]
    esperado = np.stack(
        [executar_processo("mediana_adaptativa", img_rgb[..., c]).saidas[0][1] for c in range(3)],
        axis=-1,
    )
    np.testing.assert_array_equal(obtido, esperado)


def test_entrada_somente_leitura_e_nao_contigua(img_rgb):
    img = img_rgb[::2, ::2]
    original = img.copy()
    img.flags.writeable = False
    spec = obter_processo("mediana_adaptativa")
    res = spec.funcao(img, {p.nome: p.padrao for p in spec.parametros})
    np.testing.assert_array_equal(img, original)
    assert res.saidas[0][1].shape == img.shape and res.saidas[0][1].dtype == np.uint8
    assert not np.shares_memory(img, res.saidas[0][1])
