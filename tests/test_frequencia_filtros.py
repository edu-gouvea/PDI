"""Máscaras analíticas, DFT independente e comportamento de filtros na frequência."""

import math

import numpy as np
import pytest

from pdi.core.frequencia import fft_centralizada, inverter_fft
from pdi.core.registro import executar_processo, obter_processo
from pdi.processos.frequencia_filtros import _mascara_butterworth, _mascara_gaussiana

PROCESSOS = ["filtro_gaussiano_frequencia", "filtro_butterworth"]


def test_mascaras_analiticas_e_ordem():
    d = np.array([0.0, 5, 10, 20, 100])
    g = _mascara_gaussiana(d, 10)
    assert g[0] == 1 and g[2] == pytest.approx(math.exp(-0.5))
    assert np.all(np.diff(g) < 0)
    for ordem in (1, 2, 20):
        b = _mascara_butterworth(d, 10, ordem)
        assert b[0] == 1 and b[2] == 0.5
        assert np.all((b >= 0) & (b <= 1))
    suave, abrupta = _mascara_butterworth(d, 10, 1), _mascara_butterworth(d, 10, 20)
    assert abrupta[1] > suave[1] and abrupta[3] < suave[3]


def test_butterworth_potencia_grande_tem_limite_finito():
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(_mascara_butterworth(np.array([0, 1e30]), 1, 20), [1, 0])


@pytest.mark.parametrize("processo", PROCESSOS)
@pytest.mark.parametrize("tipo", ["passa-baixa", "passa-alta"])
def test_pipeline_confere_com_dft_matricial_explicita(processo, tipo):
    img = np.array([[0, 15, 100, 90], [10, 240, 60, 30], [200, 5, 120, 70]], np.uint8)
    expandida = np.pad(img, ((1, 2), (2, 2)), mode="reflect").astype(np.float64)
    h, w = expandida.shape
    eh = np.exp(-2j * np.pi * np.outer(np.arange(h), np.arange(h)) / h)
    ew = np.exp(-2j * np.pi * np.outer(np.arange(w), np.arange(w)) / w)
    f = np.roll(eh @ expandida @ ew.T, (h // 2, w // 2), axis=(0, 1))
    mascara = np.empty((h, w))
    for y, x in np.ndindex(mascara.shape):
        d = math.hypot(y - h // 2, x - w // 2)
        baixa = math.exp(-d * d / 8) if processo == PROCESSOS[0] else 1 / (1 + (d / 2) ** 6)
        mascara[y, x] = 1 - baixa if tipo == "passa-alta" else baixa
    filtrada = np.roll(f * mascara, (-h // 2, -w // 2), axis=(0, 1))
    espacial = (eh.conj().T @ filtrada @ ew.conj() / (h * w)).real[1:4, 2:6]
    if tipo == "passa-alta":
        esperado = np.rint((espacial - espacial.min()) * 255 / np.ptp(espacial))
    else:
        esperado = np.clip(np.rint(espacial), 0, 255)
    parametros = {"d0": 2.0, "tipo": tipo}
    if processo == PROCESSOS[1]:
        parametros["ordem"] = 3
    res = executar_processo(processo, img, parametros)
    np.testing.assert_allclose(res.saidas[0][1], esperado, atol=1)


@pytest.mark.parametrize("processo", PROCESSOS)
def test_mascaras_complementares_e_soma_das_respostas(processo, img_cinza):
    baixa = executar_processo(processo, img_cinza, {"tipo": "passa-baixa", "d0": 4})
    alta = executar_processo(processo, img_cinza, {"tipo": "passa-alta", "d0": 4})
    h1, h2 = baixa.acessorios[1][1], alta.acessorios[1][1]
    np.testing.assert_allclose(h1.astype(int) + h2.astype(int), 255, atol=1)
    f = fft_centralizada(img_cinza)
    yy, xx = np.indices(f.shape)
    d = np.hypot(yy - f.shape[0] // 2, xx - f.shape[1] // 2)
    h = _mascara_gaussiana(d, 4) if processo == PROCESSOS[0] else _mascara_butterworth(d, 4, 2)
    reconstruida = inverter_fft(f * h, img_cinza.shape) + inverter_fft(f * (1 - h), img_cinza.shape)
    np.testing.assert_allclose(reconstruida, img_cinza, atol=1e-10)


@pytest.mark.parametrize("processo", PROCESSOS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (3, 1), (3, 5), (2, 3, 3)])
@pytest.mark.parametrize("nivel", [0, 83, 255])
def test_constante_preservada_em_pb_e_anulada_em_pa(processo, shape, nivel):
    img = np.full(shape, nivel, np.uint8)
    baixa = executar_processo(processo, img, {"tipo": "passa-baixa"})
    alta = executar_processo(processo, img, {"tipo": "passa-alta"})
    np.testing.assert_array_equal(baixa.saidas[0][1], img)
    np.testing.assert_array_equal(alta.saidas[0][1], np.zeros_like(img))
    np.testing.assert_array_equal(alta.acessorios[2][1], np.zeros_like(alta.acessorios[2][1]))


@pytest.mark.parametrize("processo", PROCESSOS)
def test_acessorios_tem_filtro_e_espectros_reais(processo, img_rgb):
    res = executar_processo(processo, img_rgb)
    assert [a[0] for a in res.acessorios] == [
        "Espectro original (log)",
        "Filtro H(u,v)",
        "Espectro filtrado (log)",
    ]
    assert res.acessorios[0][1].shape == (64, 96, 3)
    assert res.acessorios[1][1].shape == (64, 96)
    assert res.acessorios[2][1].shape == (64, 96, 3)
    mascara = res.acessorios[1][1]
    assert mascara[32, 48] == 255 and mascara[0, 0] < 255
    for _, acessorio in res.acessorios:
        assert acessorio.dtype == np.uint8


@pytest.mark.parametrize("processo", PROCESSOS)
def test_corte_muda_resultado_e_corte_grande_preserva_imagem(processo, img_rgb):
    baixa = executar_processo(processo, img_rgb, {"d0": 1}).saidas[0][1]
    alta = executar_processo(processo, img_rgb, {"d0": 1000}).saidas[0][1]
    assert np.any(baixa != alta)
    np.testing.assert_allclose(alta, img_rgb, atol=1)
    assert np.std(baixa.astype(float)) < np.std(img_rgb.astype(float))


def test_ordem_butterworth_muda_resultado(img_rgb):
    a = executar_processo(PROCESSOS[1], img_rgb, {"d0": 10, "ordem": 1}).saidas[0][1]
    b = executar_processo(PROCESSOS[1], img_rgb, {"d0": 10, "ordem": 20}).saidas[0][1]
    assert np.any(a != b)


@pytest.mark.parametrize("processo", PROCESSOS)
def test_pb_rgb_equivale_a_canais_independentes(processo, img_rgb):
    res = executar_processo(processo, img_rgb).saidas[0][1]
    canais = [executar_processo(processo, img_rgb[..., c]).saidas[0][1] for c in range(3)]
    np.testing.assert_array_equal(res, np.stack(canais, axis=-1))


@pytest.mark.parametrize("processo", PROCESSOS)
def test_entrada_somente_leitura_e_nao_contigua(processo, img_rgb):
    img = img_rgb[::2, ::2]
    original = img.copy()
    img.flags.writeable = False
    spec = obter_processo(processo)
    res = spec.funcao(img, {p.nome: p.padrao for p in spec.parametros})
    np.testing.assert_array_equal(img, original)
    assert res.saidas[0][1].shape == img.shape
    assert not np.shares_memory(res.saidas[0][1], img)
