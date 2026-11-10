"""Testes dos ruídos: gaussiano aditivo (e sal e pimenta, na próxima etapa)."""

import math
import time

import numpy as np
import pytest

from pdi.core.registro import executar_processo

N = 512  # imagens de 512×512 = 262 144 pixels: estatísticas bem estáveis


def _cinza_medio(valor=128, forma=(N, N)):
    return np.full(forma, valor, np.uint8)


def _gauss(img, **p):
    padrao = {"media": 0.0, "desvio": 20.0, "fracao": 1.0, "semente": 1}
    return executar_processo("ruido_gaussiano", img, {**padrao, **p})


def _diferenca(res, img):
    return res.saidas[0][1].astype(np.float64) - img.astype(np.float64)


# ------------------------------------------------------------- estatística do ruído


@pytest.mark.parametrize(("mu", "sigma"), [(0, 20), (10, 5), (-15, 30)])
def test_media_e_desvio_do_ruido_sao_os_pedidos(mu, sigma):
    # Cinza 128 com |μ| + 4σ < 128: praticamente nada satura, então a diferença
    # saída − entrada É o ruído (arredondado). Erro padrão da média: σ/√N ≈ 0,06.
    img = _cinza_medio()
    d = _diferenca(_gauss(img, media=mu, desvio=sigma), img)
    assert d.mean() == pytest.approx(mu, abs=0.3)
    assert d.std() == pytest.approx(sigma, rel=0.02)


def test_formato_do_histograma_e_gaussiano():
    # Regra 68-95: ~68% dos valores a menos de 1σ da média e ~95% a menos de 2σ.
    # A saída é arredondada: |d| ≤ 20 vale para todo η em [−20,5; 20,5], então o
    # esperado é P(|η| < 20,5) = erf(20,5 / (σ·√2)) ≈ 69,5% (e não 68,3%).
    sigma = 20
    img = _cinza_medio()
    d = _diferenca(_gauss(img, desvio=sigma), img)
    for k in (1, 2):
        esperado = math.erf((k * sigma + 0.5) / (sigma * math.sqrt(2)))
        assert np.mean(np.abs(d) <= k * sigma) == pytest.approx(esperado, abs=0.005)


def test_sigma_zero_soma_so_a_media():
    img = _cinza_medio(100, (8, 8))
    np.testing.assert_array_equal(_gauss(img, media=25, desvio=0).saidas[0][1], 125)


# ------------------------------------------------------------- fração de pixels


@pytest.mark.parametrize("fracao", [0.0, 0.1, 0.3, 0.5, 1.0])
def test_fracao_afeta_exatamente_essa_quantidade_de_pixels(fracao):
    # σ = 0 e μ = 50: todo pixel afetado sobe exatamente 50, os outros ficam iguais.
    img = _cinza_medio(100, (100, 100))
    res = _gauss(img, media=50, desvio=0, fracao=fracao)
    afetados = int(np.sum(res.saidas[0][1] == 150))
    assert afetados == round(fracao * 10_000)
    assert int(np.sum(res.saidas[0][1] == 100)) == 10_000 - afetados


def test_fracao_zero_nao_muda_nada(img_rgb):
    res = _gauss(img_rgb, fracao=0.0)
    np.testing.assert_array_equal(res.saidas[0][1], img_rgb)
    assert res.info["pixels afetados"].startswith("0 de")


# ---------------------------------------------------------------------------- RGB


def test_rgb_mesmos_pixels_nos_tres_canais():
    img = np.full((50, 50, 3), 100, np.uint8)
    saida = _gauss(img, media=50, desvio=0, fracao=0.4).saidas[0][1]
    afetado = saida == 150
    assert (afetado[..., 0] == afetado[..., 1]).all() and (afetado[..., 1] == afetado[..., 2]).all()


def test_rgb_ruido_independente_por_canal():
    img = np.full((N, N, 3), 128, np.uint8)
    d = _diferenca(_gauss(img, desvio=20), img)
    correlacao = np.corrcoef(d[..., 0].ravel(), d[..., 1].ravel())[0, 1]
    assert abs(correlacao) < 0.02  # independentes: correlação ~ 0
    assert saida_tem_cor(img + d)  # o granulado é colorido, não cinza


def saida_tem_cor(img):
    return not np.array_equal(img[..., 0], img[..., 1])


# ------------------------------------------------------------------------- clip


def test_clip_satura_em_0_e_255():
    branco = _cinza_medio(255, (100, 100))
    res = _gauss(branco, media=50, desvio=0)
    assert (res.saidas[0][1] == 255).all()
    assert res.info["valores saturados pelo clip"] == "100.0%"
    preto = _cinza_medio(0, (100, 100))
    assert (_gauss(preto, media=-50, desvio=0).saidas[0][1] == 0).all()


def test_media_da_saida_desvia_quando_ha_saturacao():
    # Num pixel claro, o ruído para cima é cortado e o para baixo não: a média cai.
    img = _cinza_medio(250)
    d = _diferenca(_gauss(img, desvio=20), img)
    assert d.mean() < -5


# ---------------------------------------------------------------------- semente


def test_mesma_semente_mesmo_ruido(img_rgb):
    a = _gauss(img_rgb, semente=42).saidas[0][1]
    b = _gauss(img_rgb, semente=42).saidas[0][1]
    c = _gauss(img_rgb, semente=43).saidas[0][1]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_semente_zero_sorteia_e_informa_a_semente_usada(img_rgb):
    res1 = _gauss(img_rgb, semente=0)
    res2 = _gauss(img_rgb, semente=0)
    s1 = res1.info["semente usada"]
    assert 1 <= s1 <= 999_999
    assert not np.array_equal(res1.saidas[0][1], res2.saidas[0][1])  # aleatório de fato
    # Digitando a semente informada, o resultado se repete.
    np.testing.assert_array_equal(_gauss(img_rgb, semente=s1).saidas[0][1], res1.saidas[0][1])


# -------------------------------------------------------------------- info e tempo


def test_info_mostra_ruido_sorteado():
    info = _gauss(_cinza_medio(), media=5, desvio=10).info
    assert info["pixels afetados"] == "262144 de 262144 (100.0%)"
    assert info["ruído sorteado: média"] == pytest.approx(5, abs=0.1)
    assert info["ruído sorteado: desvio"] == pytest.approx(10, abs=0.1)


def test_rapido_em_512x512_rgb():
    img = np.zeros((N, N, 3), np.uint8)
    inicio = time.perf_counter()
    _gauss(img, fracao=0.5)
    assert time.perf_counter() - inicio < 0.5


# ----------------------------------------------------------------- sal e pimenta


def _sp(img, **p):
    padrao = {"tipo": "sal e pimenta", "probabilidade": 0.1, "proporcao_sal": 0.5, "semente": 1}
    return executar_processo("ruido_sal_pimenta", img, {**padrao, **p})


def _contar(saida, valor):
    """Pixels iguais a ``valor`` (em RGB, nos três canais)."""
    igual = saida == valor
    return int(np.sum(igual if saida.ndim == 2 else igual.all(axis=-1)))


@pytest.mark.parametrize(
    ("prob", "prop", "sal", "pimenta"),
    [(0.1, 0.5, 500, 500), (0.1, 0.3, 300, 700), (0.25, 1.0, 2500, 0), (0.05, 0.0, 0, 500)],
)
def test_quantidade_exata_de_sal_e_pimenta(prob, prop, sal, pimenta):
    img = _cinza_medio(128, (100, 100))  # 10 000 pixels, nenhum 0 ou 255 de antemão
    saida = _sp(img, probabilidade=prob, proporcao_sal=prop).saidas[0][1]
    assert _contar(saida, 255) == sal
    assert _contar(saida, 0) == pimenta
    assert _contar(saida, 128) == 10_000 - sal - pimenta  # o resto não muda


@pytest.mark.parametrize(("tipo", "valor"), [("sal", 255), ("pimenta", 0)])
def test_so_sal_ou_so_pimenta_ignora_a_proporcao(tipo, valor):
    img = _cinza_medio(128, (100, 100))
    saida = _sp(img, tipo=tipo, probabilidade=0.2, proporcao_sal=0.5).saidas[0][1]
    assert _contar(saida, valor) == 2000
    assert _contar(saida, 255 - valor) == 0


def test_probabilidade_zero_e_um():
    img = _cinza_medio(128, (50, 50))
    np.testing.assert_array_equal(_sp(img, probabilidade=0.0).saidas[0][1], img)
    tudo = _sp(img, probabilidade=1.0).saidas[0][1]
    assert set(np.unique(tudo).tolist()) == {0, 255}


def test_rgb_pixel_inteiro_vira_branco_ou_preto():
    img = np.full((100, 100, 3), (200, 50, 120), np.uint8)
    saida = _sp(img, probabilidade=0.2).saidas[0][1]
    alterado = (saida != img).any(axis=-1)
    assert int(alterado.sum()) == 2000
    # Todo pixel alterado é (255, 255, 255) ou (0, 0, 0): nada de pontos coloridos.
    cores = {tuple(c) for c in saida[alterado].tolist()}
    assert cores == {(255, 255, 255), (0, 0, 0)}


def test_espalhado_uniformemente_pela_imagem():
    img = _cinza_medio()
    saida = _sp(img, probabilidade=0.2).saidas[0][1]
    corrompido = saida != 128
    metade = N // 2
    for quadrante in (
        corrompido[:metade, :metade],
        corrompido[:metade, metade:],
        corrompido[metade:, :metade],
        corrompido[metade:, metade:],
    ):
        assert quadrante.mean() == pytest.approx(0.2, abs=0.01)


def test_sal_e_pimenta_semente(img_rgb):
    a = _sp(img_rgb, semente=5).saidas[0][1]
    np.testing.assert_array_equal(a, _sp(img_rgb, semente=5).saidas[0][1])
    assert not np.array_equal(a, _sp(img_rgb, semente=6).saidas[0][1])
    res = _sp(img_rgb, semente=0)
    reproduzido = _sp(img_rgb, semente=res.info["semente usada"]).saidas[0][1]
    np.testing.assert_array_equal(reproduzido, res.saidas[0][1])


def test_sal_e_pimenta_info():
    res = _sp(_cinza_medio(128, (100, 100)), probabilidade=0.1, proporcao_sal=0.3)
    assert res.info["pixels com sal (255)"] == "300 (3.0%)"
    assert res.info["pixels com pimenta (0)"] == "700 (7.0%)"
    assert res.saidas[0][0] == "Com ruído sal e pimenta"


def test_mediana_remove_sal_e_pimenta():
    # Propriedade que justifica o processo na apresentação: com p baixo, a mediana 3×3
    # (aqui a do OpenCV, só como referência) recupera quase toda a imagem lisa.
    import cv2

    img = _cinza_medio(128, (200, 200))
    ruidosa = _sp(img, probabilidade=0.05).saidas[0][1]
    filtrada = cv2.medianBlur(ruidosa, 3)
    assert np.mean(filtrada == 128) > 0.99


def test_sal_e_pimenta_rapido_em_512x512_rgb():
    img = np.zeros((N, N, 3), np.uint8)
    inicio = time.perf_counter()
    _sp(img, probabilidade=0.5)
    assert time.perf_counter() - inicio < 0.5
