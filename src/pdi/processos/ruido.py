"""Ruído aditivo gaussiano e ruídos sal, pimenta e sal e pimenta (I1)."""

from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.registro import registrar_processo
from pdi.core.util import clip_para_uint8

SEMENTE = Param(
    "semente",
    "Semente",
    TipoParam.INT,
    0,
    minimo=0,
    maximo=999_999,
    passo=1,
    ajuda="0 = ruído diferente a cada execução. Outro valor = mesmo ruído sempre (reprodutível).",
)


def _gerador(semente: int) -> tuple[np.random.Generator, int]:
    """Gerador de números aleatórios e a semente efetivamente usada.

    Semente 0 = sorteia uma semente nova. Ela é devolvida para aparecer nos dados do
    resultado: digitando-a de volta, o mesmo ruído é reproduzido exatamente.
    """
    if semente == 0:
        semente = int(np.random.default_rng().integers(1, SEMENTE.maximo + 1))
    return np.random.default_rng(semente), semente


def _sortear_indices(rng: np.random.Generator, total: int, fracao: float) -> np.ndarray:
    """EXATAMENTE round(fração·total) índices distintos, em ordem aleatória."""
    return rng.choice(total, size=round(fracao * total), replace=False)


def _sortear_pixels(rng: np.random.Generator, forma: tuple[int, int], fracao: float) -> np.ndarray:
    """Máscara (H, W) com EXATAMENTE round(fração·N) pixels True, sorteados sem repetição."""
    mascara = np.zeros(forma[0] * forma[1], dtype=bool)
    mascara[_sortear_indices(rng, mascara.size, fracao)] = True
    return mascara.reshape(forma)


@registrar_processo(
    id="ruido_gaussiano",
    nome="Ruído aditivo gaussiano",
    grupo="Ruído",
    responsavel=1,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        Param(
            "media",
            "Média μ",
            TipoParam.FLOAT,
            0.0,
            minimo=-128.0,
            maximo=128.0,
            passo=1.0,
            ajuda="Deslocamento médio do ruído. μ > 0 clareia, μ < 0 escurece.",
        ),
        Param(
            "desvio",
            "Desvio padrão σ",
            TipoParam.FLOAT,
            20.0,
            minimo=0.0,
            maximo=128.0,
            passo=1.0,
            ajuda="Intensidade do ruído. Maior σ = granulado mais forte.",
        ),
        Param(
            "fracao",
            "Fração de pixels afetados (0 a 1)",
            TipoParam.FLOAT,
            1.0,
            minimo=0.0,
            maximo=1.0,
            passo=0.05,
            ajuda="1 = ruído em todos os pixels; 0,3 = em 30% deles, sorteados.",
        ),
        SEMENTE,
    ),
    descricao=(
        "Soma ruído η ~ N(μ, σ²) aos pixels sorteados: g = f + η. Em RGB, cada canal "
        "recebe um ruído independente. Saturação por clip em [0, 255]."
    ),
)
def ruido_gaussiano(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    rng, semente = _gerador(p["semente"])
    f = img.astype(np.float64)

    # η ~ N(μ, σ²): um valor sorteado para cada pixel e cada canal (canais independentes).
    eta = rng.normal(loc=p["media"], scale=p["desvio"], size=img.shape)

    # Só os pixels sorteados recebem ruído; nos demais, η = 0. Em RGB, a mesma máscara
    # vale para os 3 canais (o pixel é afetado ou não), mas o valor de η difere por canal.
    afetados = _sortear_pixels(rng, img.shape[:2], p["fracao"])
    mascara = afetados if img.ndim == 2 else afetados[..., np.newaxis]
    eta = np.where(mascara, eta, 0.0)

    g = f + eta
    saida = clip_para_uint8(g)  # valores < 0 viram 0 e > 255 viram 255

    total = afetados.size
    k = int(afetados.sum())
    eta_afetados = eta[afetados]  # só os valores realmente somados
    saturados = float(np.mean((g < -0.5) | (g >= 255.5))) * 100  # os que o clip cortou
    info: dict[str, Any] = {
        "pixels afetados": f"{k} de {total} ({100 * k / total:.1f}%)",
        "semente usada": semente,
    }
    if k:
        info["ruído sorteado: média"] = round(float(eta_afetados.mean()), 2)
        info["ruído sorteado: desvio"] = round(float(eta_afetados.std()), 2)
    info["valores saturados pelo clip"] = f"{saturados:.1f}%"
    return Resultado(saidas=[("Com ruído gaussiano", saida)], info=info)


@registrar_processo(
    id="ruido_sal_pimenta",
    nome="Ruído sal, pimenta e sal e pimenta",
    grupo="Ruído",
    responsavel=1,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        Param(
            "tipo",
            "Tipo",
            TipoParam.ESCOLHA,
            "sal e pimenta",
            opcoes=("sal e pimenta", "sal", "pimenta"),
            ajuda="Sal = pixels brancos (255); pimenta = pixels pretos (0).",
        ),
        Param(
            "probabilidade",
            "Probabilidade p (0 a 1)",
            TipoParam.FLOAT,
            0.05,
            minimo=0.0,
            maximo=1.0,
            passo=0.01,
            ajuda="Fração de pixels corrompidos. p = 0,1 corrompe exatamente 10% dos pixels.",
        ),
        Param(
            "proporcao_sal",
            "Proporção de sal (0 a 1)",
            TipoParam.FLOAT,
            0.5,
            minimo=0.0,
            maximo=1.0,
            passo=0.05,
            ajuda="Só para 'sal e pimenta': 0,5 = metade sal, metade pimenta.",
        ),
        SEMENTE,
    ),
    descricao=(
        "Uma fração p dos pixels, sorteada, vira 255 (sal) ou 0 (pimenta). Em RGB, o "
        "pixel inteiro vira branco ou preto. Sem conversão de escala: os valores são 0 e 255."
    ),
)
def ruido_sal_pimenta(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    rng, semente = _gerador(p["semente"])
    altura, largura = img.shape[:2]

    # Sorteia os pixels corrompidos (já em ordem aleatória) e decide quantos são sal.
    corrompidos = _sortear_indices(rng, altura * largura, p["probabilidade"])
    k = corrompidos.size
    if p["tipo"] == "sal":
        n_sal = k
    elif p["tipo"] == "pimenta":
        n_sal = 0
    else:
        n_sal = round(p["proporcao_sal"] * k)
    # Como a ordem já é aleatória, os primeiros n_sal viram sal e os demais, pimenta.
    linhas_sal, colunas_sal = np.unravel_index(corrompidos[:n_sal], (altura, largura))
    linhas_pim, colunas_pim = np.unravel_index(corrompidos[n_sal:], (altura, largura))

    saida = img.copy()  # a entrada chega somente leitura
    # Em RGB, img[l, c] é o pixel com os 3 canais: o pixel inteiro vira branco ou preto.
    saida[linhas_sal, colunas_sal] = 255
    saida[linhas_pim, colunas_pim] = 0

    total = altura * largura
    info = {
        "pixels com sal (255)": f"{n_sal} ({100 * n_sal / total:.1f}%)",
        "pixels com pimenta (0)": f"{k - n_sal} ({100 * (k - n_sal) / total:.1f}%)",
        "semente usada": semente,
    }
    return Resultado(saidas=[(f"Com ruído {p['tipo']}", saida)], info=info)
