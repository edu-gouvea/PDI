"""Filtros na FFT centralizada com padding refletido e recorte espacial (I3)."""

from typing import Any

import numpy as np

from pdi.core.contrato import Param, Resultado, TipoParam
from pdi.core.frequencia import fft_centralizada, grade_distancias, inverter_fft
from pdi.core.registro import registrar_processo
from pdi.core.util import clip_para_uint8, normalizar_para_uint8


def _mascara_gaussiana(distancias: np.ndarray, d0: float) -> np.ndarray:
    return np.exp(-0.5 * (distancias / d0) ** 2)


def _mascara_butterworth(distancias: np.ndarray, d0: float, ordem: int) -> np.ndarray:
    # Para razões muito grandes, o limite é H=0; o overflow de potência é
    # intencionalmente tratado como infinito no denominador, sem produzir NaN.
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + (distancias / d0) ** (2 * ordem))


def _filtrar(img: np.ndarray, p: dict[str, Any], *, butterworth: bool) -> Resultado:
    # O padding altera o shape: aplicar_por_canal é reservado às funções que
    # mantêm H×W. Aqui, as conversões por canal são explícitas.
    espectro = (
        fft_centralizada(img)
        if img.ndim == 2
        else np.stack([fft_centralizada(img[..., c]) for c in range(3)], axis=-1)
    )
    distancias = grade_distancias(espectro.shape[:2])
    baixa = (
        _mascara_butterworth(distancias, p["d0"], p["ordem"])
        if butterworth
        else _mascara_gaussiana(distancias, p["d0"])
    )
    alta = p["tipo"] == "passa-alta"
    mascara = 1.0 - baixa if alta else baixa

    # Evita transformar resíduos da FFT de uma constante em textura visível ao
    # normalizar passa-alta/espectros. O limiar é relativo à precisão de float64.
    tolerancia = 64 * np.finfo(np.float64).eps * max(1.0, float(np.abs(espectro).max()))
    espectro[np.abs(espectro) < tolerancia] = 0
    filtrado = espectro * (mascara[..., np.newaxis] if img.ndim == 3 else mascara)
    espacial = (
        inverter_fft(filtrado, img.shape[:2])
        if img.ndim == 2
        else np.stack([inverter_fft(filtrado[..., c], img.shape[:2]) for c in range(3)], axis=-1)
    )
    tolerancia_espacial = 64 * np.finfo(np.float64).eps * max(1.0, float(img.max()))
    espacial[np.abs(espacial) < tolerancia_espacial] = 0
    saida = normalizar_para_uint8(espacial) if alta else clip_para_uint8(espacial)
    info = {"tipo": p["tipo"], "D0": p["d0"], "FFT": f"{mascara.shape[1]}×{mascara.shape[0]}"}
    if butterworth:
        info["ordem n"] = p["ordem"]
    return Resultado(
        saidas=[
            (f"{p['tipo'].capitalize()} {'Butterworth' if butterworth else 'gaussiano'}", saida)
        ],
        acessorios=[
            ("Espectro original (log)", normalizar_para_uint8(np.log1p(np.abs(espectro)))),
            ("Filtro H(u,v)", clip_para_uint8(255.0 * mascara)),
            ("Espectro filtrado (log)", normalizar_para_uint8(np.log1p(np.abs(filtrado)))),
        ],
        info=info,
    )


TIPO = Param(
    "tipo",
    "Tipo",
    TipoParam.ESCOLHA,
    "passa-baixa",
    opcoes=("passa-baixa", "passa-alta"),
    ajuda="Passa-baixa suaviza (mantém o centro do espectro); passa-alta realça bordas.",
)

D0 = Param(
    "d0",
    "Frequência de corte D0",
    TipoParam.FLOAT,
    30.0,
    minimo=1.0,
    maximo=1000.0,
    passo=1.0,
    ajuda=("Raio em pixels da FFT com padding (2H×2W). Passa-baixa: menor D0 = mais desfoque."),
)


@registrar_processo(
    id="filtro_gaussiano_frequencia",
    nome="Passa-alta e passa-baixa gaussiano",
    grupo="Frequência",
    responsavel=3,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(TIPO, D0),
    descricao=(
        "H = exp(−D²/2D0²) (PB) ou 1 − H (PA), por canal. FFT com padding refletido 2H×2W "
        "e recorte. PB: arredondamento/clip; PA com sinal: min-max global. Mostra H e espectros."
    ),
)
def filtro_gaussiano_frequencia(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return _filtrar(img, p, butterworth=False)


@registrar_processo(
    id="filtro_butterworth",
    nome="Passa-alta e passa-baixa Butterworth",
    grupo="Frequência",
    responsavel=3,
    aceita_cinza=True,
    aceita_rgb=True,
    parametros=(
        TIPO,
        D0,
        Param(
            "ordem",
            "Ordem n",
            TipoParam.INT,
            2,
            minimo=1,
            maximo=20,
            passo=1,
            ajuda="Maior n = transição mais abrupta, próxima do filtro ideal (aparece ringing).",
        ),
    ),
    descricao=(
        "H = 1/(1 + (D/D0)^(2n)) (PB) ou 1 − H (PA), por canal. Padding refletido 2H×2W "
        "e recorte. PB: arredondamento/clip; PA com sinal: min-max global. Mostra H e espectros."
    ),
)
def filtro_butterworth(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    return _filtrar(img, p, butterworth=True)
