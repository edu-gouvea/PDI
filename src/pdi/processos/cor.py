"""Decomposição RGB e decomposição HSV (I1)."""

from typing import Any

import numpy as np

from pdi.core.contrato import Resultado
from pdi.core.registro import registrar_processo
from pdi.core.util import clip_para_uint8

NOMES_RGB = ("R (vermelho)", "G (verde)", "B (azul)")


@registrar_processo(
    id="decomposicao_rgb",
    nome="Decomposição RGB",
    grupo="Cor",
    responsavel=1,
    aceita_cinza=False,
    aceita_rgb=True,
    descricao=(
        "Separa a imagem em três imagens em cinza, uma por canal R, G e B. Cada pixel "
        "mostra o quanto daquela cor ele tem: claro = muito, escuro = pouco. "
        "Sem conversão: os valores são os do próprio canal (já em [0, 255])."
    ),
)
def decomposicao_rgb(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    # Uma imagem RGB (H, W, 3) já é a pilha de três matrizes (H, W): basta separá-las.
    # .copy(): cada canal vira um array próprio e contíguo, independente da entrada.
    canais = [img[..., i].copy() for i in range(3)]
    medias = {
        f"média {nome[0]}": round(float(c.mean()), 1)
        for nome, c in zip(NOMES_RGB, canais, strict=True)
    }
    return Resultado(saidas=list(zip(NOMES_RGB, canais, strict=True)), info=medias)


def rgb_para_hsv(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte RGB uint8 em (H em graus [0, 360), S em [0, 1], V em [0, 1]), em float64.

    Com R, G, B normalizados em [0, 1]:

    - V = max(R, G, B)                       → o quão "forte" é a cor mais intensa
    - C = max − min (croma)                  → o quanto a cor difere de um cinza
    - S = C / V  (0 se V = 0)                → a pureza da cor
    - H depende de qual canal é o máximo (o "setor" do círculo de cores):
        máximo R: H = 60° · ((G − B)/C mod 6)    → entre magenta (300°) e amarelo (60°)
        máximo G: H = 60° · ((B − R)/C + 2)      → entre amarelo (60°) e ciano (180°)
        máximo B: H = 60° · ((R − G)/C + 4)      → entre ciano (180°) e magenta (300°)
      Se C = 0 (pixel cinza, R = G = B), a matiz é indefinida; por convenção H = 0.
    """
    rgb = img.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    v = rgb.max(axis=-1)
    c = v - rgb.min(axis=-1)
    s = np.divide(c, v, out=np.zeros_like(v), where=v > 0)  # evita 0/0 no preto

    h = np.zeros_like(v)
    tem_cor = c > 0
    # Empates (ex.: amarelo, R = G máximos) dão o mesmo ângulo nas duas fórmulas,
    # então a prioridade R → G → B não muda o resultado.
    setor_r = tem_cor & (v == r)
    setor_g = tem_cor & (v == g) & ~setor_r
    setor_b = tem_cor & ~setor_r & ~setor_g
    h[setor_r] = ((g - b)[setor_r] / c[setor_r]) % 6
    h[setor_g] = (b - r)[setor_g] / c[setor_g] + 2
    h[setor_b] = (r - g)[setor_b] / c[setor_b] + 4
    return h * 60.0, s, v


NOMES_HSV = ("H (matiz)", "S (saturação)", "V (valor)")


@registrar_processo(
    id="decomposicao_hsv",
    nome="Decomposição HSV",
    grupo="Cor",
    responsavel=1,
    aceita_cinza=False,
    aceita_rgb=True,
    descricao=(
        "Converte para HSV e mostra matiz (H), saturação (S) e valor (V) como três "
        "imagens em cinza. Exibição por escala linear, sem min-max: H·255/360, S·255, V·255."
    ),
)
def decomposicao_hsv(img: np.ndarray, p: dict[str, Any]) -> Resultado:
    h, s, v = rgb_para_hsv(img)
    # Cada canal vai para [0, 255] por uma escala FIXA (não min-max): assim um mesmo
    # tom de cinza significa sempre a mesma matiz/saturação, em qualquer imagem.
    saidas = [
        clip_para_uint8(h * 255.0 / 360.0),
        clip_para_uint8(s * 255.0),
        clip_para_uint8(v * 255.0),
    ]
    sem_cor = float(np.mean(s == 0)) * 100
    info = {
        "pixels sem matiz (cinza, S = 0)": f"{sem_cor:.1f}%",
        "S médio": round(float(s.mean()), 3),
        "V médio": round(float(v.mean()), 3),
    }
    return Resultado(saidas=list(zip(NOMES_HSV, saidas, strict=True)), info=info)
