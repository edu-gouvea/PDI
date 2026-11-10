"""Utilitários compartilhados: normalizar_para_uint8, clip e aplicar_por_canal (I1).

Política de volta para uint8 (seção 7.2 e 8.3, item 4, do CLAUDE.md)
---------------------------------------------------------------------
Todo processo calcula em float64 e precisa devolver uint8. Há duas formas, e cada
processo deve documentar na ``descricao`` qual usa:

- ``clip_para_uint8``: quando o **valor absoluto tem significado** e o resultado
  esperado já está em [0, 255]; o que passar disso é saturado. Ex.: ruído
  (``f + η``), potência e log com ganho ``c`` escolhido pelo usuário, filtros de
  suavização, máscara de aguçamento.
- ``normalizar_para_uint8`` (min-max): quando a saída tem **faixa arbitrária**, com
  negativos ou valores muito grandes, e o que importa é a forma relativa. Ex.:
  Laplaciano ajustado, magnitude do Sobel, espectro ``log(1+|F|)``, filtro ``H(u,v)``.

Atenção: min-max sempre estica o contraste até [0, 255]. Usado no lugar do clip, ele
esconde o efeito do ganho ``c`` (multiplicar tudo por ``c`` e normalizar dá o mesmo
resultado para qualquer ``c``).
"""

from collections.abc import Callable
from typing import Any

import numpy as np

TIPO_CINZA = "cinza"
TIPO_RGB = "rgb"


def tipo_da_imagem(img: np.ndarray) -> str:
    """Devolve ``TIPO_CINZA`` para ``(H, W)`` ou ``TIPO_RGB`` para ``(H, W, 3)``.

    Levanta ``ValueError`` se ``img`` não seguir a convenção do projeto (seção 7.2 do
    CLAUDE.md): ``np.ndarray`` uint8, não vazio, com 2 dimensões ou 3 com 3 canais.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError(f"A imagem deveria ser um np.ndarray, mas é {type(img).__name__}.")
    if img.dtype != np.uint8:
        raise ValueError(f"A imagem deveria ser uint8, mas é {img.dtype}.")
    if img.size == 0:
        raise ValueError("A imagem está vazia.")
    if img.ndim == 2:
        return TIPO_CINZA
    if img.ndim == 3 and img.shape[2] == 3:
        return TIPO_RGB
    raise ValueError(
        f"Formato de imagem não suportado: shape {img.shape}. Use (H, W) ou (H, W, 3)."
    )


def _como_float_finito(arr: np.ndarray) -> np.ndarray:
    """Converte para float64 e recusa NaN/infinito (sinal de bug, ex.: log de negativo)."""
    x = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("O resultado contém valores inválidos (NaN ou infinito).")
    return x


def normalizar_para_uint8(arr: np.ndarray) -> np.ndarray:
    """Reescala linearmente ``[min, max]`` de ``arr`` para ``[0, 255]`` (normalização min-max).

    Fórmula: ``s = round(255 · (x − min) / (max − min))``. O menor valor vira 0 (preto)
    e o maior vira 255 (branco). Se a imagem for constante (``max == min``), devolve
    zeros. Em RGB, min e max são **globais** (dos três canais juntos), para não
    distorcer as cores. Não modifica ``arr``.
    """
    x = _como_float_finito(arr)
    minimo, maximo = x.min(), x.max()
    if maximo == minimo:
        return np.zeros(x.shape, dtype=np.uint8)
    # Multiplica antes de dividir: 50·255/100 = 127.5 exato, enquanto 50·(255/100)
    # dá 127.4999... em ponto flutuante e arredondaria para baixo.
    s = (x - minimo) * 255.0 / (maximo - minimo)
    return np.rint(s).astype(np.uint8)


def clip_para_uint8(arr: np.ndarray) -> np.ndarray:
    """Arredonda ao inteiro mais próximo e satura em ``[0, 255]``.

    Fórmula: ``s = min(max(round(x), 0), 255)``. Valores negativos viram 0 e valores
    acima de 255 viram 255; o restante não muda de escala. Não modifica ``arr``.
    (``np.rint`` arredonda empates para o par: 2.5 → 2, 3.5 → 4.)
    """
    x = _como_float_finito(arr)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def aplicar_por_canal(
    func: Callable[..., np.ndarray], img: np.ndarray, *args: Any, **kwargs: Any
) -> np.ndarray:
    """Aplica ``func`` (que trabalha em 2D) à imagem cinza ou a cada canal R, G, B.

    - Cinza ``(H, W)``: devolve ``func(img, *args, **kwargs)``.
    - RGB ``(H, W, 3)``: aplica a cada canal **independentemente** e empilha na
      ordem R, G, B.

    ``func`` deve devolver um array 2D com o mesmo ``(H, W)`` da entrada. O dtype da
    saída é o que ``func`` devolver; a conversão para uint8 fica a cargo de quem chama.

    Exemplo::

        def _suavizar(canal, janela): ...
        saida = aplicar_por_canal(_suavizar, img, janela=p["janela"])
    """
    if img.ndim == 2:
        return _checar_saida_canal(func(img, *args, **kwargs), img.shape, func)
    if img.ndim == 3:
        canais = [
            _checar_saida_canal(func(img[..., c], *args, **kwargs), img.shape[:2], func)
            for c in range(img.shape[2])
        ]
        return np.stack(canais, axis=-1)
    raise ValueError(f"aplicar_por_canal espera (H, W) ou (H, W, C), recebeu shape {img.shape}.")


def _checar_saida_canal(saida: Any, shape: tuple[int, ...], func: Callable) -> np.ndarray:
    nome = getattr(func, "__name__", repr(func))
    if not isinstance(saida, np.ndarray) or saida.shape != tuple(shape):
        recebido = getattr(saida, "shape", type(saida).__name__)
        raise ValueError(f"{nome} deveria devolver um array {tuple(shape)}, devolveu {recebido}.")
    return saida
