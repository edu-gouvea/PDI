"""Utilitários compartilhados: normalizar_para_uint8, clip e aplicar_por_canal (I1)."""

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
