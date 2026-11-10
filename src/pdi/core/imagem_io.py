"""Carregar e salvar PNG e detectar o tipo da imagem: cinza (H, W) ou RGB (H, W, 3) (I1).

O Pillow é usado SOMENTE para ler e escrever o arquivo. Toda conversão (alfa,
16 bits, 1 bit) é feita aqui com NumPy, para sabermos exatamente o que acontece.

Regras de carregamento (o PNG pode vir em vários "modos"):

==========  ==========================================  ==========================
Modo        O que é                                     Resultado
==========  ==========================================  ==========================
L           cinza 8 bits                                cinza
RGB         colorida 24 bits                            RGB
RGBA, LA    com transparência (alfa)                    composta sobre branco + aviso
P           paleta de cores                             RGB (com alfa: sobre branco)
1           preto e branco, 1 bit                       cinza 0/255 + aviso
I;16, I     cinza 16 bits                               cinza 8 bits + aviso
==========  ==========================================  ==========================

Depois disso, uma imagem RGB com os três canais idênticos é tratada como cinza
(com aviso), porque na prática ela é cinza; assim a equalização fica disponível.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from pdi.core.util import TIPO_CINZA, clip_para_uint8, tipo_da_imagem

# Todo PNG começa com estes 8 bytes. Conferir o conteúdo (e não a extensão) recusa,
# por exemplo, um JPEG renomeado para .png.
_ASSINATURA_PNG = b"\x89PNG\r\n\x1a\n"
# Posição da profundidade de bits no cabeçalho: 8 (assinatura) + 4 (tamanho do bloco)
# + 4 ("IHDR") + 4 (largura) + 4 (altura) = byte 24.
_POS_PROFUNDIDADE = 24

AVISO_ALFA = "Transparência descartada: a imagem foi composta sobre fundo branco."
AVISO_16_BITS = "Imagem de 16 bits convertida para 8 bits."
AVISO_1_BIT = "Imagem de 1 bit (preto e branco) convertida para 8 bits."
AVISO_RGB_CINZA = "Imagem RGB com os três canais iguais: tratada como escala de cinza."


@dataclass
class ImagemCarregada:
    pixels: np.ndarray  # uint8, (H, W) ou (H, W, 3)
    tipo: str  # TIPO_CINZA ou TIPO_RGB
    caminho: Path
    avisos: list[str] = field(default_factory=list)  # para a barra de status

    @property
    def resumo(self) -> str:
        """Texto curto para a barra de status, ex.: 'RGB 24 bits · 512×384'."""
        altura, largura = self.pixels.shape[:2]
        nome = "Cinza 8 bits" if self.tipo == TIPO_CINZA else "RGB 24 bits"
        return f"{nome} · {largura}×{altura}"


def carregar_png(caminho: str | Path, rgb_igual_vira_cinza: bool = True) -> ImagemCarregada:
    """Lê um PNG e devolve a imagem no formato do projeto (uint8, cinza ou RGB).

    Levanta ``ValueError`` com mensagem amigável se o arquivo não existir, não for PNG,
    estiver corrompido ou usar um modo de cor não suportado.
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ValueError(f"Arquivo não encontrado: {caminho}")

    with caminho.open("rb") as f:
        cabecalho = f.read(_POS_PROFUNDIDADE + 1)
    if not cabecalho.startswith(_ASSINATURA_PNG):
        raise ValueError(f"'{caminho.name}' não é um arquivo PNG. O programa só abre imagens .png.")
    profundidade = cabecalho[_POS_PROFUNDIDADE] if len(cabecalho) > _POS_PROFUNDIDADE else 8

    avisos: list[str] = []
    try:
        with Image.open(caminho) as im:
            im.load()
            modo = im.mode
            if modo == "P":
                # Paleta: cada pixel é um índice numa tabela de cores. Convertê-la para
                # RGB(A) é sem perda; a transparência pode estar na própria paleta.
                modo = "RGBA" if "transparency" in im.info else "RGB"
                im = im.convert(modo)
            arr = np.array(im)
    except (OSError, SyntaxError, Image.DecompressionBombError) as e:
        raise ValueError(f"Não foi possível ler '{caminho.name}': arquivo corrompido.") from e

    if modo in ("L", "RGB"):
        pixels = arr
    elif modo in ("RGBA", "LA"):
        # Último canal = alfa; os anteriores = cor (3 para RGBA, 1 para LA).
        cor = arr[..., :-1] if modo == "RGBA" else arr[..., 0]
        pixels = _compor_sobre_branco(cor, arr[..., -1])
        avisos.append(AVISO_ALFA)
    elif modo == "1":
        pixels = arr.astype(np.uint8) * 255  # False → 0, True → 255
        avisos.append(AVISO_1_BIT)
    elif modo in ("I;16", "I;16B", "I;16L", "I"):
        pixels = _de_16_para_8_bits(arr)
    else:
        raise ValueError(f"Modo de cor '{modo}' não suportado em '{caminho.name}'.")

    if profundidade == 16:
        # Inclui o RGB de 16 bits, que o Pillow já entrega em 8 bits sem avisar.
        avisos.append(AVISO_16_BITS)

    if rgb_igual_vira_cinza and pixels.ndim == 3 and _canais_iguais(pixels):
        pixels = pixels[..., 0]
        avisos.append(AVISO_RGB_CINZA)

    if pixels.dtype != np.uint8:
        raise ValueError(f"Formato de pixel não suportado em '{caminho.name}' ({pixels.dtype}).")
    pixels = np.ascontiguousarray(pixels)  # cópia própria, contígua na memória
    return ImagemCarregada(pixels, tipo_da_imagem(pixels), caminho, avisos)


def salvar_png(img: np.ndarray, caminho: str | Path) -> Path:
    """Salva ``img`` (uint8, cinza ou RGB) como PNG sem perda e devolve o caminho usado.

    Se o nome não terminar em ``.png``, a extensão é acrescentada. Levanta
    ``ValueError`` se a imagem estiver fora do formato do projeto ou se não for
    possível gravar (pasta inexistente, sem permissão etc.).
    """
    tipo_da_imagem(img)  # recusa dtype/shape inválidos com mensagem clara
    caminho = Path(caminho)
    if caminho.suffix.lower() != ".png":
        caminho = caminho.with_name(caminho.name + ".png")
    try:
        # uint8 (H, W) vira modo L; uint8 (H, W, 3) vira modo RGB.
        Image.fromarray(np.ascontiguousarray(img)).save(caminho, format="PNG")
    except OSError as e:
        motivo = e.strerror or str(e)
        raise ValueError(f"Não foi possível salvar em '{caminho}': {motivo}.") from e
    return caminho


# --------------------------------------------------------------------- auxiliares


def _compor_sobre_branco(cor: np.ndarray, alfa: np.ndarray) -> np.ndarray:
    """Composição alfa sobre fundo branco: s = a·cor + (1 − a)·255, com a = alfa/255."""
    a = alfa.astype(np.float64) / 255.0
    if cor.ndim == 3:
        a = a[..., np.newaxis]  # mesmo alfa para R, G e B
    return clip_para_uint8(a * cor + (1.0 - a) * 255.0)


def _de_16_para_8_bits(arr: np.ndarray) -> np.ndarray:
    """Reescala [0, 65535] para [0, 255]: s = round(x / 257), pois 65535 / 257 = 255."""
    return clip_para_uint8(arr.astype(np.float64) / 257.0)


def _canais_iguais(rgb: np.ndarray) -> bool:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return bool(np.array_equal(r, g) and np.array_equal(g, b))
