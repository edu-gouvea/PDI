# Imagens de teste

| Arquivo | Tipo | Tamanho | Bom para testar |
|---|---|---|---|
| `cameraman.png` | Cinza 8 bits | 512×512 | Filtros espaciais, limiarização, bordas (Laplaciano, Sobel), frequência |
| `cameraman_baixo_contraste.png` | Cinza 8 bits | 512×512 | Equalização de histograma, log, potência, fatiamento (tons só em [80, 160]) |
| `astronauta.png` | RGB 24 bits | 512×512 | Decomposição RGB/HSV, filtros por canal, ruídos |
| `cafe.png` | RGB 24 bits | 600×400 | Decomposição HSV (cores saturadas), imagem não quadrada |

## Origem e licença

Todas vêm das imagens de exemplo do scikit-image (`skimage.data`):

- `cameraman.png`: `skimage.data.camera()`. CC0, foto de Lav Varshney.
- `cameraman_baixo_contraste.png`: derivada da anterior com `s = 80 + r·80/255`.
- `astronauta.png`: `skimage.data.astronaut()`. Foto da NASA (Eileen Collins), sem restrições
  de copyright conhecidas.
- `cafe.png`: `skimage.data.coffee()`. CC0, foto de Rachel Michetti.
