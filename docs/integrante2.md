# Implementação do integrante 2

Os sete processos de intensidade e suavização estão implementados em NumPy. O contrato
`(img, p) -> Resultado`, os identificadores, os parâmetros e a compatibilidade com cinza/RGB
foram mantidos. O componente `CartaoHistograma` do integrante 1 é reutilizado pelo painel
existente, recebendo os acessórios `Histograma("Antes", h1)` e `Histograma("Depois", h2)`.

## Sprints

Cada sprint seguiu planejamento, implementação e validação. A próxima começou somente
depois da validação positiva da anterior.

| Sprint | Planejamento e implementação | Critério de validação |
|---|---|---|
| 1 — Base compartilhada | Histograma de 256 níveis; convolução, correlação e vizinhanças em blocos. | Contagens exatas, kernels assimétricos, bordas, imagens pequenas, comparação com OpenCV e suíte existente: 411 testes passaram, 5 ignorados por compatibilidade; Ruff aprovado. |
| 2 — Intensidade | Limiarização, log, potência, equalização e fatiamento; ganho preservado por clip; erro para A > B. | Fórmulas escalares, extremos, RGB por canal, CDF conhecida e scikit-image: 165 testes passaram, 5 ignorados; Ruff aprovado. |
| 3 — Suavização | Gaussiano separável; mediana, mínimo e máximo em blocos. | OpenCV e referência escalar, σ e janela ajustáveis, janela 31, constantes, canais independentes e entrada somente leitura: 242 testes passaram, 5 ignorados; Ruff aprovado. |
| 4 — Integração | Testes da interface com processos reais, histogramas, PNG, encadeamento e recuperação de erro; documentação para o grupo. | Suíte completa: 535 testes passaram, 5 ignorados por compatibilidade; Ruff, formatação e `git diff --check` aprovados. |

## Fórmulas e parâmetros

| Processo | Implementação | Efeito dos parâmetros |
|---|---|---|
| Limiarização | `s = 255` se `r > k`; caso contrário, `0`. | Aumentar k reduz a região branca. Um pixel igual a k fica preto. |
| Log | `s = clip(round(c · ln(1 + r)))`. | Expande os tons escuros; aumentar c clareia até saturar. O padrão 45,99 leva 255 a 255 após arredondar. |
| Potência | `s = clip(round(255 · c · (r/255)^γ))`. | γ < 1 clareia; γ > 1 escurece; c é ganho. c = γ = 1 dá identidade; c = 0 dá preto. |
| Equalização | `s_k = round(255 · CDF(k))`, com `CDF(k) = Σ_{j≤k} h[j]/N`. | Não tem parâmetros. Exibe as contagens antes e depois em 256 níveis. |
| Fatiamento | Pixels em `[A, B]`, incluindo os limites, viram 255. | Fora da faixa, preserva a entrada ou zera conforme `fundo`. A > B gera `ValueError` em português. |
| Gaussiano espacial | Pesos proporcionais a `exp(-(x²+y²)/(2σ²))`, normalizados para somar 1. | Maior σ espalha os pesos. A janela ímpar limita o suporte; uma janela pequena trunca o gaussiano mesmo com σ grande. |
| Mediana/mínimo/máximo | Seleciona a estatística da vizinhança ímpar. | Mediana reduz impulsos; mínimo expande regiões escuras e remove sal; máximo expande regiões claras e remove pimenta. Maior janela amplia o efeito. |

Log, potência e gaussiano usam float64 e `clip_para_uint8`, com arredondamento ao par nos
empates. Não usam normalização min-max: ela cancelaria o ganho c e mudaria o brilho absoluto
da suavização. Limiarização e fatiamento selecionam níveis uint8 diretamente. Os filtros de
ordem também selecionam níveis existentes; como a janela é ímpar, sua mediana é inteira.

A equalização usa a CDF discreta declarada originalmente no processo, sem subtrair a menor
CDF não nula. Essa escolha coincide com a fórmula do Gonzalez e com `skimage.exposure.equalize_hist`
(até o arredondamento). Difere da variante `cv2.equalizeHist`, que desconta `CDF_min`.
Consequência: uma imagem constante, inclusive preta, vira branca. Esse caso é intencional e
está coberto por testes; não se deve comparar essa implementação diretamente com a variante
do OpenCV esperando igualdade.

## Bordas, RGB e memória

Os filtros espaciais usam `np.pad(..., mode="reflect")`: refletem sem repetir o pixel da
extremidade, equivalente a `cv2.BORDER_REFLECT_101`. Por exemplo, `[a, b, c, d]`, com margem 2,
vira `[c, b | a, b, c, d | c, b]`. Preservam as dimensões da entrada, inclusive quando a
janela é maior que a imagem ou quando há apenas uma linha/coluna.

Em RGB, log, potência e suavização operam independentemente em R, G e B. Limiarização,
equalização e fatiamento continuam restritos a cinza pelo registro. Nenhuma função modifica
a entrada, que pode ser somente leitura ou uma view não contígua.

O gaussiano aplica o vetor de pesos na horizontal e na vertical, sem arredondar entre as
passagens. Isso equivale ao kernel 2D e reduz o trabalho de aproximadamente `H·W·janela²`
para `2·H·W·janela` por canal.

As vizinhanças são views e as reduções são feitas por faixas de linhas de cerca de 1.048.576
elementos, ou uma linha se ela exceder esse limite. Isso limita as cópias temporárias da
mediana, evitando materializar todas as vizinhanças de uma imagem 512×512 com janela 31.
A memória total ainda depende do tamanho da imagem e das cópias de entrada/saída do aplicativo.

Na verificação local com RGB aleatório 512×512 e janela 31, os quatro filtros completaram:
gaussiano com σ = 20 em cerca de 0,05 s, mediana em 10,14 s, mínimo em 1,14 s e máximo em
1,05 s. O pico do processo de verificação, sem a interface Qt, foi de 60.860 KiB (cerca de
59 MiB), medido com `/usr/bin/time`. Esses números variam por máquina; mostram que a janela
máxima cabe em memória, mas a mediana ainda pode exigir espera. A interface já executa os
processos em segundo plano e pausa a pré-visualização automática quando passam de 1 s.

## API compartilhada com o integrante 3

```python
from pdi.core.espacial import convoluir, correlacionar, iterar_vizinhancas
from pdi.core.histograma import calcular_histograma
from pdi.core.util import aplicar_por_canal, clip_para_uint8

# kernel: matriz 2D com lados ímpares positivos, inclusive retangular.
# O tamanho do kernel determina a janela; kernels 1×n e n×1 são aceitos.
resposta_float = aplicar_por_canal(convoluir, img, kernel, borda="reflect")
saida_uint8 = clip_para_uint8(resposta_float)

# Para Sobel, se o sinal deve seguir a orientação da máscara escrita:
resposta_float = aplicar_por_canal(correlacionar, img, kernel)

# Histograma de cinza ou de um canal RGB selecionado explicitamente:
contagens = calcular_histograma(img_cinza)  # shape (256,), soma = número de pixels
```

`convoluir` inverte o kernel nos dois eixos; `correlacionar` não o inverte. Ambas retornam
float64 sem clip nem normalização, preservando negativos para Laplaciano e Sobel. Para
kernels simétricos, os resultados coincidem. Aceitam `reflect`, `symmetric`, `edge` e
`constant` (zeros) como política de borda. A conversão final é responsabilidade do chamador.

`iterar_vizinhancas(canal, janela)` produz pares `(fatia_de_linhas, view)`; a view tem shape
`(linhas_do_bloco, largura_da_imagem, altura_da_janela, largura_da_janela)` e é somente leitura.
Permite ao integrante 3 implementar outros filtros de vizinhança sem duplicar o padding.

## Como validar

Com as dependências de `requirements-dev.txt` instaladas:

```sh
python -m pytest -q
ruff check .
ruff format --check .
git diff --check
```

Os testes numéricos ficam em `test_histograma.py`, `test_espacial.py`, `test_intensidade.py`
e `test_suavizacao.py`. OpenCV e scikit-image aparecem apenas nos testes. Os testes de
`test_ui_execucao.py` verificam os processos reais passando pelo worker e pelo painel Qt
em modo offscreen, incluindo salvar PNG e corrigir uma faixa A > B.

Os processos do integrante 3 também estão implementados, conforme [integrante3.md](integrante3.md).
A suíte de fumaça cobre agora os 17 processos reais; os testes específicos de cada módulo
verificam as respectivas fórmulas.
