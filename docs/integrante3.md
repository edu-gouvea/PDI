# Implementação do integrante 3

Os seis processos de realce, frequência e restauração estão implementados em NumPy.
Mantêm os identificadores, os parâmetros e o contrato `(img, p) -> Resultado` da interface.
Todos aceitam cinza e RGB e processam os canais independentemente. A normalização para
exibição em RGB usa os extremos globais, mantendo a escala relativa entre os canais.

## Execução por sprints

A base foi validada antes da implementação: 535 testes passaram, com 5 casos ignorados por
incompatibilidade cinza/RGB, e Ruff aprovado. Cada sprint começou com planejamento,
implementação e validação. Só após validação positiva começou a próxima.

| Sprint | Planejamento e implementação | Validação |
|---|---|---|
| 1 — FFT | Padding refletido, FFT centralizada, inversa, recorte e grade de distâncias. | 116 testes passaram, 5 ignorados; DFT explícita, dimensões pares/ímpares, constantes e preservação de valores negativos. Ruff aprovado. |
| 2 — Realce espacial | Aguçamento com média uniforme; Laplaciano clássico e extensão binomial; Sobel binomial estendido. | 189 testes passaram, 5 ignorados; referências OpenCV, sinais, ganhos, janelas até os limites e entrada somente leitura. Ruff aprovado. |
| 3 — Frequência | Máscaras gaussianas e Butterworth, inversa com recorte e acessórios para inspeção. | A primeira validação identificou o uso indevido de `aplicar_por_canal` com FFT que muda o shape. A sprint foi replanejada para converter explicitamente os canais sem alterar o contrato compartilhado. Após a correção: 163 testes passaram, 5 ignorados; DFT matricial independente, máscaras complementares e testes de constantes. Ruff aprovado. |
| 4 — Restauração | Mediana adaptativa com estágios A/B e processamento dos pixels pendentes em blocos. | 229 testes passaram, 5 ignorados; referência escalar, crescimento, fallback em S_max, constantes, imagens de dois níveis e ruído impulsivo denso. Ruff aprovado. |
| 5 — Integração | Seis processos reais na interface, PNG, acessórios, encadeamento, desempenho e documentação; remoção do auxiliar de stubs sem uso. | Interface e fumaça: 111 aprovados, 5 ignorados. Suíte completa: 692 aprovados, 5 ignorados por compatibilidade. Após corrigir o espaçamento de um exemplo na documentação, Ruff, formatação e `git diff --check` aprovados. |

## Aguçamento

`g = f + k · (f − média_uniforme(f, janela))`.

A média usa pesos `1/janela²` e a convolução do integrante 2, com bordas refletidas sem
repetir o pixel da extremidade. Os cálculos usam float64, sem arredondar a média antes de
formar a máscara. A imagem final usa arredondamento e clip em `[0, 255]`.

`k = 0` preserva a imagem; `k = 1` faz o aguçamento usual; aumentar k amplia a diferença
entre o original e a suavização e pode saturar a saída. Aumentar a janela muda a escala dos
detalhes destacados. Uma imagem constante permanece constante.

## Laplaciano

A janela 3 usa o Laplaciano de quatro vizinhos, com centro negativo:

```text
 0   1   0
 1  -4   1
 0   1   0
```

A imagem principal é `g = clip(round(f − Δf))`. O sinal da subtração é necessário porque
o centro do kernel é negativo. O acessório `("Laplaciano ajustado", imagem_uint8)` usa
normalização min-max da resposta com sinal. Uma resposta constante é exibida em preto.

Para janelas ímpares maiores, convolui-se esse kernel clássico com um kernel binomial
normalizado de lado `janela − 2`, obtendo suporte final `janela × janela`. O vetor binomial
é a linha de ordem `janela − 3` do triângulo de Pascal dividida por `2^(janela − 3)`.
Por exemplo, janela 5 usa a suavização separável `[1, 2, 1]/4` antes do Laplaciano.
A soma do kernel final é zero; ele anula constantes. Janelas maiores calculam o Laplaciano
em uma escala mais suave. Essa extensão atende à janela ajustável exigida pelo professor;
a especificação não determina uma máscara estendida específica.

## Sobel

Calcula `Gx` e `Gy` por correlação e devolve `sqrt(Gx² + Gy²)`, normalizado por min-max.
Usar correlação mantém a orientação da derivada escrita na máscara. Na janela 3:

```text
Gx: -1  0  1       Gy: -1 -2 -1
    -2  0  2            0  0  0
    -1  0  1            1  2  1
```

Nas janelas maiores, a suavização é o vetor binomial de ordem `janela − 1`, e a derivada é
a convolução de `[-1, 0, 1]` com o vetor binomial de ordem `janela − 3`. O produto externo
forma Gx; sua transposta forma Gy. Essa construção coincide com o Sobel estendido do
OpenCV, usado apenas como referência nos testes. Toda a derivação das máscaras é feita
no código com coeficientes binomiais e NumPy.

Janelas maiores incorporam mais suavização e mudam a escala das bordas detectadas. As
bordas da imagem usam reflexão, como no Laplaciano e aguçamento. A saída mostra a magnitude
das bordas, não sua soma com a imagem original. Uma imagem constante produz preto.

## Frequência

O fluxo de cada canal é:

1. Converter para float64 e completar por reflexão, dobrando H e W.
2. Aplicar `np.fft.fft2` e `fftshift`, colocando o componente DC no centro.
3. Multiplicar o espectro pela máscara real H(u,v).
4. Aplicar `ifftshift` e `ifft2`; conservar a parte real com sinal.
5. Recortar a região central para o tamanho original e converter para exibição.

O padding é distribuído nos dois lados: antes entram `H//2`, `W//2` pixels; depois entra o
restante. A reflexão evita a borda preta introduzida por padding com zeros e preserva
imagens constantes no passa-baixa. A FFT continua periódica no domínio expandido; o padding
reduz a influência da junção periódica sobre a região recortada, sem eliminar toda condição
de contorno. `fftshift`/`ifftshift` também funcionam com dimensões ímpares sem padding.

`D(u,v) = hypot(u − H_fft//2, v − W_fft//2)`. D0 mede pixels (bins) **do espectro com padding**;
não está em ciclos/pixel nem na escala da FFT sem padding.

| Filtro | Passa-baixa | Passa-alta |
|---|---|---|
| Gaussiano | `H = exp(−D² / (2·D0²))` | `1 − H` |
| Butterworth | `H = 1 / (1 + (D/D0)^(2n))` | `1 − H` |

No centro, PB = 1 e PA = 0. Em `D = D0`, o gaussiano vale `exp(−1/2)` e Butterworth vale
`1/2`, independentemente da ordem. Diminuir D0 aumenta a suavização do passa-baixa; aumentar
a ordem de Butterworth torna a transição mais abrupta e pode produzir ringing.

A imagem passa-baixa usa arredondamento e clip, preservando o brilho absoluto. A resposta
passa-alta contém valores positivos e negativos e usa min-max global para exibição. Portanto,
sua saída PNG mostra o contraste relativo dessa resposta; não deve ser somada diretamente à
imagem original como se ainda contivesse os valores com sinal. Não se toma o módulo da inversa.

Cada processo devolve três acessórios:

- **Espectro original (log)**: `log(1 + abs(F))`, com min-max.
- **Filtro H(u,v)**: `round(255·H)`, com clip, preservando a escala física de transmissão 0–1.
- **Espectro filtrado (log)**: `log(1 + abs(F·H))`, com min-max.

Os espectros são normalizados separadamente para inspeção; seus brilhos não medem a energia
relativa antes/depois. Resíduos menores que `64·eps_float64·escala` são zerados antes da
exibição. Isso impede que ruído numérico de uma imagem constante vire textura por min-max.
Butterworth trata o overflow da potência para razões enormes como o limite H = 0.

As rotinas reutilizáveis ficam em `pdi.core.frequencia`:

```python
from pdi.core.frequencia import fft_centralizada, inverter_fft, grade_distancias

espectro = fft_centralizada(canal)  # complexo, shape (2H, 2W)
distancias = grade_distancias(espectro.shape)
recuperada = inverter_fft(espectro, canal.shape)  # float64, shape (H, W)
```

O helper `aplicar_por_canal` do projeto exige preservar o shape, então não serve para FFT
com padding ou inversa com recorte. Nessas duas conversões, os filtros empilham explicitamente
os canais RGB. Para as operações espaciais, o helper continua sendo reutilizado.

## Mediana adaptativa

Para cada pixel, começa com janela 3 e calcula mínimo, máximo e mediana:

- **Estágio A**: se `mínimo < mediana < máximo`, passa a B; senão aumenta a janela em 2.
- **Estágio B**: se `mínimo < pixel_original < máximo`, preserva o pixel; senão usa a mediana.
- Se A ainda falhar em `S_max`, usa a mediana dessa última janela.

O filtro lê sempre a imagem original; pixels já resolvidos não influenciam os vizinhos.
A seleção produz níveis uint8 exatos, sem subtrações em uint8 e sem normalização.
O padding refletido e as vizinhanças em blocos são reutilizados do integrante 2.

Só os pixels pendentes entram nas reduções das janelas seguintes. Uma imagem constante
pode ser copiada diretamente. Uma imagem de apenas dois níveis nunca passa A e vai direto
à mediana de S_max: uma otimização exata, conferida com a referência escalar. S_max maior
permite tratar impulsos mais densos, mas custa mais tempo e não recupera detalhes que o
ruído já eliminou.

## Validação e desempenho

Com o ambiente de desenvolvimento instalado:

```sh
python -m pytest -q
ruff check .
ruff format --check .
git diff --check
```

Os novos testes estão em `test_frequencia.py`, `test_realce.py`, `test_frequencia_filtros.py`
e `test_restauracao.py`. As referências incluem DFT calculada explicitamente, OpenCV para
máscaras espaciais e uma mediana adaptativa escalar independente. O teste da interface
executa os seis processos em cinza e RGB pelo worker Qt, renderiza os acessórios, salva
PNG, verifica os pixels reabertos e encadeia a saída. Os 17 processos agora são reais;
o arquivo `_stub.py`, sem consumidores, foi removido.

Na verificação local com RGB aleatório 512×512, sem interface Qt:

| Processo e parâmetros | Tempo aproximado |
|---|---|
| Aguçamento: k = 10, janela 31 | 0,67 s |
| Laplaciano: janela 15 | 0,24 s |
| Sobel: janela 15 | 0,29 s |
| Gaussiano PA: D0 = 1 | 0,55 s |
| Butterworth PB: D0 = 1000, ordem 20 | 0,55 s |
| Mediana adaptativa: S_max = 31 | 0,25 s |
| Mediana adaptativa: dois níveis, S_max = 31 | 6,99 s |

O pico do processo completo de verificação foi 258.016 KiB (cerca de 252 MiB), medido com
`/usr/bin/time`. Não inclui Qt e varia por máquina. A FFT usa arrays complexos e padding
2H×2W, portanto o consumo cresce com o tamanho da imagem; a mediana usa blocos para limitar
cópias das vizinhanças. O tempo da mediana adaptativa depende muito de quantos pixels
exigem janelas maiores. A interface já executa os processos em segundo plano e pausa a
pré-visualização automática após uma execução que ultrapassa 1 segundo.
