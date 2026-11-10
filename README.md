# PDI: Processamento Digital de Imagens

Projeto da 3ª unidade de Introdução ao Processamento Digital de Imagens (UFPB/CI/DSC).

Programa com interface gráfica para aplicar 17 processos de PDI a imagens PNG em escala de
cinza (8 bits) ou coloridas RGB (24 bits). Mostra a imagem original e o resultado lado a lado,
com os dados e as imagens acessórias de cada processo (histogramas, Laplaciano ajustado etc.),
e salva o resultado em um novo PNG.

Toda a matemática dos processos é implementada à mão com NumPy. OpenCV e scikit-image são
usados apenas nos testes, como referência para comparar resultados.

---

## Como executar

Requisito: **Python 3.11 ou mais novo** ([python.org/downloads](https://www.python.org/downloads/)).
Na primeira execução, os scripts baixam as dependências (cerca de 300 MB de download e
850 MB depois de instaladas, quase tudo do Qt), então é preciso internet. Nas seguintes,
o programa abre direto.

### Linux / macOS

```bash
./run.sh
```

### Windows

Dê **duplo clique em `run.bat`**, ou rode no terminal:

```bat
run.bat
```

No instalador do Python para Windows, marque **"Add python.exe to PATH"**.

### Abrir já com uma imagem

```bash
./run.sh imagens_teste/cameraman.png        # Windows: run.bat imagens_teste\cameraman.png
```

### Instalação manual (sem os scripts)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pdi
```

---

## Como usar

1. **Abra uma imagem PNG**: *Arquivo → Abrir imagem* (Ctrl+O). A barra de status mostra o
   tipo (cinza ou RGB), as dimensões e avisos de conversão (transparência, 16 bits etc.).
2. **Escolha um processo** na lista à esquerda. Processos que não se aplicam à imagem aberta
   aparecem acinzentados; passe o mouse para ver o motivo (ex.: a equalização de histograma
   só vale para imagens em cinza).
3. **Ajuste os parâmetros**. Cada valor pode ser digitado com precisão na caixa numérica
   (vírgula ou ponto) ou ajustado pelo slider. Passe o mouse sobre um parâmetro para ver o
   efeito dele e a faixa permitida. Janelas de filtro aceitam apenas valores ímpares.
4. Clique em **Aplicar** (Ctrl+Enter ou F5). O processamento roda em segundo plano, sem
   travar a janela, e o tempo gasto aparece no canto inferior direito.
5. Veja o resultado ao lado da original. Processos com várias saídas (decomposição RGB/HSV)
   mostram uma aba por saída. Dados e imagens acessórias aparecem embaixo.
6. **Salve** com *Arquivo → Salvar saída* (Ctrl+S). Com várias saídas, é salva a aba
   selecionada. O nome sugerido descreve o processo e os parâmetros, por exemplo
   `cameraman_filtro-ordem_mediana_janela5.png`.

**Pré-visualização automática**: marque a caixa no rodapé dos parâmetros para reaplicar sozinho
a cada mudança. Ela pausa para processos que levam mais de 1 s; nesse caso, use Aplicar.

**Encadear processos**: *Usar saída como entrada* transforma o resultado na nova entrada.
Exemplo: aplicar ruído sal e pimenta e, em seguida, um filtro de mediana sobre a imagem ruidosa.

**Ruídos reproduzíveis**: com semente 0, cada aplicação sorteia um ruído novo e mostra nos dados
qual semente foi usada. Digitando essa semente, o mesmo ruído é gerado de novo.

### Atalhos

| Ação | Atalho |
|---|---|
| Abrir imagem | Ctrl+O |
| Salvar saída | Ctrl+S |
| Aplicar | Ctrl+Enter ou F5 |
| Sair | Ctrl+Q |

---

## Processos

| Grupo | Processo | Parâmetros | Imagens |
|---|---|---|---|
| Cor | Decomposição RGB | (nenhum) | RGB |
| Cor | Decomposição HSV | (nenhum) | RGB |
| Intensidade | Limiarização | limiar k | cinza |
| Intensidade | Transformação logarítmica | ganho c | cinza e RGB |
| Intensidade | Transformação de potência (gama) | ganho c, gama γ | cinza e RGB |
| Intensidade | Equalização de histograma | (nenhum; mostra histogramas antes e depois) | cinza |
| Intensidade | Fatiamento por intensidade | faixa [A, B], fundo preservado ou zerado | cinza |
| Espacial | Filtro de média gaussiana | σ, janela | cinza e RGB |
| Espacial | Filtros de mediana, mínimo e máximo | tipo, janela | cinza e RGB |
| Espacial | Máscara de aguçamento | ganho k, janela | cinza e RGB |
| Espacial | Realce por Laplaciano | janela (mostra o Laplaciano ajustado) | cinza e RGB |
| Espacial | Gradiente de Sobel | janela | cinza e RGB |
| Frequência | Passa-alta e passa-baixa gaussiano | tipo, frequência de corte D0 | cinza e RGB |
| Frequência | Passa-alta e passa-baixa Butterworth | tipo, D0, ordem n | cinza e RGB |
| Restauração | Filtro adaptativo de mediana | tamanho máximo da janela | cinza e RGB |
| Ruído | Ruído aditivo gaussiano | μ, σ, fração de pixels, semente | cinza e RGB |
| Ruído | Ruído sal, pimenta e sal e pimenta | tipo, probabilidade p, proporção de sal, semente | cinza e RGB |

Em imagens RGB, os filtros são aplicados a cada canal separadamente.

### Imagens de teste

A pasta [`imagens_teste/`](imagens_teste/) traz imagens em cinza e RGB para experimentar,
com origem e licença descritas no [README dela](imagens_teste/README.md).

---

## Problemas comuns

| Sintoma | Solução |
|---|---|
| `Python 3.11 ou mais novo não encontrado` | Instale o Python 3.11+ ([python.org](https://www.python.org/downloads/)); no Windows, marque "Add python.exe to PATH". |
| `Não foi possível criar o ambiente virtual` (Ubuntu/Debian) | `sudo apt install python3-venv` e rode `./run.sh` de novo. |
| Linux com X11: `could not load the Qt platform plugin "xcb"` | `sudo apt install libxcb-cursor0` |
| `./run.sh: Permission denied` | `chmod +x run.sh` (ou rode `bash run.sh`). |
| Dependências corrompidas ou instalação interrompida | Apague a pasta `.venv` e rode o script de novo. |

No Ubuntu 26.04 (GNOME 50), o tema GTK do Qt derruba o programa ao abrir. O `python -m pdi`
já contorna isso usando o tema `gnome` embutido no Qt; para escolher outro tema, defina
`QT_QPA_PLATFORMTHEME` antes de rodar.

---

## Desenvolvimento

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # inclui requirements.txt, pytest, ruff, OpenCV e scikit-image

pytest                             # todos os testes (os de interface rodam sem abrir janela)
ruff check . && ruff format .      # lint e formatação, antes de cada PR
```

### Estrutura

```
src/pdi/
├── core/          contrato de processos, registro, leitura/escrita de PNG, utilitários
├── processos/     os 17 processos, um módulo por grupo
└── ui/            interface em PySide6 (Qt)
tests/             testes em pytest
imagens_teste/     imagens de exemplo
scripts/           apoio ao run.sh / run.bat
```

### Como implementar ou adicionar um processo

O contrato de processos (o que cada processo recebe e devolve) está em
[`src/pdi/core/contrato.py`](src/pdi/core/contrato.py). Cada processo é uma função
`f(img, p) -> Resultado` registrada com `@registrar_processo`; a interface descobre os processos
pelo registro, sem nenhuma mudança na UI. Para implementar um processo existente, troque o corpo
da função correspondente em [`src/pdi/processos/`](src/pdi/processos/).

Especificação do professor: [`docs/especificacao-projeto.pdf`](docs/especificacao-projeto.pdf).
