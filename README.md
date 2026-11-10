# PDI: Projeto da 3ª unidade (UFPB/CI/DSC)

Software com interface gráfica para aplicar processos de Processamento Digital de Imagens
em PNGs em escala de cinza (8 bits) ou RGB (24 bits).

> Instruções completas de execução (incluindo `run.sh` / `run.bat`) virão na Etapa 5.

## Ambiente de desenvolvimento

Requer Python 3.11 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # já inclui requirements.txt

python -m pdi                      # roda o app
pytest                             # roda os testes
ruff check . && ruff format .      # lint e formatação
```

Contrato de processos (o que cada processo recebe e devolve):
[`src/pdi/core/contrato.py`](src/pdi/core/contrato.py). Para implementar um processo,
troque o corpo da função correspondente em [`src/pdi/processos/`](src/pdi/processos/).

Especificação do professor: [`docs/especificacao-projeto.pdf`](docs/especificacao-projeto.pdf).
