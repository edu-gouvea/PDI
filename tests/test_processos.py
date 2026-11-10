"""Testes de fumaça de TODOS os processos registrados (stubs ou reais).

Garante que os 17 processos da especificação existem, com a compatibilidade cinza/RGB
combinada pelo grupo, e que cada um roda com os parâmetros padrão numa imagem cinza e
numa RGB. Continua valendo depois que os stubs forem substituídos pelo código real.
"""

import pytest

from pdi.core.registro import executar_processo, listar_processos, processos_por_grupo
from pdi.core.util import TIPO_CINZA, TIPO_RGB, tipo_da_imagem

# id -> (responsável, aceita_cinza, aceita_rgb), na ordem do PDF.
ESPERADOS = {
    "decomposicao_rgb": (1, False, True),
    "decomposicao_hsv": (1, False, True),
    "limiarizacao": (2, True, False),
    "transformacao_log": (2, True, True),
    "transformacao_potencia": (2, True, True),
    "equalizacao_histograma": (2, True, False),
    "fatiamento_intensidade": (2, True, False),
    "filtro_gaussiano": (2, True, True),
    "filtro_ordem": (2, True, True),
    "mascara_agucamento": (3, True, True),
    "realce_laplaciano": (3, True, True),
    "gradiente_sobel": (3, True, True),
    "filtro_gaussiano_frequencia": (3, True, True),
    "filtro_butterworth": (3, True, True),
    "mediana_adaptativa": (3, True, True),
    "ruido_gaussiano": (1, True, True),
    "ruido_sal_pimenta": (1, True, True),
}


def test_os_17_processos_estao_registrados_na_ordem_do_pdf():
    assert [s.id for s in listar_processos()] == list(ESPERADOS)


@pytest.mark.parametrize("spec", listar_processos(), ids=lambda s: s.id)
def test_responsavel_e_compatibilidade(spec):
    assert (spec.responsavel, spec.aceita_cinza, spec.aceita_rgb) == ESPERADOS[spec.id]


def test_equalizacao_desabilitada_para_rgb():
    # Exemplo explícito do professor no PDF.
    spec = next(s for s in listar_processos() if s.id == "equalizacao_histograma")
    assert not spec.aceita_rgb


@pytest.mark.parametrize("spec", listar_processos(), ids=lambda s: s.id)
def test_todo_processo_tem_descricao_e_ajuda(spec):
    assert spec.descricao.strip()
    for p in spec.parametros:
        assert p.ajuda.strip(), f"{spec.id}.{p.nome} sem tooltip"


@pytest.mark.parametrize("spec", listar_processos(), ids=lambda s: s.id)
def test_filtros_espaciais_tem_janela(spec):
    # Nota do professor: em TODOS os filtros no espaço, a janela deve ser ajustável.
    if spec.grupo in ("Espacial", "Restauração"):
        assert any(p.somente_impar for p in spec.parametros), f"{spec.id} sem janela"


def test_grupos_na_ordem_da_interface():
    assert list(processos_por_grupo()) == [
        "Cor",
        "Intensidade",
        "Espacial",
        "Frequência",
        "Restauração",
        "Ruído",
    ]


@pytest.mark.parametrize("tipo", [TIPO_CINZA, TIPO_RGB])
@pytest.mark.parametrize("spec", listar_processos(), ids=lambda s: s.id)
def test_processo_roda_com_parametros_padrao(spec, tipo, img_cinza, img_rgb):
    img = img_cinza if tipo == TIPO_CINZA else img_rgb
    if (tipo == TIPO_CINZA and not spec.aceita_cinza) or (tipo == TIPO_RGB and not spec.aceita_rgb):
        pytest.skip(f"{spec.id} não se aplica a {tipo}")

    res = executar_processo(spec, img)  # já valida o Resultado contra o contrato

    principal = res.saidas[0][1]
    assert principal.shape[:2] == img.shape[:2]
    tipo_da_imagem(principal)
