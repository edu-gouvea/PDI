"""Testes do registro de processos e da execução validada."""

import numpy as np
import pytest

from pdi.core import registro
from pdi.core.contrato import Histograma, Param, Resultado, TipoParam
from pdi.core.registro import (
    ErroDeImplementacao,
    executar_processo,
    listar_processos,
    motivo_desabilitado,
    obter_processo,
    processos_por_grupo,
    registrar_processo,
    validar_parametros,
)
from pdi.core.util import TIPO_CINZA, TIPO_RGB

JANELA = Param("janela", "Janela", TipoParam.INT, 3, 1, 31, 2, somente_impar=True)
SIGMA = Param("sigma", "Desvio padrão σ", TipoParam.FLOAT, 1.0, 0.1, 20.0, 0.1)
MODO = Param("modo", "Modo", TipoParam.ESCOLHA, "passa-baixa", opcoes=("passa-baixa", "passa-alta"))
PRESERVAR = Param("preservar", "Preservar fundo", TipoParam.BOOL, True)


def _identidade(img, p):
    return Resultado(saidas=[("Saída", img.copy())], info=dict(p))


def _registrar(id="teste", grupo="Espacial", cinza=True, rgb=True, parametros=(), funcao=None):
    """Atalho para registrar um processo de teste e devolver o spec."""
    registrar_processo(
        id=id,
        nome=id.title(),
        grupo=grupo,
        responsavel=1,
        aceita_cinza=cinza,
        aceita_rgb=rgb,
        parametros=parametros,
    )(funcao or _identidade)
    return obter_processo(id)


# ------------------------------------------------------------------------- registro


def test_decorador_registra_e_devolve_a_funcao(registro_limpo):
    @registrar_processo(
        id="meu_proc", nome="Meu", grupo="Cor", responsavel=1, aceita_cinza=False, aceita_rgb=True
    )
    def meu_proc(img, p):
        return Resultado(saidas=[("x", img)])

    assert callable(meu_proc) and meu_proc.__name__ == "meu_proc"
    spec = obter_processo("meu_proc")
    assert spec.funcao is meu_proc
    assert spec.parametros == () and spec.descricao == ""


def test_id_duplicado(registro_limpo):
    _registrar("a")
    with pytest.raises(ValueError, match="Já existe"):
        _registrar("a")


def test_listar_preserva_ordem_e_agrupa(registro_limpo):
    _registrar("c", grupo="Ruído")
    _registrar("a", grupo="Cor")
    _registrar("b", grupo="Ruído")
    assert [s.id for s in listar_processos()] == ["c", "a", "b"]
    grupos = processos_por_grupo()
    assert list(grupos) == ["Cor", "Ruído"]  # ordem de GRUPOS, sem grupos vazios
    assert [s.id for s in grupos["Ruído"]] == ["c", "b"]


def test_obter_inexistente(registro_limpo):
    with pytest.raises(ValueError, match="não está registrado"):
        obter_processo("nada")


@pytest.mark.parametrize(
    ("kwargs", "trecho"),
    [
        ({"id": "Com Espaco"}, "id deve"),
        ({"grupo": "Espacial "}, "grupo"),  # erro de digitação comum
        ({"cinza": False, "rgb": False}, "precisa aceitar"),
        ({"parametros": (JANELA, JANELA)}, "repetido"),
        ({"parametros": ("sigma",)}, "objetos Param"),
        ({"parametros": (Param("k", "k", TipoParam.INT, 5),)}, "minimo e maximo"),
        ({"parametros": (Param("k", "k", TipoParam.INT, 300, 0, 255),)}, "padrão inválido"),
        ({"parametros": (Param("k", "k", TipoParam.INT, 1.5, 0, 255),)}, "padrão inválido"),
        ({"parametros": (Param("j", "j", TipoParam.INT, 4, 1, 9, somente_impar=True),)}, "ímpar"),
        ({"parametros": (Param("s", "s", TipoParam.FLOAT, 1.0, 0, 9, somente_impar=True),)}, "INT"),
        ({"parametros": (Param("m", "m", TipoParam.ESCOLHA, "a"),)}, "opcoes"),
        ({"parametros": (Param("m", "m", TipoParam.ESCOLHA, "c", opcoes=("a", "b")),)}, "padrão"),
        ({"parametros": (Param("b", "b", TipoParam.BOOL, 1),)}, "padrão"),
        ({"parametros": (Param("nome ruim", "x", TipoParam.BOOL, True),)}, "identificador"),
    ],
)
def test_spec_invalido_e_recusado_no_registro(registro_limpo, kwargs, trecho):
    with pytest.raises(ValueError, match=trecho):
        _registrar(**{"id": "proc", **kwargs})
    assert listar_processos() == []  # nada fica registrado pela metade


def test_responsavel_invalido(registro_limpo):
    with pytest.raises(ValueError, match="responsavel"):
        registrar_processo(
            id="x", nome="X", grupo="Cor", responsavel=4, aceita_cinza=True, aceita_rgb=True
        )(_identidade)


# --------------------------------------------------------------------- compatibilidade


def test_motivo_desabilitado(registro_limpo):
    so_rgb = _registrar("so_rgb", cinza=False)
    so_cinza = _registrar("so_cinza", rgb=False)
    ambos = _registrar("ambos")
    assert "coloridas" in motivo_desabilitado(so_rgb, TIPO_CINZA)
    assert motivo_desabilitado(so_rgb, TIPO_RGB) is None
    assert "cinza" in motivo_desabilitado(so_cinza, TIPO_RGB)
    assert motivo_desabilitado(ambos, TIPO_CINZA) is None
    assert motivo_desabilitado(ambos, TIPO_RGB) is None


def test_executar_recusa_imagem_incompativel(registro_limpo, img_rgb):
    _registrar("equalizacao", rgb=False)
    with pytest.raises(ValueError, match="não se aplica"):
        executar_processo("equalizacao", img_rgb)


def test_executar_recusa_imagem_fora_da_convencao(registro_limpo):
    _registrar("p")
    with pytest.raises(ValueError, match="uint8"):
        executar_processo("p", np.zeros((4, 4), dtype=np.float32))


# ------------------------------------------------------------------------ parâmetros


def test_parametros_faltantes_recebem_padrao(registro_limpo):
    spec = _registrar(parametros=(JANELA, SIGMA, MODO, PRESERVAR))
    assert validar_parametros(spec, {"sigma": 2}) == {
        "janela": 3,
        "sigma": 2.0,
        "modo": "passa-baixa",
        "preservar": True,
    }


def test_parametros_convertem_tipos_numpy(registro_limpo):
    spec = _registrar(parametros=(JANELA, SIGMA, PRESERVAR))
    p = validar_parametros(
        spec, {"janela": np.int64(5), "sigma": np.float32(0.5), "preservar": np.bool_(False)}
    )
    assert p == {"janela": 5, "sigma": 0.5, "preservar": False}
    assert type(p["janela"]) is int and type(p["sigma"]) is float and type(p["preservar"]) is bool


@pytest.mark.parametrize(
    ("valores", "trecho"),
    [
        ({"janela": 4}, "ímpar"),
        ({"janela": 33}, "no máximo 31"),
        ({"janela": -1}, "no mínimo 1"),
        ({"janela": 3.0}, "inteiro"),
        ({"janela": True}, "número"),
        ({"sigma": 0.0}, "no mínimo 0.1"),
        ({"sigma": float("nan")}, "número"),
        ({"sigma": "2"}, "número"),
        ({"modo": "passa-banda"}, "uma de"),
        ({"preservar": 1}, "verdadeiro ou falso"),
        ({"inexistente": 1}, "não tem"),
    ],
)
def test_parametros_invalidos(registro_limpo, valores, trecho):
    spec = _registrar(parametros=(JANELA, SIGMA, MODO, PRESERVAR))
    with pytest.raises(ValueError, match=trecho):
        validar_parametros(spec, valores)


def test_mensagem_usa_rotulo_da_interface(registro_limpo):
    spec = _registrar(parametros=(SIGMA,))
    with pytest.raises(ValueError, match="Desvio padrão σ"):
        validar_parametros(spec, {"sigma": 100})


# --------------------------------------------------------------------------- execução


def test_executar_passa_parametros_completos(registro_limpo, img_cinza):
    _registrar(parametros=(JANELA, SIGMA))
    res = executar_processo("teste", img_cinza, {"janela": 7})
    assert res.info == {"janela": 7, "sigma": 1.0}
    np.testing.assert_array_equal(res.saidas[0][1], img_cinza)


def test_executar_aceita_spec_ou_id(registro_limpo, img_cinza):
    spec = _registrar()
    assert executar_processo(spec, img_cinza).saidas[0][0] == "Saída"


def test_processo_nao_consegue_modificar_entrada(registro_limpo, img_cinza):
    def malcomportado(img, p):
        img[0, 0] = 99  # viola o contrato
        return Resultado(saidas=[("x", img)])

    _registrar(funcao=malcomportado)
    original = img_cinza.copy()
    with pytest.raises(ValueError, match="read-only"):
        executar_processo("teste", img_cinza)
    np.testing.assert_array_equal(img_cinza, original)


@pytest.mark.parametrize(
    ("resultado", "trecho"),
    [
        (np.zeros((4, 4), np.uint8), "não Resultado"),
        (Resultado(saidas=[]), "nenhuma imagem"),
        (Resultado(saidas=[("x", np.zeros((4, 4)))]), "uint8"),
        (Resultado(saidas=[np.zeros((4, 4), np.uint8)]), "tupla"),
        (
            Resultado(
                saidas=[("x", np.zeros((4, 4), np.uint8))],
                acessorios=[Histograma("h", np.zeros(10))],
            ),
            "shape \\(256,\\)",
        ),
        (
            Resultado(
                saidas=[("x", np.zeros((4, 4), np.uint8))], acessorios=[("lap", np.zeros((4, 4)))]
            ),
            "acessório 'lap'",
        ),
    ],
)
def test_saida_fora_do_contrato(registro_limpo, img_cinza, resultado, trecho):
    _registrar(funcao=lambda img, p: resultado)
    with pytest.raises(ErroDeImplementacao, match=trecho):
        executar_processo("teste", img_cinza)


def test_erro_de_implementacao_identifica_responsavel(registro_limpo, img_cinza):
    _registrar(funcao=lambda img, p: None)
    with pytest.raises(ErroDeImplementacao, match="Integrante 1"):
        executar_processo("teste", img_cinza)


def test_resultado_completo_valido(registro_limpo, img_rgb):
    def completo(img, p):
        canais = [(c, img[..., i].copy()) for i, c in enumerate("RGB")]
        return Resultado(
            saidas=canais,
            acessorios=[Histograma("R", np.bincount(img[..., 0].ravel(), minlength=256))],
            info={"canais": 3},
        )

    _registrar(funcao=completo, cinza=False)
    res = executar_processo("teste", img_rgb)
    assert [t for t, _ in res.saidas] == ["R", "G", "B"]


def test_carregar_processos_nao_falha():
    # Registro real (sem fixture): importar todos os módulos de processo deve funcionar.
    specs = registro.carregar_processos()
    assert isinstance(specs, list)
