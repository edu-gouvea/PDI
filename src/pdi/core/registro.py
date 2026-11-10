"""Registro de processos: decorador ``@registrar_processo`` e execução validada (I1).

A interface **nunca** importa funções de processo diretamente. Ela chama
``carregar_processos()`` e trabalha só com os ``ProcessoSpec`` registrados aqui.

O registro também protege o grupo contra erros comuns:

- **Na importação** (``_validar_spec``): id repetido, grupo com erro de digitação,
  parâmetro sem limites, padrão fora da faixa, janela par etc. O erro aparece ao
  rodar ``pytest`` ou abrir o app, e não no dia da apresentação.
- **Na execução** (``executar_processo``): imagem incompatível, parâmetros fora da
  faixa, processo que tenta modificar a entrada ou devolve saída fora do contrato.
"""

import importlib
import re
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np

from pdi.core.contrato import Histograma, Param, ProcessoSpec, Resultado, TipoParam
from pdi.core.util import TIPO_CINZA, TIPO_RGB, tipo_da_imagem

FuncaoProcesso = Callable[[np.ndarray, dict[str, Any]], Resultado]

# Grupos válidos, na ordem em que aparecem na interface.
GRUPOS: tuple[str, ...] = ("Cor", "Intensidade", "Espacial", "Frequência", "Ruído", "Restauração")

# id -> spec. Dicionários preservam a ordem de inserção (ordem de registro).
_PROCESSOS: dict[str, ProcessoSpec] = {}

_PADRAO_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class ErroDeImplementacao(RuntimeError):
    """O processo violou o contrato (ex.: saída float64). É bug de código, não do usuário."""


# --------------------------------------------------------------------------- registro


def registrar_processo(
    *,
    id: str,
    nome: str,
    grupo: str,
    responsavel: int,
    aceita_cinza: bool,
    aceita_rgb: bool,
    parametros: tuple[Param, ...] = (),
    descricao: str = "",
) -> Callable[[FuncaoProcesso], FuncaoProcesso]:
    """Decorador que registra a função como um processo disponível na interface.

    Devolve a própria função, sem alterações, então ela continua chamável nos testes.
    """

    def decorador(funcao: FuncaoProcesso) -> FuncaoProcesso:
        spec = ProcessoSpec(
            id=id,
            nome=nome,
            grupo=grupo,
            responsavel=responsavel,
            aceita_cinza=aceita_cinza,
            aceita_rgb=aceita_rgb,
            parametros=tuple(parametros),
            funcao=funcao,
            descricao=descricao,
        )
        _validar_spec(spec)
        if spec.id in _PROCESSOS:
            raise ValueError(f"Já existe um processo registrado com id '{spec.id}'.")
        _PROCESSOS[spec.id] = spec
        return funcao

    return decorador


def carregar_processos() -> list[ProcessoSpec]:
    """Importa ``pdi.processos`` (o que dispara os decoradores) e devolve a lista."""
    importlib.import_module("pdi.processos")
    return listar_processos()


def listar_processos() -> list[ProcessoSpec]:
    """Todos os processos registrados, na ordem de registro."""
    return list(_PROCESSOS.values())


def processos_por_grupo() -> dict[str, list[ProcessoSpec]]:
    """Processos agrupados na ordem de ``GRUPOS``. Grupos vazios são omitidos."""
    agrupados: dict[str, list[ProcessoSpec]] = {g: [] for g in GRUPOS}
    for spec in _PROCESSOS.values():
        agrupados[spec.grupo].append(spec)
    return {g: specs for g, specs in agrupados.items() if specs}


def obter_processo(id_processo: str) -> ProcessoSpec:
    try:
        return _PROCESSOS[id_processo]
    except KeyError:
        raise ValueError(f"Processo '{id_processo}' não está registrado.") from None


# --------------------------------------------------------------------- compatibilidade


def motivo_desabilitado(spec: ProcessoSpec, tipo: str) -> str | None:
    """Motivo (para tooltip) de o processo não se aplicar a ``tipo``; ``None`` se aplica."""
    if tipo == TIPO_CINZA and not spec.aceita_cinza:
        return "Disponível apenas para imagens coloridas (RGB)."
    if tipo == TIPO_RGB and not spec.aceita_rgb:
        return "Disponível apenas para imagens em escala de cinza."
    return None


def aceita_imagem(spec: ProcessoSpec, img: np.ndarray) -> bool:
    return motivo_desabilitado(spec, tipo_da_imagem(img)) is None


# --------------------------------------------------------------------------- execução


def validar_parametros(spec: ProcessoSpec, valores: Mapping[str, Any]) -> dict[str, Any]:
    """Confere os valores e devolve um dict novo e completo (faltantes recebem o padrão).

    Levanta ``ValueError`` com mensagem amigável se algum valor for inválido.
    """
    conhecidos = {p.nome for p in spec.parametros}
    desconhecidos = set(valores) - conhecidos
    if desconhecidos:
        nomes = ", ".join(sorted(desconhecidos))
        raise ValueError(f"'{spec.nome}' não tem o(s) parâmetro(s): {nomes}.")

    return {p.nome: _converter_valor(p, valores.get(p.nome, p.padrao)) for p in spec.parametros}


def executar_processo(
    processo: ProcessoSpec | str,
    img: np.ndarray,
    parametros: Mapping[str, Any] | None = None,
) -> Resultado:
    """Executa o processo com todas as verificações do contrato.

    É o único caminho que a interface usa para rodar um processo.
    """
    spec = obter_processo(processo) if isinstance(processo, str) else processo

    motivo = motivo_desabilitado(spec, tipo_da_imagem(img))
    if motivo:
        raise ValueError(f"'{spec.nome}' não se aplica a esta imagem. {motivo}")

    p = validar_parametros(spec, parametros or {})

    # Cópia somente leitura: se o processo tentar escrever na entrada, o NumPy levanta
    # erro na hora, e a imagem original exibida na interface nunca é corrompida.
    entrada = img.copy()
    entrada.flags.writeable = False

    resultado = spec.funcao(entrada, p)
    _validar_resultado(spec, resultado)
    return resultado


# ------------------------------------------------------------------ validações internas


def _converter_valor(p: Param, valor: Any) -> Any:
    """Converte ``valor`` para o tipo de ``p`` e confere limites. ``ValueError`` se inválido."""
    rotulo = f"'{p.rotulo}'"

    if p.tipo is TipoParam.BOOL:
        if not isinstance(valor, bool | np.bool_):
            raise ValueError(f"{rotulo} deve ser verdadeiro ou falso.")
        return bool(valor)

    if p.tipo is TipoParam.ESCOLHA:
        if valor not in p.opcoes:
            raise ValueError(f"{rotulo} deve ser uma de: {', '.join(p.opcoes)}.")
        return valor

    # Numéricos. bool é subclasse de int em Python, por isso a exclusão explícita.
    if isinstance(valor, bool | np.bool_):
        raise ValueError(f"{rotulo} deve ser um número.")
    if p.tipo is TipoParam.INT:
        if not isinstance(valor, Integral):
            raise ValueError(f"{rotulo} deve ser um número inteiro.")
        valor = int(valor)
    else:
        if not isinstance(valor, Real) or not np.isfinite(valor):
            raise ValueError(f"{rotulo} deve ser um número.")
        valor = float(valor)

    if p.minimo is not None and valor < p.minimo:
        raise ValueError(f"{rotulo} deve ser no mínimo {_fmt(p.minimo)} (recebido {_fmt(valor)}).")
    if p.maximo is not None and valor > p.maximo:
        raise ValueError(f"{rotulo} deve ser no máximo {_fmt(p.maximo)} (recebido {_fmt(valor)}).")
    if p.somente_impar and valor % 2 == 0:
        raise ValueError(f"{rotulo} deve ser ímpar (recebido {valor}).")
    return valor


def _fmt(x: float) -> str:
    """3.0 -> '3', 0.5 -> '0.5'."""
    return f"{x:g}"


def _validar_spec(spec: ProcessoSpec) -> None:
    """Confere a especificação no momento do registro. ``ValueError`` se inconsistente."""
    pref = f"Processo '{spec.id}':"
    if not _PADRAO_ID.match(spec.id):
        raise ValueError(
            f"{pref} id deve usar só minúsculas, dígitos e '_' (ex.: 'filtro_mediana')."
        )
    if not spec.nome.strip():
        raise ValueError(f"{pref} nome vazio.")
    if spec.grupo not in GRUPOS:
        raise ValueError(f"{pref} grupo '{spec.grupo}' inválido. Use um de: {', '.join(GRUPOS)}.")
    if spec.responsavel not in (1, 2, 3):
        raise ValueError(f"{pref} responsavel deve ser 1, 2 ou 3.")
    if not (spec.aceita_cinza or spec.aceita_rgb):
        raise ValueError(f"{pref} precisa aceitar cinza, RGB ou ambos.")
    if not callable(spec.funcao):
        raise ValueError(f"{pref} funcao não é chamável.")

    if not all(isinstance(p, Param) for p in spec.parametros):
        raise ValueError(f"{pref} parametros deve conter apenas objetos Param.")
    nomes = [p.nome for p in spec.parametros]
    if len(nomes) != len(set(nomes)):
        raise ValueError(f"{pref} há parâmetros com nome repetido.")
    for p in spec.parametros:
        _validar_param(pref, p)


def _validar_param(pref: str, p: Param) -> None:
    pref = f"{pref} parâmetro '{p.nome}':"
    if not p.nome.isidentifier():
        raise ValueError(f"{pref} nome deve ser um identificador Python válido.")
    if p.somente_impar and p.tipo is not TipoParam.INT:
        raise ValueError(f"{pref} somente_impar só vale para TipoParam.INT.")

    if p.tipo in (TipoParam.INT, TipoParam.FLOAT):
        # A interface precisa dos limites para montar spinbox e slider.
        if p.minimo is None or p.maximo is None:
            raise ValueError(f"{pref} parâmetros numéricos precisam de minimo e maximo.")
        if p.minimo > p.maximo:
            raise ValueError(f"{pref} minimo maior que maximo.")
        if p.passo is not None and p.passo <= 0:
            raise ValueError(f"{pref} passo deve ser positivo.")
    elif p.tipo is TipoParam.ESCOLHA and not p.opcoes:
        raise ValueError(f"{pref} TipoParam.ESCOLHA precisa de opcoes.")

    # O próprio padrão tem de passar pela validação que os valores do usuário passam.
    try:
        _converter_valor(p, p.padrao)
    except ValueError as e:
        raise ValueError(f"{pref} padrão inválido. {e}") from None


def _validar_resultado(spec: ProcessoSpec, resultado: Any) -> None:
    """Confere a saída do processo. ``ErroDeImplementacao`` se fora do contrato."""
    pref = f"Erro interno em '{spec.nome}' (Integrante {spec.responsavel}):"

    if not isinstance(resultado, Resultado):
        raise ErroDeImplementacao(f"{pref} devolveu {type(resultado).__name__}, não Resultado.")
    if not resultado.saidas:
        raise ErroDeImplementacao(f"{pref} nenhuma imagem de saída.")

    def checar_imagem(item: Any, onde: str) -> None:
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
            raise ErroDeImplementacao(f"{pref} {onde} deve ser uma tupla (titulo, imagem).")
        try:
            tipo_da_imagem(item[1])
        except ValueError as e:
            raise ErroDeImplementacao(f"{pref} {onde} '{item[0]}': {e}") from None

    for item in resultado.saidas:
        checar_imagem(item, "saída")
    for item in resultado.acessorios:
        if isinstance(item, Histograma):
            valores = np.asarray(item.valores)
            if valores.shape != (256,):
                raise ErroDeImplementacao(
                    f"{pref} histograma '{item.titulo}' deve ter shape (256,), tem {valores.shape}."
                )
        else:
            checar_imagem(item, "acessório")
    if not isinstance(resultado.info, dict):
        raise ErroDeImplementacao(f"{pref} info deve ser um dict.")
