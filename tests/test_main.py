"""Testes da configuração de ambiente do Qt feita em ``python -m pdi``."""

import os

import pytest

from pdi.__main__ import configurar_ambiente_qt


@pytest.fixture
def ambiente(monkeypatch):
    """Ambiente Linux limpo: sem tema nem regras de log definidos."""
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("QT_QPA_PLATFORMTHEME", raising=False)
    monkeypatch.delenv("QT_LOGGING_RULES", raising=False)
    return monkeypatch


@pytest.mark.parametrize("desktop", ["ubuntu:GNOME", "GNOME", "X-Cinnamon", "XFCE", ""])
def test_evita_plugin_gtk3(ambiente, desktop):
    ambiente.setenv("XDG_CURRENT_DESKTOP", desktop)
    configurar_ambiente_qt()
    assert os.environ["QT_QPA_PLATFORMTHEME"] == "gnome"


@pytest.mark.parametrize("desktop", ["KDE", "LXQt"])
def test_nao_mexe_em_kde_e_lxqt(ambiente, desktop):
    ambiente.setenv("XDG_CURRENT_DESKTOP", desktop)
    configurar_ambiente_qt()
    assert "QT_QPA_PLATFORMTHEME" not in os.environ


def test_respeita_escolha_do_usuario(ambiente):
    ambiente.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    ambiente.setenv("QT_QPA_PLATFORMTHEME", "qt6ct")
    configurar_ambiente_qt()
    assert os.environ["QT_QPA_PLATFORMTHEME"] == "qt6ct"


def test_fora_do_linux_nao_mexe_em_nada(ambiente):
    ambiente.setattr("sys.platform", "win32")
    configurar_ambiente_qt()
    assert "QT_QPA_PLATFORMTHEME" not in os.environ
    assert "QT_LOGGING_RULES" not in os.environ
