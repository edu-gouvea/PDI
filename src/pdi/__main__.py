"""Ponto de entrada: ``python -m pdi [imagem.png]`` (I1).

O caminho de uma imagem é opcional; se informado, ela já abre como entrada.
"""

import os
import sys


def configurar_ambiente_qt() -> None:
    """Ajusta variáveis do Qt ANTES de criar o QApplication. Respeita o que o usuário definir.

    No GNOME, o Qt carrega por padrão o plugin de tema ``gtk3`` do PySide6. No GNOME 50
    (Ubuntu 26.04) esse plugin lê uma configuração que não existe mais
    (``org.gnome.settings-daemon.plugins.xsettings antialiasing``), e a GLib aborta o
    programa (core dumped). O tema ``gnome`` embutido no Qt não usa GTK e funciona. KDE e
    LXQt não passam pelo plugin GTK e ficam como estão.
    """
    if not sys.platform.startswith("linux"):
        return
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    if "QT_QPA_PLATFORMTHEME" not in os.environ and not any(d in desktop for d in ("kde", "lxqt")):
        os.environ["QT_QPA_PLATFORMTHEME"] = "gnome"
    # Aviso inofensivo do Wayland ao fechar janelas; só polui o terminal.
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    configurar_ambiente_qt()

    # Imports aqui dentro: o Qt só é carregado depois de configurado.
    from PySide6.QtWidgets import QApplication

    from pdi.ui.janela_principal import JanelaPrincipal

    app = QApplication(argv)
    app.setApplicationName("PDI")
    janela = JanelaPrincipal()
    janela.show()
    if len(argv) > 1:
        janela.abrir_arquivo(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
