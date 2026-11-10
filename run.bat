@echo off
rem Abre o programa com um duplo clique (ou: run.bat [imagem.png])
rem
rem Na primeira vez: cria o ambiente virtual (.venv) e instala as dependencias.
rem Nas seguintes: abre direto (so reinstala se o requirements.txt mudar).
setlocal
cd /d "%~dp0"

rem 1) Python 3.11 ou mais novo: tenta o lancador "py" e depois "python"
set "PYTHON="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && set "PYTHON=py -3"
if not defined PYTHON (
    python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && set "PYTHON=python"
)
if not defined PYTHON (
    echo.
    echo ERRO: Python 3.11 ou mais novo nao encontrado.
    echo   Instale em https://www.python.org/downloads/
    echo   e marque a opcao "Add python.exe to PATH" no instalador.
    goto :erro
)

rem 2) Ambiente virtual
if not exist ".venv\Scripts\python.exe" (
    echo Criando o ambiente virtual em .venv ^(so na primeira vez^)...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERRO: nao foi possivel criar o ambiente virtual.
        goto :erro
    )
)

rem 3) Dependencias (so quando o requirements.txt mudou)
".venv\Scripts\python.exe" scripts\verificar_requisitos.py
if errorlevel 1 (
    echo Instalando as dependencias ^(pode levar alguns minutos na primeira vez^)...
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERRO: a instalacao das dependencias falhou. Confira a conexao com a internet.
        goto :erro
    )
    ".venv\Scripts\python.exe" scripts\verificar_requisitos.py --marcar
)

rem 4) Abre o programa
".venv\Scripts\python.exe" -m pdi %*
if errorlevel 1 goto :erro
exit /b 0

:erro
echo.
pause
exit /b 1
