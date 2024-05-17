@echo off
setlocal

REM Get current working directory
set "CURRENT_DIR=%cd%"

REM Default values
set "VENV_DIR=venv"
set "ROOT_DIR=%USERPROFILE%\deforum"
set "COMFY_DIR=src\ComfyUI"
set "SILENT=0"

REM Parse arguments
:PARSE_ARGUMENTS
for %%A in (%*) do (
    if "%%A"=="-s" (
        set "SILENT=1"
    ) else (
        echo Invalid option: %%A
        exit /b 1
    )
)
goto :AFTER_PARSE_ARGUMENTS

:AFTER_PARSE_ARGUMENTS

REM Collect VENV_PATH
if "%SILENT%"=="0" (
    set /p "VENV_INPUT=Enter virtual environment path (default: %VENV_DIR%): "
    if not "%VENV_INPUT%"=="" (
        set "VENV_DIR=%VENV_INPUT%"
    )
)

REM Collect ROOT_PATH
if "%SILENT%"=="0" (
    set /p "ROOT_INPUT=Enter root directory path (default: %ROOT_DIR%): "
    if not "%ROOT_INPUT%"=="" (
        set "ROOT_DIR=%ROOT_INPUT%"
    )
)

REM Collect COMFY_PATH
if "%SILENT%"=="0" (
    set /p "COMFY_INPUT=Enter ComfyUI directory path (default: %COMFY_DIR%): "
    if not "%COMFY_INPUT%"=="" (
        set "COMFY_DIR=%COMFY_INPUT%"
    )
)

REM Convert paths to full paths
for /f "delims=" %%i in ('powershell -Command "[System.IO.Path]::GetFullPath('%VENV_DIR%')"') do set "VENV_PATH=%%i"
for /f "delims=" %%i in ('powershell -Command "[System.IO.Path]::GetFullPath('%ROOT_DIR%')"') do set "ROOT_PATH=%%i"
for /f "delims=" %%i in ('powershell -Command "[System.IO.Path]::GetFullPath('%COMFY_DIR%')"') do set "COMFY_PATH=%%i"

REM Trim trailing spaces from paths
for /f "tokens=* delims= " %%A in ("%VENV_PATH%") do set "VENV_PATH=%%A"
for /f "tokens=* delims= " %%A in ("%ROOT_PATH%") do set "ROOT_PATH=%%A"
for /f "tokens=* delims= " %%A in ("%COMFY_PATH%") do set "COMFY_PATH=%%A"

REM Save to .env file
(
    echo VENV_PATH=%VENV_PATH%
    echo ROOT_PATH=%ROOT_PATH%
    echo COMFY_PATH=%COMFY_PATH%
) > "%CURRENT_DIR%\.env"

git submodule update --init --recursive

REM Create virtual environment and install dependencies
pip install virtualenv
python -m virtualenv "%VENV_PATH%" -p python3.10
call "%VENV_PATH%\Scripts\activate.bat"
pip install -e .

echo Installation completed successfully.
