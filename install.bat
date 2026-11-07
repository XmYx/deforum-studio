@echo off
REM Deforum installer for Windows.
REM   install.bat        interactive
REM   install.bat -s     silent, defaults
REM Environment overrides: DEFORUM_PYTHON (3.11-3.14, default 3.12),
REM   DEFORUM_TORCH_BACKEND (cu132 / cu130 / cu128 / cpu, default: cu132 when nvidia-smi works, else cpu)
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "VENV_DIR=venv"
set "ROOT_DIR=%USERPROFILE%\deforum"
set "COMFY_DIR=src\ComfyUI"
set "EXTRA_MODELS="
set "SILENT=0"
if "%~1"=="-s" set "SILENT=1"
if "%DEFORUM_PYTHON%"=="" set "DEFORUM_PYTHON=3.12"

if "%SILENT%"=="0" (
    set /p "VENV_INPUT=Virtual environment path (default: %VENV_DIR%): "
    if not "!VENV_INPUT!"=="" set "VENV_DIR=!VENV_INPUT!"
    set /p "ROOT_INPUT=Deforum data folder (default: %ROOT_DIR%): "
    if not "!ROOT_INPUT!"=="" set "ROOT_DIR=!ROOT_INPUT!"
    set /p "EXTRA_INPUT=Existing ComfyUI models folder to reuse (optional): "
    if not "!EXTRA_INPUT!"=="" set "EXTRA_MODELS=!EXTRA_INPUT!"
)

if "%DEFORUM_TORCH_BACKEND%"=="" (
    nvidia-smi -L >nul 2>&1 && (set "DEFORUM_TORCH_BACKEND=cu132") || (set "DEFORUM_TORCH_BACKEND=cpu")
)
echo Python %DEFORUM_PYTHON%, torch backend: %DEFORUM_TORCH_BACKEND%

for %%D in (checkpoints diffusion_models text_encoders vae loras) do if not exist "%ROOT_DIR%\models\%%D" mkdir "%ROOT_DIR%\models\%%D"
git submodule update --init src/ComfyUI

where uv >nul 2>&1
if %errorlevel%==0 (
    uv venv -p %DEFORUM_PYTHON% "%VENV_DIR%" || exit /b 1
    set "PIP=uv pip install -p %VENV_DIR%"
) else (
    py -%DEFORUM_PYTHON% -m venv "%VENV_DIR%" || exit /b 1
    "%VENV_DIR%\Scripts\python.exe" -m pip install -U pip
    set "PIP=%VENV_DIR%\Scripts\python.exe -m pip install"
)

!PIP! torch torchvision --index-url https://download.pytorch.org/whl/%DEFORUM_TORCH_BACKEND% || exit /b 1
!PIP! -r "%COMFY_DIR%\requirements.txt" || exit /b 1
!PIP! -e . || exit /b 1

for /f "delims=" %%i in ('powershell -NoProfile -Command "[System.IO.Path]::GetFullPath('%VENV_DIR%')"') do set "VENV_PATH=%%i"
for /f "delims=" %%i in ('powershell -NoProfile -Command "[System.IO.Path]::GetFullPath('%ROOT_DIR%')"') do set "ROOT_PATH=%%i"
for /f "delims=" %%i in ('powershell -NoProfile -Command "[System.IO.Path]::GetFullPath('%COMFY_DIR%')"') do set "COMFY_PATH=%%i"
(
    echo VENV_PATH=%VENV_PATH%
    echo ROOT_PATH=%ROOT_PATH%
    echo COMFY_PATH=%COMFY_PATH%
    if not "%EXTRA_MODELS%"=="" echo EXTRA_MODEL_PATHS=%EXTRA_MODELS%
) > .env

"%VENV_PATH%\Scripts\python.exe" -c "import torch; from deforum.utils.device import get_device; print('torch', torch.__version__, 'device:', get_device())"
echo.
echo Done. Put models in %ROOT_PATH%\models\checkpoints (or set EXTRA_MODEL_PATHS in .env), then run start.bat
