@echo off
setlocal

REM Function to create directory if it does not exist
:CREATE_DIR_IF_NOT_EXISTS
IF NOT EXIST "%1" (
    mkdir "%1"
)
exit /b

REM Check if .env file exists
IF NOT EXIST .env (
    echo .env file not found. Running install.bat...
    call install.bat
)

REM Load environment variables from .env file
for /f "delims=" %%i in ('.env') do set %%i

REM Activate virtual environment
call "%VENV_PATH%\Scripts\activate.bat"

REM Menu options
set "options[1]=deforum ui: PyQt6 UI for configuring and running animations"
set "options[2]=deforum webui: Streamlit web UI for configuring and running animations"
set "options[3]=deforum animatediff: Command-line tool for running animations"
set "options[4]=deforum test: Run through all motion presets for testing purposes"
set "options[5]=deforum api: FastAPI server"
set "options[6]=deforum setup: Install Stable-Fast optimizations"
set "options[7]=deforum runsingle --file: Run single settings file"
set "options[8]=deforum config"
set "options[9]=deforum unittest: Run unit test"

REM Display menu and get user choice
echo Select an option:
for %%i in (1 2 3 4 5 6 7 8 9) do (
    call echo %%i. %%options[%%i]%%
)

REM Read user choice
set /p "choice=Enter the number of your choice: "

REM Execute the corresponding command
if "%choice%"=="1" (
    deforum ui
) else if "%choice%"=="2" (
    deforum webui
) else if "%choice%"=="3" (
    deforum animatediff
) else if "%choice%"=="4" (
    deforum test
) else if "%choice%"=="5" (
    deforum api
) else if "%choice%"=="6" (
    deforum setup
) else if "%choice%"=="7" (
    set /p "preset_file=Enter the preset file path (default: %USERPROFILE%\deforum\presets\preset.txt): "
    if "%preset_file%"=="" (
        set "preset_file=%USERPROFILE%\deforum\presets\preset.txt"
    )
    deforum runsingle --file "%preset_file%"
) else if "%choice%"=="8" (
    deforum config
) else if "%choice%"=="9" (
    deforum unittest
) else (
    echo Invalid choice.
)

endlocal
