@echo off
setlocal

REM Function to create directory if it does not exist
:CREATE_DIR_IF_NOT_EXISTS
IF NOT EXIST "%1" (
    mkdir "%1"
)
exit /b

REM Check if .env file exists
IF NOT EXIST ".env" (
    echo .env file not found. Running install.bat...
    call install.bat
)

REM Load environment variables from .env file
FOR /F "tokens=1,* delims==" %%i IN ('.env') DO (
    IF NOT "%%i"=="" (
        set "%%i=%%j"
    )
)

REM Activate virtual environment
call "%VENV_PATH%\Scripts\activate.bat"

REM Menu options
echo Select an option:
echo 1. deforum ui: PyQt6 UI for configuring and running animations
echo 2. deforum webui: Streamlit web UI for configuring and running animations
echo 3. deforum animatediff: Command-line tool for running animations
echo 4. deforum test: Run through all motion presets for testing purposes
echo 5. deforum api: FastAPI server
echo 6. deforum setup: Install Stable-Fast optimizations
echo 7. deforum runsingle --file %USERPROFILE%\deforum\presets\preset.txt: Run single settings file
echo 8. deforum config
echo 9. deforum unittest: Run unit test

REM Read user choice
set /p choice=Enter the number of your choice: 

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
    deforum runsingle --file %USERPROFILE%\deforum\presets\preset.txt
) else if "%choice%"=="8" (
    deforum config
) else if "%choice%"=="9" (
    deforum unittest
) else (
    echo Invalid choice.
)

endlocal
