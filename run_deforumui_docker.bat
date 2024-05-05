@echo off
SETLOCAL EnableExtensions

set XMING_PATH="C:\Program Files (x86)\Xming\Xming.exe"
set VCXSRV_PATH="C:\Program Files\VcXsrv\vcxsrv.exe"

if exist %XMING_PATH% (
    echo Xming is installed.
) else if exist %VCXSRV_PATH% (
    echo VcXsrv is installed.
) else (
    echo Neither Xming nor VcXsrv is installed.
    echo Please download and install one of the following X server applications:
    echo Xming can be downloaded from: https://sourceforge.net/projects/xming/
    echo VcXsrv can be downloaded from: https://sourceforge.net/projects/vcxsrv/
    pause
    exit /b
)

REM Continue with the Docker commands
REM Check if the Docker image already exists
SET IMAGE_NAME=deforum-desktop
FOR /f "tokens=*" %%i IN ('docker images -q %IMAGE_NAME%') DO SET IMAGE_EXISTS=%%i

REM Build the Docker image only if it does not exist
IF "%IMAGE_EXISTS%"=="" (
    echo Image %IMAGE_NAME% does not exist. Building...
    docker build -f Dockerfile-desktop -t %IMAGE_NAME% .
) ELSE (
    echo Image %IMAGE_NAME% already exists. Skipping build.
)

REM Run the Docker container
docker run -it --rm ^
    --gpus all ^
    -e DISPLAY=host.docker.internal:0.0 ^
    -v %USERPROFILE%\deforum:/home/user/deforum ^
    --network host ^
    %IMAGE_NAME%
