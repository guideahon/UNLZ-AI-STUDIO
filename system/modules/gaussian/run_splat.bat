@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo [SharpSplat] Activating environment...
call .venv\Scripts\activate

:INPUT_LOOP
set "INPUT_PATH="
set /p "INPUT_PATH=Enter path to image or folder (or drag and drop here): "
set "INPUT_PATH=!INPUT_PATH:"=!"

if "%INPUT_PATH%"=="" goto INPUT_LOOP
if not exist "%INPUT_PATH%" (
    echo Error: Path does not exist.
    goto INPUT_LOOP
)

set "TIMESTAMP=%DATE:~10,4%-%DATE:~4,2%-%DATE:~7,2%_%TIME:~0,2%-%TIME:~3,2%-%TIME:~6,2%"
set "TIMESTAMP=!TIMESTAMP: =0!"
set "OUTPUT_DIR=output\splat_%TIMESTAMP%"

echo [SharpSplat] Processing...
echo Input: "%INPUT_PATH%"
echo Output: "%OUTPUT_DIR%"

mkdir "%OUTPUT_DIR%" 2>nul

echo Running sharp predict...
sharp predict -i "%INPUT_PATH%" -o "%OUTPUT_DIR%"

if errorlevel 1 (
    echo [SharpSplat] Error during prediction.
    pause
    exit /b 1
)

echo [SharpSplat] Prediction complete.
echo [SharpSplat] Setting up viewer...

if exist "%OUTPUT_DIR%\gaussians\scene.ply" (
    copy "viewer_template.html" "%OUTPUT_DIR%\gaussians\index.html" >nul
    echo [SharpSplat] Viewer created at: %OUTPUT_DIR%\gaussians\index.html
    
    echo.
    echo ========================================================
    echo DONE!
    echo To view on Quest 3:
    echo 1. Connect Quest 3 to PC.
    echo 2. Copy the folder "%OUTPUT_DIR%\gaussians" to the Quest.
    echo 3. Open index.html in Meta Browser OR open scene.ply in Scaniverse.
    echo ========================================================
    
    explorer "%OUTPUT_DIR%\gaussians"
) else (
    echo [SharpSplat] Warning: scene.ply not found in output.
)

pause
