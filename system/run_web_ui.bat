@echo off
cd /d "%~dp0"
echo =================================================
echo UNLZ AI STUDIO - WEB UI
echo =================================================
echo.
where npm >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm not found. Install Node.js LTS and try again.
    echo https://nodejs.org/
    pause
    exit /b 1
)

cd /d "%~dp0web_ui"
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
)
echo.
echo Starting Web UI at http://localhost:3000
npm run dev
echo.
echo Web UI closed.
pause
