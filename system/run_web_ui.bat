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
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found. Install Python and try again.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

cd /d "%~dp0web_ui"
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
)
echo.
echo Starting Web Bridge on http://127.0.0.1:8787
if not exist "%~dp0logs" mkdir "%~dp0logs"
powershell -NoProfile -Command ^
  "$conn = Get-NetTCPConnection -LocalPort 8787 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1;" ^
  "if ($conn) { Stop-Process -Id $conn.OwningProcess -Force }"
powershell -NoProfile -Command ^
  "$out='%~dp0logs\\web_bridge.out.log'; $err='%~dp0logs\\web_bridge.err.log';" ^
  "Start-Process -FilePath 'python' -ArgumentList '%~dp0web_bridge.py' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err"
for /l %%i in (1,1,8) do (
  powershell -NoProfile -Command "if ((Test-NetConnection -ComputerName 127.0.0.1 -Port 8787).TcpTestSucceeded) { exit 0 } else { exit 1 }"
  if not errorlevel 1 goto bridge_ready
  timeout /t 1 /nobreak >nul
)
echo WARNING: Web Bridge did not start. Check %~dp0logs\web_bridge.err.log
:bridge_ready
echo.
echo Starting Web UI at http://localhost:3000
npm run dev
echo.
echo Web UI closed.
pause
