@echo off
cd /d "%~dp0"

if not exist ".venv" (
    echo Environment not found. Running setup...
    call setup.bat
)

echo Starting SharpSplat GUI...
start "" .venv\Scripts\pythonw.exe gui.py
