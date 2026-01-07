@echo off
setlocal

echo [SharpSplat] Setting up environment...

cd /d "%~dp0"

if not exist ".venv" (
    echo [SharpSplat] Creating virtual environment...
    python -m venv .venv
) else (
    echo [SharpSplat] Virtual environment already exists.
)

call .venv\Scripts\activate

echo [SharpSplat] Upgrading pip...
python -m pip install --upgrade pip

echo [SharpSplat] Installing GUI dependencies...
pip install customtkinter packaging

echo [SharpSplat] Installing dependencies from ml-sharp...
pip install -r ..\ml-sharp\requirements.txt

echo [SharpSplat] Installing ml-sharp in editable mode...
pip install -e ..\ml-sharp

echo [SharpSplat] Setup complete!
pause
