@echo off
cd /d "%~dp0"
echo Launching UNLZ AI Studio...
python studio_gui.py
echo.
echo Application crashed! See error above.
pause
