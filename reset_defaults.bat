@echo off
setlocal

set SETTINGS_FILE=system\data\app_settings.json

if exist "%SETTINGS_FILE%" (
  del /f /q "%SETTINGS_FILE%"
  echo Configuracion reiniciada: %SETTINGS_FILE%
) else (
  echo No se encontro %SETTINGS_FILE%. No hay nada para resetear.
)

echo Listo.
endlocal
