@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "jarvis.py" %*
echo.
echo JARVIS stopped. Press any key to close this window.
pause >nul
