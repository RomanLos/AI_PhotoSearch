@echo off
set PYTHON_DIR=%~dp0system\python-3.10.11.amd64
set SCRIPT_PATH=%~dp0app.py

"%PYTHON_DIR%\python.exe" -m streamlit run "%SCRIPT_PATH%"
pause
