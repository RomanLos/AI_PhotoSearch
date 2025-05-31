@echo off
cd /d "%~dp0"
echo Running AI-PhotoSearch...
"%~dp0system\python\python.exe" -m streamlit run app2.py
pause