@echo off
cd /d "%~dp0"
call .venv_build\Scripts\activate.bat
python gui.py
pause