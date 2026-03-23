@echo off
cd /d "%~dp0"
set "PYTHONPATH=%CD%"
".venv\Scripts\python.exe" scripts\run_morning.py --config config.json
