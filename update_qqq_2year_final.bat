@echo off
cd /d "%~dp0"
.\.venv\Scripts\python.exe scripts\update_qqq_2year_final.py --csv data\QQQ_2Year_Final.csv --symbol QQQ --period 8d --backup
