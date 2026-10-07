@echo off
rem Starts the presentation demo at http://localhost:8501 (models load on first Generate, ~15 s).
rem Settings (file watcher off) live in .streamlit\config.toml.
cd /d "%~dp0"
.venv\Scripts\python -m streamlit run demo\app.py
