@echo off
rem Transcribe: drop audio/video files or folders on the window (or on this file); a .txt
rem (and optionally .srt) is written next to each one.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" transcribe\transcribe_gui.py %*
