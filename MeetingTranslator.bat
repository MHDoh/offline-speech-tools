@echo off
rem Meeting Translator. No arguments: the window. With arguments: see README, e.g.
rem   MeetingTranslator.bat --prompt "Baris Bey, Bahar Hanim, Acme"
rem   MeetingTranslator.bat --file recording.mp3
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
if "%~1"=="" (
    start "" ".venv\Scripts\pythonw.exe" translator.py
    exit /b
)
".venv\Scripts\python.exe" translator.py %*
