@echo off
rem One-time setup: Python packages into .venv, then the models (~4.4 GB).
rem   setup.bat               everything
rem   setup.bat voicetype     only what VoiceType needs (also: translator)
setlocal EnableExtensions
cd /d "%~dp0"
title Offline Speech Tools setup
echo.
echo  Offline Speech Tools - setup
echo  ============================
echo  Installs the Python packages into .venv and downloads the models.
echo.

rem --- 1. Python 3.12 (the GPU build of whisper.cpp is made for it) -----------------
set "PYEXE="
set "PYARGS="
py -3.12 -c "import sys" >nul 2>&1 && (set "PYEXE=py" & set "PYARGS=-3.12")
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE (
    echo Python 3.12 not found, installing it with winget...
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
if not defined PYEXE (
    echo.
    echo  Could not find or install Python 3.12.
    echo  Install it from https://www.python.org/downloads/ and run setup.bat again.
    goto :fail
)

rem --- 2. Python packages -------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv ...
    "%PYEXE%" %PYARGS% -m venv .venv || goto :fail
)
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip --quiet
echo Installing Python packages...
"%VPY%" -m pip install -r requirements.txt || goto :fail

rem whisper.cpp's Python binding, built with Vulkan so Whisper runs on AMD, NVIDIA and Intel GPUs
set "WHL=pywhispercpp-1.5.2.dev0+gf7bf62118.d20261005-cp312-cp312-win_amd64.whl"
echo Installing the GPU (Vulkan) build of pywhispercpp...
curl -fsSL -o "%TEMP%\%WHL%" "https://github.com/MHDoh/offline-speech-tools/releases/download/v1.0/pywhispercpp-vulkan-cp312-win_amd64.whl" && "%VPY%" -m pip install "%TEMP%\%WHL%"
if errorlevel 1 (
    echo GPU build not available, installing the CPU build instead.
    "%VPY%" -m pip install pywhispercpp || goto :fail
)
del "%TEMP%\%WHL%" 2>nul

rem --- 3. Models ----------------------------------------------------------------------
echo.
echo Downloading models (skips what is already there)...
"%VPY%" get_models.py %* || goto :fail

echo.
echo  Done. Start the apps with:
echo    MeetingTranslator.bat    live Turkish to English translation of what your speakers play
echo    VoiceType.bat            dictation into any text box: hold Alt + Copilot key
echo    Transcribe.bat           drop audio or video files in, get .txt and .srt
echo  VoiceType.bat autostart    starts VoiceType with Windows
echo.
pause
exit /b 0

:fail
echo.
echo  Setup failed, see the messages above. Running setup.bat again continues where it stopped.
pause
exit /b 1
