@echo off
rem VoiceType: runs in the tray; hold Alt + Copilot key and speak (change the key in Settings).
rem   VoiceType.bat               start it
rem   VoiceType.bat autostart     also start it with Windows
rem   VoiceType.bat noautostart   stop starting it with Windows
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
set "LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\VoiceType.lnk"
if /i "%~1"=="noautostart" (
    del "%LNK%" 2>nul
    echo VoiceType will no longer start with Windows.
    exit /b
)
if /i "%~1"=="autostart" (
    powershell -NoProfile -Command "$l = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK); $l.TargetPath = '%~dp0.venv\Scripts\pythonw.exe'; $l.Arguments = [char]34 + '%~dp0voicetype.py' + [char]34; $l.WorkingDirectory = '%~dp0'; $l.IconLocation = '%~dp0assets\voicetype.ico'; $l.Save()"
    echo VoiceType will start with Windows. Undo with: VoiceType.bat noautostart
)
start "" ".venv\Scripts\pythonw.exe" voicetype.py
