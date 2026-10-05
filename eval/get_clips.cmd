@echo off
REM Downloads the two test clips used in eval\README.md (needs: .venv\Scripts\python -m pip install yt-dlp; ffmpeg on PATH).
REM   tedx : "Iletisiminiz Kadarsiniz" (TEDxIstanbul, 17.5 min), human Turkish + English subtitles
REM   alp  : "Alparslan Confronts Kekaumenos" (TRT, 5.4 min drama clip). Its English subtitles are
REM          burned into the video; refs\alp.en.vtt is a transcription of them.
cd /d "%~dp0"
if not exist clips mkdir clips
cd clips
..\..\.venv\Scripts\python -m yt_dlp -f bestaudio --write-subs --sub-langs "en,tr" --sub-format vtt -o "tedx.%%(ext)s" "https://www.youtube.com/watch?v=34RsefZzs_Y"
..\..\.venv\Scripts\python -m yt_dlp -f bestaudio -o "alp.%%(ext)s" "https://www.youtube.com/watch?v=fP6S-67iMnY"
for %%c in (tedx alp) do for %%f in (%%c.webm %%c.m4a %%c.opus) do if exist %%f ffmpeg -v error -y -i %%f -ac 1 -ar 16000 %%c.wav
