@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo   Obi — Observability Assistant — Windows Build Script
echo ============================================================
echo.

set PYTHON_VER=3.12.8
set PYTHON_EMBED_URL=https://www.python.org/ftp/python/%PYTHON_VER%/python-%PYTHON_VER%-embed-amd64.zip
set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip

REM ── Step 1: Check Python is available ──────────────────────────────────────
echo [1/7] Checking Python ...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.12+ and add to PATH.
    pause
    exit /b 1
)
python --version
echo.

REM ── Step 2: Create virtual environment ─────────────────────────────────────
echo [2/7] Setting up virtual environment ...
if not exist "build_venv" (
    python -m venv build_venv
)
call build_venv\Scripts\activate.bat

REM ── Step 3: Install build dependencies ─────────────────────────────────────
echo [3/7] Installing dependencies ...
pip install --upgrade pip
pip install -r requirements_build.txt
echo.

REM ── Step 4: Run PyInstaller ────────────────────────────────────────────────
echo [4/7] Running PyInstaller ...
pyinstaller --clean --noconfirm obi.spec
if errorlevel 1 (
    echo ERROR: PyInstaller failed.
    pause
    exit /b 1
)
echo.

REM ── Step 5: Download embedded Python for Wav2Lip subprocess ────────────────
echo [5/7] Downloading embedded Python %PYTHON_VER% ...
set DIST_DIR=dist\Obi
set PYTHON_DIR=%DIST_DIR%\_python

if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
if not exist "_downloads" mkdir "_downloads"

if not exist "_downloads\python-embed.zip" (
    curl -L -o "_downloads\python-embed.zip" "%PYTHON_EMBED_URL%"
)
echo Extracting embedded Python ...
powershell -Command "Expand-Archive -Path '_downloads\python-embed.zip' -DestinationPath '%PYTHON_DIR%' -Force"

REM Install pip in embedded Python so Wav2Lip deps can work
curl -L -o "%PYTHON_DIR%\get-pip.py" https://bootstrap.pypa.io/get-pip.py
"%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location

REM Uncomment the import site line in python312._pth
powershell -Command "(Get-Content '%PYTHON_DIR%\python312._pth') -replace '#import site','import site' | Set-Content '%PYTHON_DIR%\python312._pth'"

REM Install Wav2Lip runtime deps into embedded Python
"%PYTHON_DIR%\python.exe" -m pip install --no-warn-script-location torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"%PYTHON_DIR%\python.exe" -m pip install --no-warn-script-location numpy opencv-python-headless librosa scipy batch-face
echo.

REM ── Step 6: Download FFmpeg ────────────────────────────────────────────────
echo [6/7] Downloading FFmpeg ...
if not exist "_downloads\ffmpeg.zip" (
    curl -L -o "_downloads\ffmpeg.zip" "%FFMPEG_URL%"
)
echo Extracting ffmpeg.exe ...
powershell -Command "$zip = [System.IO.Compression.ZipFile]::OpenRead('_downloads\ffmpeg.zip'); $entry = $zip.Entries | Where-Object { $_.Name -eq 'ffmpeg.exe' -and $_.FullName -like '*/bin/ffmpeg.exe' } | Select-Object -First 1; if ($entry) { [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, '%DIST_DIR%\ffmpeg.exe', $true) }; $zip.Dispose()"
echo.

REM ── Step 7: Copy data files ────────────────────────────────────────────────
echo [7/7] Copying data files ...
copy /Y avatar.png "%DIST_DIR%\avatar.png"

if not exist "%DIST_DIR%\Wav2Lip" mkdir "%DIST_DIR%\Wav2Lip"
xcopy /E /Y /Q Wav2Lip\*.py "%DIST_DIR%\Wav2Lip\"
if not exist "%DIST_DIR%\Wav2Lip\checkpoints" mkdir "%DIST_DIR%\Wav2Lip\checkpoints"
copy /Y Wav2Lip\checkpoints\wav2lip_gan.pth "%DIST_DIR%\Wav2Lip\checkpoints\"

xcopy /E /Y /Q Wav2Lip\models\* "%DIST_DIR%\Wav2Lip\models\"
xcopy /E /Y /Q Wav2Lip\face_detection\* "%DIST_DIR%\Wav2Lip\face_detection\"

if not exist "%DIST_DIR%\temp" mkdir "%DIST_DIR%\temp"

REM Copy cached greeting if available
if exist "temp\greeting_cached.wav" (
    copy /Y temp\greeting_cached.wav "%DIST_DIR%\temp\"
    copy /Y temp\greeting_cached.mp4 "%DIST_DIR%\temp\"
    echo Cached greeting included — first launch will be instant.
)

echo.
echo ============================================================
echo   BUILD COMPLETE
echo   Output: dist\Obi\Obi.exe
echo   Size:
dir /s "%DIST_DIR%" | find "File(s)"
echo ============================================================
echo.
echo To run: dist\Obi\Obi.exe
echo To distribute: zip the entire dist\Obi folder.
echo.
pause
