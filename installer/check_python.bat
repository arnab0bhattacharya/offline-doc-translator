@echo off
REM installer/check_python.bat
REM ===========================
REM Checks if Python 3.14+ is installed and accessible.
REM Returns exit code 0 if found, 1 if not found.

python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH.
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v

REM Extract major.minor
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

if %PYMAJOR% GEQ 3 (
    if %PYMINOR% GEQ 14 (
        echo Python %PYVER% found.
        exit /b 0
    )
)

echo Python %PYVER% found but version 3.14+ required.
exit /b 1
