@echo off
REM Launch_App.bat
REM ==============================================
REM Convenient launcher for Offline Document Translator
REM Runs the standalone application from dist\
REM ==============================================

if not exist "%~dp0dist\OfflineTranslator\OfflineTranslator.exe" (
    echo [!] Application executable not found in dist\OfflineTranslator\
    echo [*] Please run: python build.py
    pause
    exit /b 1
)

start "" "%~dp0dist\OfflineTranslator\OfflineTranslator.exe"
