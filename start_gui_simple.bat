@echo off
REM ============================================================================
REM Simple launcher - Edit the variables below for your setup
REM ============================================================================

REM YOUR SETTINGS - EDIT THESE
set ENV_NAME=pyonline
set SCRIPT_NAME=RealTimeGUIv4t.py

REM ============================================================================

echo Starting Real-time Neural Analysis System...
echo Environment: %ENV_NAME%
echo.

call conda activate %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate environment: %ENV_NAME%
    echo.
    echo Make sure:
    echo 1. Conda is installed and in PATH
    echo 2. Environment exists: conda env list
    pause
    exit /b 1
)

if not exist "%SCRIPT_NAME%" (
    echo ERROR: Could not find %SCRIPT_NAME%
    echo Current directory: %CD%
    pause
    exit /b 1
)

python "%SCRIPT_NAME%"

echo.
if %ERRORLEVEL% NEQ 0 (
    echo Application exited with error code: %ERRORLEVEL%
)
pause

