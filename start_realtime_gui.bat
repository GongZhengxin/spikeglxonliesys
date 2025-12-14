@echo off
REM ============================================================================
REM Real-time Neural Analysis System Launcher
REM Compatible with Conda/Miniconda installations on Windows
REM ============================================================================

setlocal EnableDelayedExpansion

echo ========================================
echo Real-time Neural Analysis System
echo ========================================
echo.

REM ============================================================================
REM Configuration - Modify these variables as needed
REM ============================================================================

REM Set your conda environment name here
set ENV_NAME=pyonline

REM Set the Python script name
set SCRIPT_NAME=RealTimeGUIv4JW.py

REM ============================================================================
REM Auto-detect Conda installation
REM ============================================================================

echo [1/4] Detecting Conda installation...

REM Method 1: Check if conda is in PATH
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Found conda in system PATH
    set "CONDA_FOUND=1"
    goto :activate_env
)

REM Method 2: Check common Anaconda installation paths
set "CONDA_PATHS=%USERPROFILE%\anaconda3 %USERPROFILE%\miniconda3 %USERPROFILE%\Anaconda3 %USERPROFILE%\Miniconda3 C:\ProgramData\anaconda3 C:\ProgramData\miniconda3 C:\ProgramData\Anaconda3 C:\ProgramData\Miniconda3 C:\Anaconda3 C:\Miniconda3"

for %%P in (%CONDA_PATHS%) do (
    if exist "%%P\Scripts\conda.exe" (
        echo [OK] Found conda at: %%P
        set "CONDA_ROOT=%%P"
        set "CONDA_FOUND=1"
        
        REM Initialize conda for this session
        call "%%P\Scripts\activate.bat" "%%P"
        if !ERRORLEVEL! EQU 0 (
            goto :activate_env
        )
    )
)

REM Method 3: Check if CONDA_PREFIX is set (already in conda environment)
if defined CONDA_PREFIX (
    echo [OK] Already in conda environment: %CONDA_PREFIX%
    set "CONDA_FOUND=1"
    goto :check_script
)

REM Method 4: Search in common drive letters
echo [INFO] Searching for conda in common locations...
for %%D in (C D E) do (
    if exist "%%D:\Anaconda3\Scripts\conda.exe" (
        echo [OK] Found conda at: %%D:\Anaconda3
        set "CONDA_ROOT=%%D:\Anaconda3"
        set "CONDA_FOUND=1"
        call "%%D:\Anaconda3\Scripts\activate.bat" "%%D:\Anaconda3"
        goto :activate_env
    )
    if exist "%%D:\Miniconda3\Scripts\conda.exe" (
        echo [OK] Found conda at: %%D:\Miniconda3
        set "CONDA_ROOT=%%D:\Miniconda3"
        set "CONDA_FOUND=1"
        call "%%D:\Miniconda3\Scripts\activate.bat" "%%D:\Miniconda3"
        goto :activate_env
    )
)

REM If conda not found
echo [ERROR] Could not find conda installation!
echo.
echo Please ensure that either:
echo   1. Conda is installed and added to PATH
echo   2. Conda is installed in a standard location
echo   3. You run this script from within a conda environment
echo.
echo Common installation paths checked:
echo   - %USERPROFILE%\anaconda3
echo   - %USERPROFILE%\miniconda3
echo   - C:\ProgramData\anaconda3
echo   - C:\Anaconda3
echo   - C:\Miniconda3
echo.
echo Press any key to exit...
pause >nul
exit /b 1

:activate_env
REM ============================================================================
REM Activate the conda environment
REM ============================================================================

echo.
echo [2/4] Activating conda environment: %ENV_NAME%...

REM Try to activate using conda command
call conda activate %ENV_NAME% 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Environment activated: %ENV_NAME%
    goto :check_script
)

REM If direct activation failed, try with full path
if defined CONDA_ROOT (
    call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% 2>nul
    if !ERRORLEVEL! EQU 0 (
        echo [OK] Environment activated: %ENV_NAME%
        goto :check_script
    )
)

echo [ERROR] Failed to activate environment: %ENV_NAME%
echo.
echo Please check:
echo   1. Environment name is correct (current: %ENV_NAME%)
echo   2. Environment exists (run: conda env list)
echo   3. Conda is properly initialized
echo.
echo To create the environment, run:
echo   conda create -n %ENV_NAME% python=3.10
echo.
echo Press any key to exit...
pause >nul
exit /b 1

:check_script
REM ============================================================================
REM Check if Python script exists
REM ============================================================================

echo.
echo [3/4] Checking for Python script...

if not exist "%SCRIPT_NAME%" (
    echo [ERROR] Could not find: %SCRIPT_NAME%
    echo.
    echo Current directory: %CD%
    echo.
    echo Please ensure you run this script from the correct directory.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo [OK] Found script: %SCRIPT_NAME%

REM ============================================================================
REM Display environment information
REM ============================================================================

echo.
echo [4/4] Environment Information:
echo ----------------------------------------

REM Check Python version
python --version 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in current environment!
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

REM Display environment location
where python
echo ----------------------------------------
echo.

REM ============================================================================
REM Run the Python script
REM ============================================================================

echo Starting Real-time Neural Analysis System...
echo.
echo ========================================
echo.

REM Run the script
python "%SCRIPT_NAME%"

REM Capture exit code
set EXIT_CODE=%ERRORLEVEL%

REM ============================================================================
REM Handle exit
REM ============================================================================

echo.
echo ========================================
echo.

if %EXIT_CODE% EQU 0 (
    echo [OK] Application exited normally
) else (
    echo [ERROR] Application exited with code: %EXIT_CODE%
    echo.
    echo Check the log messages above for details.
    echo.
    timeout /t 30 /nobreak >nul
)

REM Exit directly without pause
exit /b %EXIT_CODE%

