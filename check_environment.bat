@echo off
REM ============================================================================
REM Environment Diagnostic Tool
REM Check conda installation and Python environment setup
REM ============================================================================

setlocal EnableDelayedExpansion

echo ========================================
echo Environment Diagnostic Tool
echo ========================================
echo.

REM Set environment name to check
set ENV_NAME=pyonline

REM ============================================================================
echo [1] System Information
echo ----------------------------------------

echo Windows Version:
ver

echo.
echo Current Directory:
cd

echo.
echo User: %USERNAME%
echo Computer: %COMPUTERNAME%
echo.

REM ============================================================================
echo [2] Conda Detection
echo ----------------------------------------

echo Checking if conda is in PATH...
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Conda found in PATH
    where conda
    echo.
    
    echo Conda version:
    conda --version
    echo.
    
    echo Conda info:
    conda info
    echo.
) else (
    echo [WARNING] Conda not found in PATH
    echo.
)

echo Checking common installation locations...
echo.

set "FOUND=0"

for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "C:\ProgramData\anaconda3" "C:\Anaconda3" "C:\Miniconda3") do (
    if exist "%%~P\Scripts\conda.exe" (
        echo [FOUND] %%~P
        set "FOUND=1"
    )
)

if !FOUND! EQU 0 (
    echo [INFO] No conda installation found in standard locations
)

echo.

REM ============================================================================
echo [3] Conda Environments
echo ----------------------------------------

conda env list 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Could not list conda environments
    echo Conda may not be properly initialized
) else (
    echo.
    echo Checking if environment "%ENV_NAME%" exists...
    conda env list | findstr /i "%ENV_NAME%" >nul
    if !ERRORLEVEL! EQU 0 (
        echo [OK] Environment "%ENV_NAME%" found
    ) else (
        echo [WARNING] Environment "%ENV_NAME%" not found
        echo.
        echo To create it, run:
        echo   conda create -n %ENV_NAME% python=3.10 -y
    )
)

echo.

REM ============================================================================
echo [4] Python Installation
echo ----------------------------------------

echo Current Python:
where python 2>nul
if %ERRORLEVEL% EQU 0 (
    python --version
    echo.
    
    echo Python executable location:
    python -c "import sys; print(sys.executable)"
    echo.
    
    echo Python path:
    python -c "import sys; print('\n'.join(sys.path))"
    echo.
) else (
    echo [WARNING] Python not found in current environment
    echo.
)

REM ============================================================================
echo [5] Required Packages Check
echo ----------------------------------------

echo Checking Python packages...
echo.

set "PACKAGES=PyQt5 pyqtgraph numpy scipy pandas yaml psutil"

for %%P in (%PACKAGES%) do (
    python -c "import %%P" 2>nul
    if !ERRORLEVEL! EQU 0 (
        echo [OK] %%P installed
        python -c "import %%P; print('    Version:', %%P.__version__)" 2>nul
    ) else (
        echo [MISSING] %%P not installed
    )
)

echo.

REM ============================================================================
echo [6] Script Files Check
echo ----------------------------------------

if exist "RealTimeGUIv4t.py" (
    echo [OK] RealTimeGUIv4t.py found
    
    echo File size:
    for %%F in (RealTimeGUIv4t.py) do echo     %%~zF bytes
    
    echo Last modified:
    for %%F in (RealTimeGUIv4t.py) do echo     %%~tF
) else (
    echo [WARNING] RealTimeGUIv4t.py not found in current directory
)

echo.

if exist "config.yaml" (
    echo [OK] config.yaml found
) else (
    echo [INFO] config.yaml not found (will be created on first run)
)

echo.

REM ============================================================================
echo [7] Recommendations
echo ----------------------------------------

set "ISSUES=0"

REM Check conda
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Conda not in PATH - Add conda to system PATH
    set /a ISSUES+=1
)

REM Check environment
if defined CONDA_PREFIX (
    echo [OK] Currently in conda environment: %CONDA_PREFIX%
) else (
    conda env list 2>nul | findstr /i "%ENV_NAME%" >nul
    if !ERRORLEVEL! EQU 0 (
        echo [!] Environment exists but not activated
        echo     Run: conda activate %ENV_NAME%
        set /a ISSUES+=1
    ) else (
        echo [!] Environment "%ENV_NAME%" does not exist
        echo     Run: conda create -n %ENV_NAME% python=3.10 -y
        set /a ISSUES+=1
    )
)

REM Check Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Python not found - Ensure Python is installed
    set /a ISSUES+=1
)

REM Check script
if not exist "RealTimeGUIv4t.py" (
    echo [!] Python script not found - Run this from the correct directory
    set /a ISSUES+=1
)

if !ISSUES! EQU 0 (
    echo.
    echo [SUCCESS] No issues detected!
    echo Your environment appears to be properly configured.
    echo.
    echo To start the application, run:
    echo   start_realtime_gui.bat
) else (
    echo.
    echo [WARNING] !ISSUES! issue(s) detected
    echo Please address the issues above before running the application.
)

echo.

REM ============================================================================
echo [8] Quick Setup Commands
echo ----------------------------------------
echo.
echo If you need to set up from scratch:
echo.
echo 1. Create environment:
echo    conda create -n %ENV_NAME% python=3.10 -y
echo.
echo 2. Activate environment:
echo    conda activate %ENV_NAME%
echo.
echo 3. Install packages:
echo    pip install PyQt5 pyqtgraph numpy scipy pandas pyyaml psutil
echo.
echo 4. Run application:
echo    python RealTimeGUIv4t.py
echo.

REM ============================================================================
echo ========================================
echo Diagnostic Complete
echo ========================================
echo.

pause

