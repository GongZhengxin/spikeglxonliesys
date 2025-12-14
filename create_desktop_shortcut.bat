@echo off
REM ============================================================================
REM Create Desktop Shortcut for Real-time Neural Analysis System
REM ============================================================================

echo Creating desktop shortcut...
echo.

REM Get current directory (where the bat file is located)
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Set target bat file
set "TARGET_BAT=%SCRIPT_DIR%\start_realtime_gui.bat"

REM Check if target exists
if not exist "%TARGET_BAT%" (
    echo ERROR: Could not find start_realtime_gui.bat
    echo Expected location: %TARGET_BAT%
    pause
    exit /b 1
)

REM Get desktop path
set "DESKTOP=%USERPROFILE%\Desktop"

REM Set shortcut name
set "SHORTCUT_NAME=Real-time Neural Analysis.lnk"

REM Check for custom icon
set "ICON_FILE=%SCRIPT_DIR%\Icon.ico"
if exist "%ICON_FILE%" (
    echo Found custom icon: Icon.ico
    set "ICON_PATH=%ICON_FILE%"
) else (
    echo Using default system icon
    set "ICON_PATH=%SystemRoot%\System32\shell32.dll,165"
)

REM Create VBS script to create shortcut
set "VBS_SCRIPT=%TEMP%\create_shortcut.vbs"

echo Set oWS = WScript.CreateObject("WScript.Shell") > "%VBS_SCRIPT%"
echo sLinkFile = "%DESKTOP%\%SHORTCUT_NAME%" >> "%VBS_SCRIPT%"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%VBS_SCRIPT%"
echo oLink.TargetPath = "%TARGET_BAT%" >> "%VBS_SCRIPT%"
echo oLink.WorkingDirectory = "%SCRIPT_DIR%" >> "%VBS_SCRIPT%"
echo oLink.Description = "Launch Real-time Neural Analysis System" >> "%VBS_SCRIPT%"
echo oLink.IconLocation = "%ICON_PATH%" >> "%VBS_SCRIPT%"
echo oLink.Save >> "%VBS_SCRIPT%"

REM Execute VBS script
cscript //nologo "%VBS_SCRIPT%"

REM Clean up
del "%VBS_SCRIPT%"

if exist "%DESKTOP%\%SHORTCUT_NAME%" (
    echo.
    echo [SUCCESS] Shortcut created successfully!
    echo.
    echo Location: %DESKTOP%\%SHORTCUT_NAME%
    echo Target: %TARGET_BAT%
    echo Working Directory: %SCRIPT_DIR%
    if exist "%ICON_FILE%" (
        echo Icon: %ICON_FILE%
    ) else (
        echo Icon: System default
    )
    echo.
    echo You can now double-click the shortcut on your desktop to launch the application.
) else (
    echo.
    echo [ERROR] Failed to create shortcut
    echo Please try creating it manually.
)

echo.
pause

