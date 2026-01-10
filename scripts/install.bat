@echo off
REM PolyglotLink Installer for Windows
REM
REM Usage:
REM   install.bat [options]
REM
REM Options:
REM   --dev       Install development dependencies
REM   --docker    Set up Docker environment
REM   --uninstall Remove PolyglotLink
REM

setlocal enabledelayedexpansion

REM Configuration
set "INSTALL_DIR=%USERPROFILE%\.polyglotlink"
set "CONFIG_DIR=%USERPROFILE%\.config\polyglotlink"
set "MIN_PYTHON_VERSION=3.10"

REM Parse arguments
set "DEV_INSTALL=false"
set "DOCKER_SETUP=false"
set "UNINSTALL=false"

:parse_args
if "%~1"=="" goto :main
if "%~1"=="--dev" set "DEV_INSTALL=true"
if "%~1"=="--docker" set "DOCKER_SETUP=true"
if "%~1"=="--uninstall" set "UNINSTALL=true"
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
shift
goto :parse_args

:show_help
echo PolyglotLink Installer for Windows
echo.
echo Usage: install.bat [options]
echo.
echo Options:
echo   --dev       Install development dependencies
echo   --docker    Set up Docker environment
echo   --uninstall Remove PolyglotLink
echo   --help      Show this help message
exit /b 0

:main
cls
echo.
echo   ____       _             _       _   _     _       _
echo  ^|  _ \ ___ ^| ^|_   _  __ _^| ^| ___ ^| ^|_^| ^|   (_)_ __ ^| ^| __
echo  ^| ^|_) / _ \^| ^| ^| ^| ^|/ _` ^| ^|/ _ \^| __^| ^|   ^| ^| '_ \^| ^|/ /
echo  ^|  __/ (_) ^| ^| ^|_^| ^| (_^| ^| ^| (_) ^| ^|_^| ^|___^| ^| ^| ^| ^|   ^<
echo  ^|_^|   \___/^|_^|\__, ^|\__, ^|_^|\___/ \__^|_____^|_^|_^| ^|_^|_^|\_\
echo                ^|___/ ^|___/
echo.
echo  Semantic API Translator for IoT Device Ecosystems
echo.

if "%UNINSTALL%"=="true" goto :uninstall

REM Check Python
echo [INFO] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is required but not installed.
    echo [INFO] Download Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%v"
echo [SUCCESS] Python %PYTHON_VERSION% found

REM Check pip
echo [INFO] Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is required but not installed.
    pause
    exit /b 1
)
echo [SUCCESS] pip found

REM Create directories
echo [INFO] Creating directories...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"
echo [SUCCESS] Directories created

REM Create virtual environment
echo [INFO] Creating virtual environment...
python -m venv "%INSTALL_DIR%\venv"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment created

REM Activate and install
echo [INFO] Installing PolyglotLink...
call "%INSTALL_DIR%\venv\Scripts\activate.bat"

pip install --upgrade pip >nul 2>&1

if "%DEV_INSTALL%"=="true" (
    pip install polyglotlink[dev,test]
) else (
    pip install polyglotlink
)

if errorlevel 1 (
    echo [ERROR] Failed to install PolyglotLink
    pause
    exit /b 1
)
echo [SUCCESS] PolyglotLink installed

REM Create launcher batch file
echo [INFO] Creating launcher...
(
echo @echo off
echo call "%INSTALL_DIR%\venv\Scripts\activate.bat"
echo python -m polyglotlink.app.main %%*
) > "%INSTALL_DIR%\polyglotlink.bat"
echo [SUCCESS] Launcher created

REM Create default config
echo [INFO] Creating default configuration...
if not exist "%CONFIG_DIR%\config.yaml" (
    (
    echo # PolyglotLink Configuration
    echo.
    echo environment: development
    echo.
    echo logging:
    echo   level: INFO
    echo   format: console
    echo.
    echo http:
    echo   enabled: true
    echo   host: "127.0.0.1"
    echo   port: 8080
    echo.
    echo mqtt:
    echo   enabled: false
    echo   broker_host: localhost
    echo   broker_port: 1883
    ) > "%CONFIG_DIR%\config.yaml"
    echo [SUCCESS] Configuration created
) else (
    echo [INFO] Configuration already exists
)

REM Create .env file
echo [INFO] Creating environment file...
if not exist "%CONFIG_DIR%\.env" (
    (
    echo # PolyglotLink Environment Variables
    echo.
    echo # Required for semantic translation
    echo OPENAI_API_KEY=
    echo.
    echo # Environment
    echo POLYGLOTLINK_ENV=development
    echo.
    echo # Logging
    echo LOG_LEVEL=INFO
    ) > "%CONFIG_DIR%\.env"
    echo [SUCCESS] Environment file created
    echo [WARNING] Please edit %CONFIG_DIR%\.env and add your API keys
) else (
    echo [INFO] Environment file already exists
)

REM Add to PATH
echo [INFO] Adding to PATH...
setx PATH "%PATH%;%INSTALL_DIR%" >nul 2>&1
echo [SUCCESS] Added to PATH

REM Docker setup
if "%DOCKER_SETUP%"=="true" (
    echo [INFO] Setting up Docker...
    docker --version >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Docker not found. Skipping Docker setup.
    ) else (
        docker pull ghcr.io/polyglotlink/polyglotlink:latest
        echo [SUCCESS] Docker image pulled
    )
)

REM Verify installation
echo [INFO] Verifying installation...
call "%INSTALL_DIR%\polyglotlink.bat" version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Could not verify installation
) else (
    echo [SUCCESS] Installation verified
)

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo   1. Open a new command prompt
echo.
echo   2. Set your API key:
echo      set OPENAI_API_KEY=your-key-here
echo.
echo   3. Start the server:
echo      polyglotlink serve
echo.
echo   4. Check health:
echo      curl http://localhost:8080/health
echo.
echo Documentation: https://polyglotlink.io/docs
echo.
pause
exit /b 0

:uninstall
echo [INFO] Uninstalling PolyglotLink...

if exist "%INSTALL_DIR%" (
    rmdir /s /q "%INSTALL_DIR%"
    echo [SUCCESS] Installation directory removed
)

echo [SUCCESS] PolyglotLink uninstalled
echo [INFO] Configuration preserved at %CONFIG_DIR%
echo [INFO] Remove manually if desired

pause
exit /b 0
