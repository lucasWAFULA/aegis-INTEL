@echo off
REM ML-TSSP HUMINT Dashboard - Windows Setup Script

echo.
echo ========================================
echo  ML-TSSP HUMINT Dashboard Setup
echo ========================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Create virtual environment
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created successfully
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create .env if it doesn't exist
if not exist ".env" (
    if exist ".env.example" (
        echo.
        echo Creating .env file from .env.example...
        copy .env.example .env
        echo WARNING: Please edit .env file with your configuration
    )
)

REM Check for required assets
echo.
echo Checking for required assets...
if not exist "Aegis-INTEL.png" (
    echo WARNING: Aegis-INTEL.png logo not found
    echo Please add your logo to the project root
)

echo.
echo ========================================
echo  Setup complete!
echo ========================================
echo.
echo To start the dashboard, run: run.bat
echo Or manually: 
echo   venv\Scripts\activate
echo   streamlit run dashboard.py
echo.
pause
