@echo off
REM ML-TSSP HUMINT Dashboard - Windows Run Script

echo.
echo ========================================
echo  Starting ML-TSSP HUMINT Dashboard
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if dashboard.py exists
if not exist "dashboard.py" (
    echo ERROR: dashboard.py not found!
    echo Make sure you're in the correct directory
    pause
    exit /b 1
)

REM Run the dashboard
echo.
echo Launching dashboard on http://localhost:8501
echo Press Ctrl+C to stop
echo.
streamlit run dashboard.py
