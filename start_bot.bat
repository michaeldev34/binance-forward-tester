@echo off
REM Automated Trading Bot Startup Script for Windows
REM This script starts the complete automated trading bot

echo.
echo ========================================
echo   AUTOMATED TRADING BOT LAUNCHER
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "run_bot.py" (
    echo ERROR: run_bot.py not found
    echo Please make sure you're in the correct directory
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    echo Please make sure you're in the correct directory
    pause
    exit /b 1
)

echo Checking Python dependencies...
pip install -r requirements.txt --quiet

if errorlevel 1 (
    echo.
    echo WARNING: Some dependencies might not be installed correctly
    echo The bot may not work properly
    echo.
)

echo.
echo Starting Automated Trading Bot...
echo.
echo IMPORTANT NOTES:
echo - This bot will trade on Binance TESTNET (no real money)
echo - The bot runs autonomously and makes decisions automatically
echo - Press Ctrl+C to stop the bot gracefully
echo - All results will be saved in the 'bot_results' directory
echo.

REM Start the bot
python run_bot.py

echo.
echo Bot has stopped.
pause
