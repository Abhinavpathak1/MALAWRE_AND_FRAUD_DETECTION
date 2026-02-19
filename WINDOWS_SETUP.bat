@echo off
REM ═══════════════════════════════════════════════════════════════════════════════
REM     🛡️ UNIFIED SECURITY AI PLATFORM - Windows Quick Setup 🛡️
REM ═══════════════════════════════════════════════════════════════════════════════

echo.
echo ╔═══════════════════════════════════════════════════════════════════╗
echo ║                                                                   ║
echo ║     🛡️  SECURITY AI PLATFORM - QUICK SETUP FOR WINDOWS 🛡️       ║
echo ║                                                                   ║
echo ╚═══════════════════════════════════════════════════════════════════╝
echo.

REM Check Python
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo ✅ Python found!
echo.

REM Install dependencies
echo [2/4] Installing dependencies...
echo This may take 2-5 minutes on first run...
echo.

python -m pip install --upgrade pip
python -m pip install streamlit scikit-learn numpy pandas joblib xgboost matplotlib seaborn

if errorlevel 1 (
    echo.
    echo ⚠️  Installation had some warnings, but may still work.
    echo    If you get errors later, try:
    echo    pip install -r requirements_unified.txt
    echo.
    pause
) else (
    echo.
    echo ✅ Dependencies installed successfully!
    echo.
)

REM Train malware models
echo [3/4] Training Malware Detection Models...
echo This takes 2-5 minutes...
echo.

if not exist "train_from_database.py" (
    echo ❌ ERROR: train_from_database.py not found!
    echo.
    echo Make sure you have ALL files from the download in this folder.
    pause
    exit /b 1
)

python train_from_database.py

if errorlevel 1 (
    echo.
    echo ⚠️  Malware model training had errors.
    echo    Check if full_training_database.csv is present.
    pause
) else (
    echo ✅ Malware models trained!
)

echo.

REM Train fraud models
echo [4/4] Training Fraud Detection Models...
echo This takes 1-3 minutes...
echo.

if not exist "train_fraud_models.py" (
    echo ⚠️  train_fraud_models.py not found - skipping fraud models
) else (
    python train_fraud_models.py
    if errorlevel 1 (
        echo ⚠️  Fraud model training had errors.
    ) else (
        echo ✅ Fraud models trained!
    )
)

echo.
echo ═══════════════════════════════════════════════════════════════════
echo   ✅ SETUP COMPLETE!
echo ═══════════════════════════════════════════════════════════════════
echo.
echo 🚀 Ready to launch! You can now run:
echo.
echo    python launch.py
echo.
echo    OR directly:
echo.
echo    streamlit run unified_security_platform.py
echo.
echo ═══════════════════════════════════════════════════════════════════
echo.
pause

REM Ask if they want to launch now
echo.
set /p launch="Would you like to launch the platform now? (Y/N): "
if /i "%launch%"=="Y" (
    echo.
    echo Launching Security AI Platform...
    echo Press Ctrl+C to stop when done.
    echo.
    python launch.py
)
