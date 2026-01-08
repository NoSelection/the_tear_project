@echo off
setlocal
echo ==================================================
echo   THE TEAR - ONE-CLICK SETUP AND TRAIN
echo ==================================================

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python not found. Please install Python 3.10+ and add to PATH.
    pause
    exit /b
)

:: Create Virtual Environment
if not exist "venv" (
    echo [*] Creating virtual environment...
    python -m venv venv
)

:: Activate and Install
echo [*] Activating environment and installing dependencies...
call venv\Scripts\activate

:: Install Torch with CUDA 12.1 (This is the big one, ~2.5GB)
echo [*] Installing PyTorch (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo [!] Failed to install PyTorch. Check your internet connection.
    pause
    exit /b
)

:: Install other requirements
echo [*] Installing training libraries (Transformers, PEFT, etc.)...
pip install transformers>=4.40.0 peft bitsandbytes accelerate datasets

:: Run Training
echo ==================================================
echo   SETUP COMPLETE - STARTING TRAINING
echo ==================================================
python src/train.py

echo ==================================================
echo   TRAINING FINISHED
echo ==================================================
pause