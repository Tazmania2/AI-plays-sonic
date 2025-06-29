@echo off
echo ========================================
echo Sonic AI Training Launcher
echo ========================================
echo.

echo [1/3] Running cleanup to ensure clean state...
python cleanup_processes.py
echo.

echo [2/3] Checking GPU availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', 'CUDA' if torch.cuda.is_available() else 'CPU')"
echo.

echo [3/3] Starting training...
echo.
echo Available commands:
echo   python train_sonic.py --help
echo   python train_sonic.py --reward_mode baseline --num_envs 4
echo   python train_sonic.py --reward_mode shaping --num_envs 4
echo   python train_sonic.py --reward_mode both --num_envs 8
echo.

set /p command="Enter training command (or press Enter for default): "
if "%command%"=="" (
    echo Starting default training (baseline mode, 4 environments)...
    python train_sonic.py --reward_mode baseline --num_envs 4
) else (
    echo Running: %command%
    %command%
)

echo.
echo Training completed. Running final cleanup...
python cleanup_processes.py
echo.
echo Done!
pause 