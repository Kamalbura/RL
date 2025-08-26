@echo off
echo Activating conda environment and running validation...
call conda activate rl_env
if errorlevel 1 (
    echo Failed to activate conda environment
    echo Trying with base environment...
    python test_basic.py
) else (
    echo Conda environment activated successfully
    python test_basic.py
)
pause
