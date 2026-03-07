@echo off
chcp 65001 >nul
echo ======================================================================
echo 使用真实wind_data数据训练时空图神经网络（ST-GNN）
echo ======================================================================
echo.

call conda activate mindspore
if %errorlevel% neq 0 (
    echo ✗ 无法激活mindspore环境
    pause
    exit /b 1
)

echo ✓ 已激活mindspore环境
echo.

cd /d "C:\Users\31876\Desktop\风能ui设计"

python 训练真实数据.py

echo.
echo ======================================================================
echo 训练完成！
echo ======================================================================
pause
