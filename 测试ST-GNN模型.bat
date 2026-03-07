@echo off
chcp 65001 >nul
echo ======================================================================
echo 测试时空图神经网络（ST-GNN）模型
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

python mindspore_gnn_model.py

echo.
echo ======================================================================
echo 测试完成！
echo ======================================================================
pause
