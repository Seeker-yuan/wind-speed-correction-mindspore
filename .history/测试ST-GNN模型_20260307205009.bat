@echo off
chcp 65001 >nul
echo ======================================================================
echo 测试时空图神经网络（ST-GNN）模型
echo ======================================================================
echo.

cd /d "C:\Users\31876\Desktop\风能ui设计"

echo 使用Python环境: D:\Anaconda_envs\envs\mindspore\python.exe
echo.

"D:\Anaconda_envs\envs\mindspore\python.exe" mindspore_gnn_model.py

echo.
echo ======================================================================
echo 测试完成！
echo ======================================================================
pause
