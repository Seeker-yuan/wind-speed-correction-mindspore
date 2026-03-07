@echo off
chcp 65001 >nul
echo ======================================================================
echo 使用真实wind_data数据训练时空图神经网络（ST-GNN）
echo ======================================================================
echo.

cd /d "C:\Users\31876\Desktop\风能ui设计"

echo 使用Python环境: D:\Anaconda_envs\envs\mindspore\python.exe
echo.

"D:\Anaconda_envs\envs\mindspore\python.exe" 训练真实数据.py

echo.
echo ======================================================================
echo 训练完成！
echo ======================================================================
pause
