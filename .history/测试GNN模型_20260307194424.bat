@echo off
echo ================================================
echo   图神经网络（GNN）模型测试
echo ================================================
echo.

REM 查找Anaconda安装路径
set ANACONDA_PATH=
if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat" set ANACONDA_PATH=C:\ProgramData\Anaconda3
if exist "C:\Users\%USERNAME%\Anaconda3\Scripts\activate.bat" set ANACONDA_PATH=C:\Users\%USERNAME%\Anaconda3
if exist "D:\Anaconda3\Scripts\activate.bat" set ANACONDA_PATH=D:\Anaconda3
if exist "C:\Anaconda3\Scripts\activate.bat" set ANACONDA_PATH=C:\Anaconda3

if "%ANACONDA_PATH%"=="" (
    echo [错误] 未找到Anaconda
    pause
    exit /b 1
)

call "%ANACONDA_PATH%\Scripts\activate.bat" mindspore
cd /d "C:\Users\31876\Desktop\风能ui设计"

echo [→] 安装scipy依赖（图神经网络需要）...
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple -q

echo.
echo [→] 运行图神经网络测试...
echo.
python mindspore_gnn_model.py

echo.
echo ================================================
pause
