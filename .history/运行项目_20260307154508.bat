@echo off
echo ================================================
echo   风速误差校正系统 - MindSpore版本
echo ================================================
echo.
echo 正在启动，请稍候...
echo.

REM 查找Anaconda安装路径
set ANACONDA_PATH=
if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat" set ANACONDA_PATH=C:\ProgramData\Anaconda3
if exist "C:\Users\%USERNAME%\Anaconda3\Scripts\activate.bat" set ANACONDA_PATH=C:\Users\%USERNAME%\Anaconda3
if exist "D:\Anaconda3\Scripts\activate.bat" set ANACONDA_PATH=D:\Anaconda3
if exist "C:\Anaconda3\Scripts\activate.bat" set ANACONDA_PATH=C:\Anaconda3

if "%ANACONDA_PATH%"=="" (
    echo [错误] 未找到Anaconda
    echo 请在Anaconda Prompt中手动运行：
    echo   cd "C:\Users\31876\Desktop\风能ui设计"
    echo   conda activate mindspore
    echo   python 汇总预测_mindspore版.py
    pause
    exit /b 1
)

REM 激活conda环境并运行
call "%ANACONDA_PATH%\Scripts\activate.bat" mindspore
cd /d "C:\Users\31876\Desktop\风能ui设计"

echo.
echo ================================================
echo 开始处理 104 台风机数据...
echo ================================================
echo.

python 汇总预测_mindspore版.py

echo.
echo ================================================
echo   处理完成！
echo ================================================
echo.
echo 输出文件位置：
echo   - cleaned_data\ (补全后的数据)
echo   - 缺损率报告_neural.xlsx
echo.
pause
