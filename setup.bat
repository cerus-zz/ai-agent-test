@echo off
REM Setup script for my_agent conda environment
REM Adjust CONDA_PATH if your conda is installed elsewhere

set CONDA_PATH=D:\XApplication\Scripts

echo Creating conda environment 'my_agent'...
call "%CONDA_PATH%\conda.exe" env create -f environment.yml

echo.
echo Environment created. Activate with:
echo   D:\XApplication\Scripts\activate my_agent
echo.
echo Then create/edit config/default.yaml with your API credentials.
echo.
echo Run experiments:
echo   python -m my_agent.main eval --dataset natural_questions --experiment naive_rag
echo   python -m my_agent.main eval --dataset natural_questions --experiment hyde
echo.
echo Or test with LangGraph adapter:
echo   python -m my_agent.main graph --query "Why is the sky blue?"

pause
