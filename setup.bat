@echo off
REM Setup script for my_agent conda environment
REM Run from Anaconda Prompt: just run "setup.bat" in this directory

echo Creating/updating conda environment 'my_agent'...
conda env update -f environment.yml --prune --solver=libmamba

echo.
echo Done. Activate with:
echo   conda activate my_agent
echo.
echo Then edit config/default.yaml with your API credentials.
echo.
echo Run experiments:
echo   python -m my_agent.main eval --dataset natural_questions --experiment naive_rag
echo   python -m my_agent.main eval --dataset natural_questions --experiment hyde
echo.
echo Or test with LangGraph adapter:
echo   python -m my_agent.main graph --query "Why is the sky blue?"
