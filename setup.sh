#!/bin/bash
# Setup script for my_agent conda environment (Linux/Mac)
# Adjust CONDA_PATH if your anaconda/miniconda is installed elsewhere

CONDA_PATH="${HOME}/anaconda3"

echo "Creating conda environment 'my_agent'..."
"${CONDA_PATH}/bin/conda" env update -f environment.yml --prune

echo ""
echo "Environment created. Activate with:"
echo "  conda activate my_agent"
echo ""
echo "Then create/edit config/default.yaml with your API credentials."
echo ""
echo "Run experiments:"
echo "  python -m my_agent.main eval --dataset natural_questions --experiment naive_rag"
echo "  python -m my_agent.main eval --dataset natural_questions --experiment hyde"
echo ""
echo "Or test with LangGraph adapter:"
echo "  python -m my_agent.main graph --query \"Why is the sky blue?\""
