#!/bin/bash

# 1. Create virtual environment using Python 3.10
python3.10 -m venv .venv

# 2. Activate the venv
source .venv/bin/activate

# 3. Upgrade pip and install packages
pip install --upgrade pip
pip install numpy pandas tensorflow coremltools

echo "✅ Environment setup complete. Run 'source .venv/bin/activate' to activate."