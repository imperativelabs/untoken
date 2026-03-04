#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update -qq
sudo apt-get install -y python3 python3-venv python3-dev git wget curl build-essential

python3 -m venv ~/untoken-env
source ~/untoken-env/bin/activate

pip install --upgrade pip

pip install "numpy<2"
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

pip install \
    "transformers>=4.36.0" \
    "datasets>=2.16.0" \
    "sentence-transformers>=2.2.0" \
    "tokenizers>=0.15.0" \
    "accelerate>=0.25.0"

pip install \
    "wandb>=0.16.0" \
    "bert-score>=0.3.13" \
    "rouge-score>=0.1.2" \
    "scipy>=1.11.0" \
    "scikit-learn>=1.3.0" \
    "pyyaml>=6.0" \
    "tqdm>=4.66.0"

pip install "pytest>=7.4.0" "black>=23.0" "ruff>=0.1.0"

pip install "spacy>=3.7.0"
python -m spacy download en_core_web_sm

pip install -e ~/untoken

pip freeze > ~/untoken/requirements-frozen.txt
echo "Setup complete. Activate with: source ~/untoken-env/bin/activate"
