#!/bin/bash
# One-time environment setup on a Longleaf login node, run from the project root.
# Installs the exact package versions used for the benchmark (requirements.txt).
# Module names change over time: check with `module avail python` and edit below.
set -euo pipefail
module purge
module load python/3.12.4
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import pyblp, pandas; print('pyblp', pyblp.__version__, 'pandas', pandas.__version__)"
