#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Rebuild diff-triangle-rasterization
cd mesh_splatting
bash compile.sh
cd ..
