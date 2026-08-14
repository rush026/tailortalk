#!/usr/bin/env bash
# Render build script — installs deps and pre-downloads ML models

set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Pre-download DINOv2 model weights so they're cached
python -c "
from transformers import AutoImageProcessor, AutoModel
AutoImageProcessor.from_pretrained('facebook/dinov2-base')
AutoModel.from_pretrained('facebook/dinov2-base')
print('DINOv2 model cached.')
"

# Pre-download rembg U2-Net model
python -c "
from rembg import new_session
new_session('u2net')
print('rembg U2-Net model cached.')
"

echo "Build complete."
