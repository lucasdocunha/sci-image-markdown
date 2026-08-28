#!/bin/bash
cd /home/lucas/sci-image-markdown
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
exec /home/lucas/miniconda3/envs/DeepLearning/bin/python -u train.py --config configs/default.yaml >> /home/lucas/sci-image-markdown/train.log 2>&1
