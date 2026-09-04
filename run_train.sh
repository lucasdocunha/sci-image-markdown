#!/bin/bash
cd /home/lucas/masters/sci-image-markdown
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
exec .venv/bin/python -u train.py --config configs/default.yaml >> train.log 2>&1
