import subprocess
import os
import sys

log_path = "/home/lucas/masters/sci-image-markdown/train.log"
with open(log_path, "a") as f:
    proc = subprocess.Popen(
        [
            "/home/lucas/masters/sci-image-markdown/.venv/bin/python",
            "-u",
            "/home/lucas/masters/sci-image-markdown/train.py",
            "--config",
            "/home/lucas/masters/sci-image-markdown/configs/default.yaml"
        ],
        cwd="/home/lucas/masters/sci-image-markdown",
        env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1"),
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    print(f"Spawned detached background process PID: {proc.pid}")
