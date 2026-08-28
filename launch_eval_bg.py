import subprocess
import os
import sys

log_path = "/home/lucas/sci-image-markdown/eval.log"
with open(log_path, "w") as f:
    proc = subprocess.Popen(
        [
            "/bin/bash",
            "/home/lucas/sci-image-markdown/run_eval_both.sh"
        ],
        cwd="/home/lucas/sci-image-markdown",
        env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1"),
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    with open("/home/lucas/sci-image-markdown/eval.pid", "w") as pf:
        pf.write(str(proc.pid))
    print(f"Spawned detached background process PID: {proc.pid}")
