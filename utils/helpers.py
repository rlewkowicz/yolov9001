import os
import yaml
from pathlib import Path
from contextlib import contextmanager

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def load_yaml(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@contextmanager
def suppress_stderr_fd():
    """
    Context manager that silences C/C++ level writes to stderr by redirecting
    file descriptor 2 to os.devnull. This catches absl/glog output emitted
    before Python-level logging can be configured.
    """
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        yield
    finally:
        try:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
        except Exception:
            pass

def import_tensorflow_silently():
    """
    Import TensorFlow with stderr suppressed and minimal logging.
    Returns the tensorflow module or None if import fails.

    vulture: ignore[unused-function] — optional TF interop kept for future export paths.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("GLOG_minloglevel", "2")

    try:
        with suppress_stderr_fd():
            import tensorflow as tf  # type: ignore
        try:
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass
        return tf
    except Exception:
        return None
