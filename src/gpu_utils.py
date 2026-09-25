"""
src/gpu_utils.py
================
Robust detection and smoke testing of Kaggle GPU environments for LightGBM.
"""

import logging
import subprocess
import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)


def check_nvidia_smi() -> bool:
    """Check if nvidia-smi is available on the system."""
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False


def get_nvidia_smi_output() -> str:
    """Return the output of nvidia-smi, or a fallback message if unavailable."""
    try:
        output = subprocess.check_output(["nvidia-smi"], encoding="utf-8", stderr=subprocess.STDOUT)
        return output
    except Exception:
        return "nvidia-smi is unavailable."


def test_lightgbm_gpu_backend(backend: str) -> bool:
    """
    Run a tiny smoke test to verify if LightGBM can actually train on the given backend.
    backend can be 'gpu' or 'cuda'.
    """
    X = np.random.rand(50, 2)
    y = np.random.randint(0, 2, size=50)
    dtrain = lgb.Dataset(X, label=y)
    
    params = {
        "objective": "binary",
        "device_type": backend,
        "verbosity": -1,
        "num_leaves": 2,
        "n_estimators": 2,
    }
    
    try:
        # If the backend is unsupported or fails to initialize, this will raise a LightGBMError
        lgb.train(params, dtrain)
        return True
    except Exception as e:
        logger.debug(f"LightGBM backend '{backend}' smoke test failed: {e}")
        return False


def detect_gpu_backend() -> str:
    """
    Detects the best available LightGBM GPU backend.
    Returns 'cuda', 'gpu', or 'cpu'.
    """
    if not check_nvidia_smi():
        logger.info("nvidia-smi not found. Falling back to CPU.")
        return "cpu"
        
    # Try 'cuda' backend first (usually faster if compiled)
    if test_lightgbm_gpu_backend("cuda"):
        return "cuda"
    # Fallback to OpenCL 'gpu' backend
    elif test_lightgbm_gpu_backend("gpu"):
        return "gpu"
    else:
        return "cpu"


def resolve_device_type(config_use_gpu: bool, config_backend: str) -> str:
    """
    Resolve the actual device_type parameter for LightGBM based on config and system capabilities.
    """
    if not config_use_gpu:
        return "cpu"
        
    if config_backend == "auto":
        actual = detect_gpu_backend()
        if actual == "cpu":
            logger.warning("GPU was requested (auto), but no working LightGBM GPU backend was found. Falling back to CPU.")
        return actual
        
    if config_backend in ["cuda", "gpu"]:
        if test_lightgbm_gpu_backend(config_backend):
            return config_backend
        else:
            raise RuntimeError(f"Explicitly requested GPU backend '{config_backend}' is not functional in this environment.")
            
    return "cpu"
