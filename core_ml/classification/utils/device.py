import os
import torch

def get_device(device_override: str | None = None) -> torch.device:
    """
    Global device selector for PyTorch components.
    Priority order:
      1. Explicit argument `device_override` (if passed)
      2. Environment variable `OCT_LOCAL_DEVICE` ('cpu', 'mps', 'cuda', 'auto')
      3. Default fallback: 'auto' (checks CUDA -> MPS -> CPU)
    """
    target = (device_override or os.getenv("OCT_LOCAL_DEVICE", "auto")).strip().lower()
    
    if target in ("cpu", "cuda", "mps"):
        return torch.device(target)
    elif target == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    return torch.device("cpu")

