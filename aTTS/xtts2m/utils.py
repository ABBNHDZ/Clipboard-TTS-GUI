"""
xtts2m.utils — shared utilities

Utilities for device detection, audio I/O, and text processing.
Centralizes common tasks to keep model and manager code clean.
Exposes resolve_device for normalizing/validating device strings.

Key features:
- Device auto-detection (cuda/mps/privateuseone/cpu)
- Safe dtype selection for mixed precision
- Memory usage monitoring
"""

import os
import psutil
import logging
import contextlib
import importlib
import importlib.util

import torch

logger = logging.getLogger(__name__)

# Constants
SUPPORTED_FP16_GPUS = [
    "RTX 30", "RTX 40", "RTX 50", "RTX 60", "NVIDIA A100", "UHD Graphics 770",
    "AMD Radeon R7", "RX 6", "RX 7", "RX 8", "RX 9", "Arc"]

def _device_supports_fp16(device_name: str) -> bool:
    """Return True if the GPU backing *device_name* can safely use FP16."""
    device_name = device_name.lower()
    for gpu in SUPPORTED_FP16_GPUS:
        if gpu.lower() in device_name:
            return True
    return False

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DTYPE = torch.float16 if DEFAULT_DEVICE == "cuda" else torch.float32
MAX_DIRECTML_DEVICES = 4


def is_pytorch_at_least_2_4() -> bool:
    """Check if PyTorch version is at least 2.4"""
    try:
        from packaging.version import Version
        return Version(torch.__version__) >= Version("2.4")
    except (ImportError, Exception):
        return False


# -----------------------------------------------------------------------------
# Device resolution and type safety
# -----------------------------------------------------------------------------


def resolve_device(req_device: str, sel_dtype: str = "", todo_resolve: bool = False) -> str:
    """
    Resolve and normalize device string to a valid PyTorch device.
    
    Args:
        req_device: req_device device (e.g., 'auto', 'cuda', 'cpu', 'mps', 'igpu0')    
    Returns:
        Valid device string (e.g., 'cuda', 'cpu', 'mps', 'privateuseone:0')
    """
    global DEFAULT_DEVICE
    req_device = (req_device or "auto").lower()

    if not todo_resolve:
        return DEFAULT_DEVICE

    # Auto detection fallback to preferred hardware
    if req_device == "auto":
        try:
            dev = detect_preferred_device()
            DEFAULT_DEVICE = dev.type if isinstance(dev, torch.device) else str(dev)
            return DEFAULT_DEVICE
        except Exception as e:
            logger.warning(f"Auto device failed: {e}; using 'cpu'")
            DEFAULT_DEVICE = "cpu"
        return DEFAULT_DEVICE

    # Explicit CUDA request with fallback to CPU
    if req_device.startswith("cuda"):
        DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return DEFAULT_DEVICE

    # DirectML handling (Windows only)
    if req_device in ("dml", "directml", "igpu0", "igpu1", "igpu3", "igpu4"):
        try:
            import torch_directml as dml
            count = dml.device_count()
            if count > 0:
                td_dev = "privateuseone"
                for i in range(count):
                    if req_device == f'igpu{i}' or req_device == f'privateuseone:{i}':
                        logger.info(f"Device {i}: {dml.device_name(i)}")
                        td_dev = f'privateuseone:{i}'
                        break
                    name = dml.device_name(i).lower()
                    if req_device in name:
                        logger.info(f"Selected Device {i}: {name}")
                        td_dev = f'privateuseone:{i}'
                        break
                DEFAULT_DEVICE = td_dev
                return DEFAULT_DEVICE
            else:
                logger.warning("No DirectML devices found.")
        except Exception as e:
            logger.warning(f"DirectML import failed: {e}")

        DEFAULT_DEVICE = "cpu"
        return DEFAULT_DEVICE

    # MPS (Apple Silicon)
    if req_device == "mps":
        try:
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                DEFAULT_DEVICE = "mps"
                return DEFAULT_DEVICE
            else:
                logger.warning("MPS unavailable.")
        except Exception:
            logger.warning("MPS check failed; fallback to CPU.")
        DEFAULT_DEVICE = "cpu"
        return DEFAULT_DEVICE

    # Vulkan (if supported)
    if req_device == "vulkan":
        try:
            if getattr(torch._C, "has_vulkan", False):
                DEFAULT_DEVICE = "vulkan"
                return DEFAULT_DEVICE
            else:
                logger.warning("Vulkan not available in this PyTorch build.")
        except Exception:
            logger.warning("Vulkan check failed; fallback to CPU.")
        DEFAULT_DEVICE = "cpu"
        return DEFAULT_DEVICE

    # XLA (TPU)
    if req_device == "xla":
        try:
            if importlib.util.find_spec("torch_xla") is not None:
                DEFAULT_DEVICE = "xla"
                return DEFAULT_DEVICE
        except Exception:
            pass
        logger.warning("XLA unavailable; fallback to CPU.")
        DEFAULT_DEVICE = "cpu"
        return DEFAULT_DEVICE

    # HIP/ROCm (uses CUDA backend)
    if req_device in ("hip", ):
        if getattr(torch.version, "hip", None):
            DEFAULT_DEVICE = "cuda"  # ROCm uses 'cuda' device type
            return DEFAULT_DEVICE
        logger.warning("HIP/ROCm not available; fallback to CPU.")
        DEFAULT_DEVICE = "cpu"
        return DEFAULT_DEVICE

    # Default: accept 'cpu' or unknown strings → fallback to cpu
    if req_device == "cpu":
        DEFAULT_DEVICE = "cpu"
        return DEFAULT_DEVICE

    logger.warning(f"Unknown device '{req_device}' not available; fallback to 'cpu'")
    DEFAULT_DEVICE = "cpu"
    return DEFAULT_DEVICE


def get_valid_dtype_for_device(device: str = "", sel_dtype: str = "") -> torch.dtype:
    """Select appropriate dtype based on device and quantization preference."""
    global DEFAULT_DEVICE, DEFAULT_DTYPE

    if sel_dtype != "":
        DEFAULT_DTYPE = torch.float16 if sel_dtype == "float16" else torch.float32
    else:
        sel_dtype = "float16" if DEFAULT_DTYPE == torch.float16 else "float32"

    if device != "":
        DEFAULT_DEVICE = device
    else:
        device = DEFAULT_DEVICE

    # DirectML: enable FP16 only on supported GPUs
    if "privateuseone" in str(device):
        try:
            import torch_directml
            gpu_name = torch_directml.device_name(0)
            if device == "privateuseone:1":
                gpu_name = torch_directml.device_name(1)

            supports_fp16 = any(gpu in gpu_name for gpu in SUPPORTED_FP16_GPUS)
            if sel_dtype == "float16" and supports_fp16:
                return torch.float16
            else:
                return torch.float32
        except Exception as e:
            logger.warning(f"Failed to import torch_directml: {e}")

    # CUDA: FP16 available with non-strict quantization
    if "cuda" in str(device):
        return torch.float16 if sel_dtype == "float16" else torch.float32

    # CPU: always use float32 (no auto-cast)
    return torch.float32


# -----------------------------------------------------------------------------
# Mixed precision context manager & safe dtype conversion
# -----------------------------------------------------------------------------

def maybe_autocast_fp16(device: str = "", sel_dtype: str = "") -> contextlib.AbstractContextManager:
    """Return autocast context for FP16 on supported devices; otherwise nullcontext."""
    global DEFAULT_DEVICE, DEFAULT_DTYPE

    if sel_dtype != "":
        DEFAULT_DTYPE = torch.float16  if sel_dtype == "float16" else torch.float32
    else:
        sel_dtype = "float32" if DEFAULT_DTYPE == torch.float32 else "float16"

    if device == "":
        device = DEFAULT_DEVICE

    # DirectML: check GPU support for FP16
    if "privateuseone" in str(device) and sel_dtype != "float32":
        import torch_directml
        gpu_name = torch_directml.device_name(0)
        if device == "privateuseone:1":
            gpu_name = torch_directml.device_name(1)
        supports_fp16 = any(gpu in gpu_name for gpu in SUPPORTED_FP16_GPUS)
        if supports_fp16:
            logger.info(f"DirectML on {gpu_name}: FP16 enabled")
            return torch.autocast(device_type=device, dtype=torch.float16)
        else:
            logger.warning(f"DirectML on {gpu_name}: FP16 disabled")
            return contextlib.nullcontext()

    # CPU: disable autocast
    if "cpu" in str(device):
        logger.warning("CPU: autocast disabled")
        return contextlib.nullcontext()

    # CUDA: enable FP16 or FP32 based on quantization
    if "cuda" in device and sel_dtype == "float16":
        logger.info("CUDA: FP16 enabled autocast")
        return torch.autocast(device_type=device, dtype=torch.float16)
    elif "cuda" in device and sel_dtype == "float32":
        logger.info("CUDA: FP32 enabled autocast")
        return torch.autocast(device_type=device, dtype=torch.float32)

    logger.warning(f"{device}: autocast disabled")
    return contextlib.nullcontext()


def maybe_to_fp16(module, device: str = "", sel_dtype: str = "") -> torch.nn.Module:
    """Try to convert module to FP16 safely; return original if failed."""
    global DEFAULT_DEVICE, DEFAULT_DTYPE

    if module is None:
        return module

    if sel_dtype != "":
        DEFAULT_DTYPE = torch.float16  if sel_dtype == "float16" else torch.float32
    else:
        sel_dtype = "float16" if DEFAULT_DTYPE == torch.float16 else "float32"

    if device == "":
        device = DEFAULT_DEVICE

    # DirectML: check GPU support for FP16
    if "privateuseone" in str(device):
        import torch_directml
        gpu_name = torch_directml.device_name(0)
        if device == "privateuseone:1":
            gpu_name = torch_directml.device_name(1)

        supports_fp16 = any(gpu in gpu_name for gpu in SUPPORTED_FP16_GPUS)
        dtype = torch.float16 if sel_dtype == "float16" and supports_fp16 else torch.float32

    # CUDA: use FP16 or FP32 based on quantization
    elif "cuda" in str(device):
        dtype = torch.float16 if sel_dtype == "float16" else torch.float32

    # Default: float32 for CPU and unknown devices
    else:
        dtype = torch.float32

    try:
        return module.to(device=device, dtype=dtype)
    except Exception as e:
        logger.info(f"Exception maybe_to_fp16: {e}")
        return module


def force_to_fp16(module, device: str = "", sel_dtype: str = "") -> torch.nn.Module:
    """Try to convert module to FP16 safely; return original if failed."""   
    if module is None:
        return module
    dtype = torch.float16 
    try:
        return module.to(dtype=dtype)
    except Exception as e:
        logger.info(f"Exception force_to_fp32: {e}")
        return module


def force_to_fp32(module, device: str = "", sel_dtype: str = "") -> torch.nn.Module:
    """Try to convert module to FP32 safely; return original if failed."""    
    if module is None:
        return module
    dtype = torch.float32 
    try:
        return module.to(dtype=dtype)
    except Exception as e:
        logger.info(f"Exception force_to_fp32: {e}")
        return module

# -----------------------------------------------------------------------------
# Device detection & listing helper functions
# -----------------------------------------------------------------------------

def detect_preferred_device() -> torch.device:
    """Detect and return the most suitable available device."""
    # 1) ROCm/hip (uses 'cuda' backend)
    try:
        if getattr(torch.version, "hip", None):
            logger.info("ROCm detected; using 'cuda'")
            return torch.device("cuda")
    except Exception:
        pass

    # 2) CUDA
    try:
        if torch.cuda.is_available():
            logger.info("CUDA detected; selecting 'cuda'")
            return torch.device("cuda")
    except Exception:
        pass

    # 3) DirectML (torch-directml)
    try:
        if importlib.util.find_spec("torch_directml") is not None:
            try:
                import torch_directml as dml
                count = dml.device_count()
                if count > 0:
                    logger.info(f"Found {count} DirectML device(s)")
                    return torch.device(dml.device(0))
            except Exception as e:
                logger.warning("DirectML import failed: %s", e)
    except Exception:
        pass
    try:
        # Check for privateuseone devices
        for i in range(MAX_DIRECTML_DEVICES):  # Reasonable upper limit for GPUs
            try:
                # Just check if the device exists
                device = torch.device(f"privateuseone:{i}")
                logger.info(f"Found privateuseone device: {i}")
                return device
            except Exception:
                # If we get an exception, no more devices to check
                break
    except Exception as e:
        logger.debug(f"Could not check privateuseone backends: {e}")

    # 4) MPS (Apple Silicon)
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            logger.info("MPS available; selecting 'mps'")
            return torch.device("mps")
    except Exception:
        pass

    # 5) Vulkan
    try:
        if getattr(torch._C, "has_vulkan", False):
            vb = getattr(torch.backends, "vulkan", None)
            if vb is not None and getattr(vb, "is_available", lambda: False)():
                logger.info("Vulkan available; selecting 'vulkan'")
                return torch.device("vulkan")
    except Exception:
        pass

    # 6) XLA
    try:
        if importlib.util.find_spec("torch_xla") is not None:
            logger.info("XLA detected; trying to use 'xla'")
            try:
                importlib.import_module("torch_xla")
                return torch.device("xla")
            except Exception:
                logger.warning("XLA import failed despite module presence")
    except Exception:
        pass

    # No accelerator found → fallback to CPU
    logger.info("No hardware accelerator detected; fallback to 'cpu'")
    return torch.device("cpu")

def is_torch_directml_available() -> bool:
    """Check if torch_directml is available and has devices."""
    try:
        import torch_directml as dml
        return dml.device_count() > 0
    except ImportError:
        return False


def list_available_devices(verbose: bool = True) -> list[str]:
    """  Lists all usable devices and their real backend.
    Returns a list of str: device_name.
    Format: "iGPU:0", "iGPU:1", etc. for DirectML devices, but internally use "privateuseone:0", etc.
    - Example : ["AUTO", "CPU", "iGPU0", "CUDA0", "mps"] """
    devices = ["AUTO", "CPU"]

    # DirectML detection via torch_directml
    try:
        if importlib.util.find_spec("torch_directml") is not None:
            try:
                import torch_directml as dml
                count = dml.device_count()
                for i in range(count):
                    name = dml.device_name(i) or "Unknown"
                    device_str = f"iGPU{i}"
                    devices.append(device_str)
                    logger.info(f"Device {i}: {name} -> {device_str}")
                    if i >= MAX_DIRECTML_DEVICES :
                        break
            except Exception as e:
                logger.warning(f"DirectML detection error: {e}")
        else:
            # Probe for privateuseone backend
            try:
                for i in range(MAX_DIRECTML_DEVICES):
                    try:
                        torch.device(f"privateuseone:{i}")
                        device_str = f"iGPU{i}"
                        devices.append(device_str)
                        logger.info(f"Found privateuseone: {i} -> {device_str}")
                    except Exception:
                        break
            except Exception as e:
                logger.debug(f"Privateuseone probe failed: {e}")
    except Exception as e:
        logger.warning(f"DirectML detection error: {e}")

    # CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_str = f"CUDA{i}"
            devices.append(device_str)
            logger.info(f"Found CUDA device: {device_str}")
            if i >= MAX_DIRECTML_DEVICES :
                break

    # MPS (Apple Silicon)
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
            logger.info("Found MPS device")
    except Exception as e:
        logger.debug(f"MPS check failed: {e}")

    # XLA (TPU)
    try:
        if importlib.util.find_spec("torch_xla"):
            devices.append("xla")
            logger.info("Found XLA backend")
    except Exception as e:
        logger.debug(f"XLA check failed: {e}")

    if verbose:
        for i, dev in enumerate(devices, 1):
            print(f"{i}. {dev}")

    return devices


# -----------------------------------------------------------------------------
# Memory monitoring & utility functions
# -----------------------------------------------------------------------------

def logger_ram_used(str_msg: str = "") -> None:
    """Log current memory usage (RAM and GPU)."""
    process = psutil.Process(os.getpid())
    ram_ = process.memory_info().rss
    ram_mb = f"{ram_ / 1e6:.2f}"

    # GPU memory if available
    alloc_ = 0
    gpu_alloc = "N/A"
    max_gpu_alloc = "N/A"
    if DEFAULT_DEVICE.startswith("privateuseone"):
        try:
            import torch_directml
            device_id = 0
            if DEFAULT_DEVICE == "privateuseone:1":
                device_id = 1
            dml_device = torch_directml.device(device_id)
            alloc_mb = torch_directml.gpu_memory(device_id=device_id, mb_per_tile=1)
            # {{ Modification: Sum the tile allocations and format }}
            total_alloc_mb = sum(alloc_mb)
            gpu_alloc = f"{total_alloc_mb:.2f}"  # Display with 2 decimal places
            gpu_reserved = "N/A"  # Reserved memory not available
        except Exception as e:
            gpu_alloc = f"DML Error: {e}"
            gpu_reserved = f"DML Error"
            try:
                alloc_ = torch.memory_allocated(DEFAULT_DEVICE)
                gpu_alloc = f"{alloc_ / 1e6:.2f}"
            except Exception as e:
                pass
    elif torch.cuda.is_available():
        alloc_ = torch.cuda.memory_allocated()
        reserved_ = torch.cuda.memory_reserved()
        max_alloc_ = torch.cuda.max_memory_allocated()
        gpu_alloc = f"{alloc_ / 1e6:.2f}"
        max_gpu_alloc = f"{max_alloc_ / 1e6:.2f}"

    logger.info(f"RAM used {str_msg}: {ram_mb} MB")
    if gpu_alloc != "N/A":
        logger.info(f"VRAM GPU used: {gpu_alloc} MB")


def get_dml_device(name_substr: str = "") -> str:
    """Select DirectML device by name substring (case-insensitive)."""
    if name_substr == "cpu":
        return "cpu"
    elif name_substr == "cuda":
        return "cuda"

    try:
        import torch_directml as dml
        count = dml.device_count()
        if count == 0:
            raise RuntimeError("No DirectML devices found.")
        name_substr = (name_substr or "").lower()
        for i in range(count):
            name = dml.device_name(i)
            if name_substr in name.lower():
                logger.info(f"Device {i}: {name}")
                return f'privateuseone:{i}'
            elif name_substr == f'igpu{i}' or name_substr == f'privateuseone:{i}':
                logger.info(f"Device {i}: {name}")
                return f'privateuseone:{i}'
        logger.warning(f"No device matches '{name_substr}', fallback to index 0 ({dml.device_name(0)})")
    except Exception as e:
        logger.info(f"Exception in get_dml_device: {e}")

    return "privateuseone"
