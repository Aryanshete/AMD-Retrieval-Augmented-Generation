"""Offline model loader and device configuration.

Ensures:
- Strict offline loading with local_files_only=True.
- AMD ROCm GPU detection via CUDA-compatible API.
- Memory and precision management (FP16 / BF16 on GPU).
- VRAM tracking and runtime device validation.
- Safe CPU fallback in offline CPU test environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Default offline model storage directory in Docker image
DEFAULT_MODELS_DIR = Path(os.environ.get("MC3_MODELS_DIR", "/models"))


@dataclass
class DeviceInfo:
    """Hardware acceleration and ROCm runtime details."""
    is_rocm: bool
    is_gpu_available: bool
    device_name: str
    device: str
    torch_version: str
    vram_allocated_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_rocm": self.is_rocm,
            "is_gpu_available": self.is_gpu_available,
            "device_name": self.device_name,
            "device": self.device,
            "torch_version": self.torch_version,
            "vram_allocated_mb": round(self.vram_allocated_mb, 2),
        }


def get_device_info() -> DeviceInfo:
    """Inspect PyTorch installation and AMD ROCm GPU acceleration state."""
    try:
        import torch
        torch_ver = torch.__version__
        is_rocm = "rocm" in torch_ver.lower()
        gpu_avail = torch.cuda.is_available()

        if gpu_avail:
            device = "cuda"
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:
                device_name = "AMD GPU (ROCm)" if is_rocm else "CUDA GPU"
            try:
                vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
            except Exception:
                vram_mb = 0.0
        else:
            req_gpu = os.environ.get("MC3_REQUIRE_GPU", "").strip().lower() in ("1", "true", "yes", "on", "strict")
            if req_gpu:
                raise RuntimeError(
                    "CRITICAL: MC3_REQUIRE_GPU=1 is enabled, but no AMD ROCm / CUDA GPU was detected! "
                    "GPU acceleration required; CPU fallback is disabled in evaluation mode."
                )
            device = "cpu"
            device_name = "CPU"
            vram_mb = 0.0

        return DeviceInfo(
            is_rocm=is_rocm,
            is_gpu_available=gpu_avail,
            device_name=device_name,
            device=device,
            torch_version=torch_ver,
            vram_allocated_mb=vram_mb,
        )
    except ImportError:
        return DeviceInfo(
            is_rocm=False,
            is_gpu_available=False,
            device_name="None (PyTorch not installed)",
            device="cpu",
            torch_version="none",
            vram_allocated_mb=0.0,
        )


class ModelLoader:
    """Loads and caches models strictly from local disk for offline evaluation."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.device_info = get_device_info()
        logger.info(
            f"ModelLoader initialized on device={self.device_info.device!r} "
            f"(ROCm={self.device_info.is_rocm}, GPU={self.device_info.device_name})"
        )

    def resolve_model_path(self, model_name_or_subpath: str) -> Path:
        """Resolve model path strictly on local disk without network download."""
        path = Path(model_name_or_subpath)
        if path.is_absolute() and path.exists():
            return path

        candidate = self.models_dir / model_name_or_subpath
        if candidate.exists():
            return candidate

        # Return candidate path for error messaging / verification
        return candidate

    def load_offline_model(
        self, model_identifier: str, use_fp16: bool = True
    ) -> Optional[Any]:
        """Load model weights strictly from disk.
        
        Guarantees:
        - NEVER issues runtime network requests (local_files_only=True).
        - Automatically mounts model to GPU if available.
        - Employs FP16 for ROCm memory efficiency.
        """
        model_path = self.resolve_model_path(model_identifier)
        if not model_path.exists():
            logger.warning(
                f"Offline model path does not exist: {model_path}. "
                "Operating in zero-network fallback mode."
            )
            return None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = torch.float16 if (self.device_info.is_gpu_available and use_fp16) else torch.float32

            # Force local offline loading without network access
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                local_files_only=True,
                torch_dtype=torch_dtype,
                device_map=self.device_info.device if self.device_info.is_gpu_available else None,
            )
            logger.info(f"Loaded offline model from {model_path} onto {self.device_info.device}")
            return model
        except Exception as exc:
            logger.warning(f"Could not load Hugging Face model from {model_path}: {exc}")
            return None
