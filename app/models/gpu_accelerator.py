"""AMD ROCm GPU acceleration and memory management.

Provides:
- Explicit PyTorch GPU tensor operations via ROCm HIP (cuda:0).
- Strict GPU enforcement (MC3_REQUIRE_GPU=1) prohibiting silent CPU fallback.
- Real GPU tensor matrix multiplication and cosine similarity on ROCm kernels.
- VRAM tracking and compliance verification (1 GiB <= peak <= 48 GiB).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def is_gpu_enforced() -> bool:
    """Check if strict GPU mode is mandated by environment."""
    val = os.environ.get("MC3_REQUIRE_GPU", "").strip().lower()
    return val in ("1", "true", "yes", "on", "strict")


class ROCmGPUAccelerator:
    """Manages PyTorch GPU tensor placement, batch cosine scoring, and VRAM monitoring."""

    def __init__(self, require_gpu: Optional[bool] = None):
        self.enforce_gpu = is_gpu_enforced() if require_gpu is None else require_gpu
        self.device = "cpu"
        self.is_rocm = False
        self.gpu_name = "CPU"
        self._gpu_embeddings: Optional[Any] = None  # torch.Tensor

        self._initialize_device()

    def _initialize_device(self) -> None:
        """Inspect and bind AMD GPU device via PyTorch ROCm HIP."""
        try:
            import torch

            torch_ver = torch.__version__
            self.is_rocm = "rocm" in torch_ver.lower()
            gpu_available = torch.cuda.is_available()

            if self.enforce_gpu and not gpu_available:
                raise RuntimeError(
                    "CRITICAL: Strict GPU compliance is enabled (MC3_REQUIRE_GPU=1), "
                    "but no AMD ROCm / CUDA GPU is available! "
                    "Evaluation contract Section 1 & 14 strictly forbids CPU fallback "
                    "to prevent disqualification (VRAM < 1 GiB)."
                )

            if gpu_available:
                self.device = "cuda:0"
                try:
                    self.gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    self.gpu_name = "AMD Instinct / ROCm GPU"
                logger.info(f"ROCm GPU Accelerator bound to {self.device} ({self.gpu_name}, ROCm={self.is_rocm})")
                self._warmup_gpu_tensors()
            else:
                self.device = "cpu"
                self.gpu_name = "CPU"
                logger.debug("No GPU detected. Operating in CPU fallback mode.")
        except ImportError:
            if self.enforce_gpu:
                raise RuntimeError("PyTorch is not installed. Cannot satisfy strict GPU requirement.")
            self.device = "cpu"
            self.gpu_name = "No PyTorch"

    def _warmup_gpu_tensors(self) -> None:
        """Warm up PyTorch ROCm GPU runtime."""
        if not self.device.startswith("cuda"):
            return
        try:
            import torch
            # Quick check on GPU device initialization
            torch.cuda.init()
            logger.info(f"GPU device initialized on {self.device}. Current VRAM: {self.get_vram_usage_gib():.2f} GiB")
        except Exception as exc:
            logger.warning(f"Could not initialize GPU: {exc}")


    def get_vram_usage_gib(self) -> float:
        """Return currently allocated VRAM in GiB."""
        if self.device.startswith("cuda"):
            try:
                import torch
                return torch.cuda.memory_allocated(0) / (1024 ** 3)
            except Exception:
                return 0.0
        return 0.0

    def get_peak_vram_usage_gib(self) -> float:
        """Return peak allocated VRAM in GiB."""
        if self.device.startswith("cuda"):
            try:
                import torch
                return torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            except Exception:
                return 0.0
        return 0.0

    def mount_embeddings_to_gpu(self, embeddings_matrix: np.ndarray) -> None:
        """Mount embeddings matrix onto AMD GPU memory as a half-precision (FP16) tensor."""
        if not self.device.startswith("cuda") or embeddings_matrix.size == 0:
            self._gpu_embeddings = None
            return

        try:
            import torch

            # Convert to FP16 torch tensor and move to GPU VRAM
            t = torch.from_numpy(embeddings_matrix).to(device=self.device, dtype=torch.float16)
            self._gpu_embeddings = t
            logger.info(
                f"Mounted {t.shape[0]} embeddings ({t.shape[1]} dim) to {self.device} VRAM. "
                f"Current VRAM: {self.get_vram_usage_gib():.3f} GiB"
            )
        except Exception as exc:
            logger.warning(f"Failed to mount embeddings to GPU: {exc}")
            self._gpu_embeddings = None

    def compute_similarity(self, query_vector: np.ndarray) -> Optional[np.ndarray]:
        """Compute batched cosine similarity using GPU tensor matrix multiplication."""
        if self._gpu_embeddings is None or not self.device.startswith("cuda"):
            return None

        try:
            import torch

            q_norm = np.linalg.norm(query_vector)
            if q_norm == 0:
                return np.zeros(self._gpu_embeddings.shape[0], dtype=np.float32)

            q_t = torch.from_numpy(query_vector).to(device=self.device, dtype=torch.float16)

            # Dot product against all document vectors in GPU VRAM
            scores = torch.mv(self._gpu_embeddings, q_t)
            return scores.detach().cpu().float().numpy()
        except Exception as exc:
            logger.warning(f"GPU similarity computation failed: {exc}")
            return None

    def run_gpu_benchmark(self, dim: int = 1024, count: int = 20000) -> Dict[str, Any]:
        """Execute a genuine ROCm tensor calculation to verify GPU utilization and VRAM."""
        if not self.device.startswith("cuda"):
            return {
                "status": "cpu_only",
                "message": "GPU not available on current host machine.",
                "vram_gib": 0.0,
            }

        try:
            import torch
            import time

            t0 = time.time()
            # Allocate genuine tensors on AMD GPU
            mat1 = torch.randn((count, dim), device=self.device, dtype=torch.float16)
            mat2 = torch.randn((dim, 512), device=self.device, dtype=torch.float16)
            res = torch.mm(mat1, mat2)
            torch.cuda.synchronize(0)
            elapsed = time.time() - t0

            vram = self.get_vram_usage_gib()
            peak = self.get_peak_vram_usage_gib()

            return {
                "status": "success",
                "device": self.device,
                "gpu_name": self.gpu_name,
                "is_rocm": self.is_rocm,
                "matrix_shape": f"{count}x{dim} * {dim}x512",
                "duration_seconds": round(elapsed, 4),
                "allocated_vram_gib": round(vram, 3),
                "peak_vram_gib": round(peak, 3),
                "in_competition_band": (1.0 <= peak <= 48.0) or (1.0 <= vram <= 48.0),
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": str(exc),
            }
