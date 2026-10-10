#!/usr/bin/env python3
"""AMD ROCm GPU Hardware and VRAM Compliance Verification Script.

Usage:
    python3 check_gpu.py
    python3 check_gpu.py --require-gpu

Checks:
1. PyTorch installation & ROCm HIP compilation.
2. AMD GPU device availability & device name (e.g. Instinct MI210 / MI250).
3. Tensor allocation & execution on GPU (testing 1-48 GiB VRAM compliance).
4. Prints step-by-step instructions for AMD Developer Cloud (ADC) free tier access.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

VRAM_FLOOR_GIB = 1.0
VRAM_CEILING_GIB = 48.0


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def check_gpu(require_gpu: bool = False) -> int:
    print_banner("AMD AI LEAGUE MATCH 3: ROCm GPU & VRAM VERIFICATION")

    # 1. Inspect PyTorch
    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch is not installed in the current environment.")
        return 1

    torch_ver = torch.__version__
    is_rocm = "rocm" in torch_ver.lower()
    hip_ver = getattr(torch.version, "hip", None)

    print(f"[*] PyTorch Version : {torch_ver}")
    print(f"[*] ROCm Build Flag : {is_rocm}")
    print(f"[*] HIP Runtime Ver : {hip_ver if hip_ver else 'N/A'}")

    # 2. Check GPU Device Availability
    gpu_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if gpu_available else 0

    print(f"[*] CUDA/ROCm Available : {gpu_available}")
    print(f"[*] Visible GPU Devices : {device_count}")

    if not gpu_available:
        print("\n" + "-" * 70)
        print("[WARNING] No AMD ROCm GPU detected on the current host machine.")
        print("          Current host is running on CPU.")
        print("-" * 70)
        print("\nWHY THIS HAPPENS:")
        print("  - Local consumer laptops (Windows/Mac) do not have enterprise AMD Instinct GPUs")
        print("    (MI210/MI250/MI300) or Linux ROCm kernel drivers installed.")
        print("  - The AMD evaluation harness runs on AMD GPU datacenter nodes where ROCm is active.")
        print("\nHOW TO SATISFY THE REQUIREMENT WITH AMD DEVELOPER CLOUD (ADC) FREE TIER:")
        print("  1. Access AMD Developer Cloud: https://cloud.amd.com (or via Lablab AI League voucher)")
        print("  2. Launch an AMD Instinct instance (Ubuntu 22.04 with MI210 / MI250)")
        print("  3. Git clone this repository onto the ADC VM")
        print("  4. Build the Docker container:")
        print("     docker build -t mc3-rag:final .")
        print("  5. Run selfcheck with continuous VRAM monitoring:")
        print("     watch -n1 'amd-smi metric --mem | grep USED_VRAM' &")
        print("     python3 selfcheck.py mc3-rag:final mc3-corpus")
        print("-" * 70)

        if require_gpu:
            print("\n[FAIL] --require-gpu was specified or MC3_REQUIRE_GPU=1 is set.")
            print("       CPU fallback is prohibited to avoid disqualification.")
            return 1
        return 0

    # 3. GPU is available: Run AMD GPU Tensor Benchmark
    device_name = torch.cuda.get_device_name(0)
    print(f"[*] Active Device [0]   : {device_name}")

    print("\n[*] Allocating genuine FP16 tensors on AMD GPU to verify VRAM & tensor cores...")
    t0 = time.time()
    try:
        # Allocate ~1.2 GiB in VRAM to safely clear the 1.0 GiB floor
        # Shape: (40000, 8192) in float16 = 40000 * 8192 * 2 bytes ~= 655 MB x 2 tensors ~= 1.3 GiB
        rows, cols = 40000, 8192
        t1 = torch.ones((rows, cols), device="cuda:0", dtype=torch.float16)
        t2 = torch.ones((cols, 256), device="cuda:0", dtype=torch.float16)

        # Run matrix multiplication on ROCm tensor cores
        res = torch.mm(t1, t2)
        torch.cuda.synchronize(0)
        elapsed = time.time() - t0

        current_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

        print(f"[PASS] AMD GPU Kernel Execution completed in {elapsed:.3f}s")
        print(f"[*] Current Allocated VRAM : {current_vram:.2f} GiB")
        print(f"[*] Peak Allocated VRAM    : {peak_vram:.2f} GiB")

        # 4. Check against official competition limits (1 - 48 GiB)
        if VRAM_FLOOR_GIB <= peak_vram <= VRAM_CEILING_GIB:
            print(f"[PASS] VRAM Peak ({peak_vram:.2f} GiB) is within the competition band [1 – 48 GiB].")
            print("[SUCCESS] AMD ROCm GPU verification PASSED!")
            return 0
        elif peak_vram < VRAM_FLOOR_GIB:
            print(f"[FAIL] Peak VRAM ({peak_vram:.2f} GiB) is below the 1.0 GiB floor (scores zero).")
            return 1
        else:
            print(f"[FAIL] Peak VRAM ({peak_vram:.2f} GiB) exceeded the 48.0 GiB ceiling.")
            return 1

    except Exception as exc:
        print(f"[FAIL] Error during GPU execution: {exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Check AMD ROCm GPU availability and VRAM limits.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        default=os.environ.get("MC3_REQUIRE_GPU", "").strip().lower() in ("1", "true", "yes", "strict"),
        help="Enforce strict GPU requirement (exit 1 if GPU is not detected).",
    )
    args = parser.parse_args()
    return check_gpu(require_gpu=args.require_gpu)


if __name__ == "__main__":
    sys.exit(main())
