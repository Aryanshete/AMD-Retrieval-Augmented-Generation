#!/usr/bin/env python3
"""Run this FIRST on the AMD notebook, before any Docker work.

It answers the questions that decide whether the plan works:

  1. Does PyTorch see the GPU, and is it a ROCm build?
  2. Do transformers + the Qwen3-VL processor import (torchvision!) on this Python/torch?
  3. Does the model follow the JSON answer format, and read text from a picture?
  4. How long does a call take, and what is the peak VRAM?
  5. (optional) does the embedding model load?

    pip install --no-deps -U transformers tokenizers safetensors     # never plain -U with torch deps
    pip install -r app/requirements.txt
    python scripts/smoke_test_model.py --model Qwen/Qwen3-VL-8B-Instruct
    python scripts/smoke_test_model.py --embed BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="HF id or local path")
    ap.add_argument("--embed", default="", help="optional embedding model id or path")
    args = ap.parse_args()

    import torch

    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  hip={torch.version.hip}")
    if "+cu" in torch.__version__ or not torch.version.hip:
        print(
            "FAIL: torch is not a ROCm build (pip replaced it?). Reinstall the base image's torch."
        )
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no GPU visible to PyTorch")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")

    try:
        import transformers

        print(f"transformers {transformers.__version__}")
    except ImportError as exc:
        print(f"FAIL: {exc}; run: pip install -r app/requirements.txt")
        return 1
    try:
        import torchvision

        print(f"torchvision {torchvision.__version__}")
    except ImportError:
        print("note: torchvision not installed; if the processor complains, see README")

    from PIL import Image, ImageDraw

    from sourcetrace.backends.qwen import QwenVLBackend
    from sourcetrace.config import Settings
    from sourcetrace.postprocess import clean_answer, parse_reply
    from sourcetrace.prompts import (
        IMAGE_INDEX_PROMPT,
        SYSTEM,
        build_answer_prompt,
        format_passage,
    )

    backend = QwenVLBackend(Settings(model_path=args.model))
    t0 = time.monotonic()
    try:
        backend.load()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL while loading the model: {type(exc).__name__}: {exc}")
        return 1
    print(f"model loaded in {time.monotonic() - t0:.0f}s")
    backend.warmup()

    passages = [
        format_passage(1, "specs/r2.pdf", "p1", "Maximum junction temperature .... 94 C", False),
        format_passage(2, "specs/r1.pdf", "p1", "WITHDRAWN. Maximum junction temperature .... 105 C", True),
    ]  # fmt: skip
    prompt = build_answer_prompt("What is the maximum junction temperature of the TQ-40?", passages)
    t0 = time.monotonic()
    raw = backend.generate(prompt, system=SYSTEM, max_new_tokens=160)
    print(f"\ntext answer ({time.monotonic() - t0:.1f}s): {raw}")
    ok_text = clean_answer(parse_reply(raw).get("answer")) == "94"
    print("  ->", "OK" if ok_text else "UNEXPECTED (expected 94)")

    refusal_prompt = build_answer_prompt("What is the unit price of the TQ-40?", passages)
    raw = backend.generate(refusal_prompt, system=SYSTEM, max_new_tokens=160)
    ok_refuse = clean_answer(parse_reply(raw).get("answer")) == ""
    print(f"refusal check: {raw}\n  ->", "OK" if ok_refuse else "BAD: answered from nothing")

    img = Image.new("RGB", (700, 260), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 40), "BOARD REVISION  REV-C2", fill="black")
    d.text((30, 100), "SN  OS4-118823-77", fill="black")
    t0 = time.monotonic()
    raw = backend.generate(IMAGE_INDEX_PROMPT, system=SYSTEM, image=img, max_new_tokens=300)
    print(f"\nimage reading ({time.monotonic() - t0:.1f}s):\n{raw}")
    ok_img = "REV-C2" in raw.replace(" ", "")
    print("  ->", "OK" if ok_img else "UNEXPECTED (REV-C2 not found)")

    print(f"\npeak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB (limit 48)")

    if args.embed:
        from sourcetrace.embed import Embedder

        emb = Embedder(args.embed)
        emb.load()
        vec = emb.encode(["junction temperature"], query=True)
        print(f"embedding model OK, dim={vec.shape[1]}")

    good = ok_text and ok_refuse and ok_img
    print("\nSMOKE TEST", "PASSED" if good else "NEEDS ATTENTION")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
