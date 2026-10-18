#!/usr/bin/env python3
"""Download model weights into a directory (used at image build time).

    python scripts/download_model.py --model-id Qwen/Qwen3-VL-8B-Instruct --dest /models/model

Only safetensors weights and tokenizer/config files are fetched so the image
stays as small as possible. Pass ``--revision`` with a commit hash to pin the
exact weights (recommended for reproducible builds).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dest", type=Path, default=Path("/models/model"))
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    args.dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(args.dest),
        ignore_patterns=[
            "*.bin",
            "*.pt",
            "*.pth",
            "*.onnx",
            "*.msgpack",
            "*.h5",
            "original/*",
            "*.gguf",
        ],
    )
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"downloaded {args.model_id}@{args.revision} to {path} ({size / 1024**3:.1f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
