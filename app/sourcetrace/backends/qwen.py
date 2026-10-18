"""Qwen3-VL on Hugging Face ``transformers``: one resident model for text and images.

``torch``/``transformers`` are imported lazily so tests run without a GPU stack.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from ..config import Settings

log = logging.getLogger("sourcetrace.qwen")


class QwenVLBackend:
    name = "qwen3-vl"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._torch: Any = None
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        path = self._settings.model_path
        use_gpu = torch.cuda.is_available()  # True on ROCm builds too
        if not use_gpu and not self._settings.allow_cpu:
            raise RuntimeError(
                "No GPU visible to PyTorch. Run with --device=/dev/kfd --device=/dev/dri "
                "(or set SOURCETRACE_ALLOW_CPU=1 for development)."
            )
        if use_gpu:
            total = torch.cuda.get_device_properties(0).total_memory
            cap = self._settings.vram_cap_gib * 1024**3
            if cap < total:
                # The harness rejects runs whose peak VRAM exceeds 48 GiB.
                torch.cuda.set_per_process_memory_fraction(cap / total)

        log.info("loading %s (gpu=%s)", path, use_gpu)
        self._processor = AutoProcessor.from_pretrained(path)
        dtype = torch.bfloat16 if use_gpu else torch.float32
        kwargs: dict[str, Any] = {"attn_implementation": "sdpa"}
        if use_gpu:
            kwargs["device_map"] = "cuda"
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, **kwargs)
        except TypeError:  # older transformers spell it torch_dtype
            self._model = AutoModelForImageTextToText.from_pretrained(
                path, torch_dtype=dtype, **kwargs
            )
        self._model.eval()
        log.info("model ready")

    def warmup(self) -> None:
        self.generate("Reply with the single word OK.", max_new_tokens=3, timeout_s=60.0)
        blank = Image.new("RGB", (224, 224), (255, 255, 255))
        self.generate(
            "Reply with the single word OK.",
            image=blank,
            max_new_tokens=3,
            timeout_s=60.0,
        )

    def generate(
        self,
        user: str,
        *,
        system: str | None = None,
        image: Image.Image | None = None,
        max_new_tokens: int = 128,
        timeout_s: float = 20.0,
    ) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("backend not loaded")
        torch = self._torch
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": user})
        messages.append({"role": "user", "content": content})
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if image is not None:
            kwargs["images"] = [image]
        inputs = self._processor(**kwargs).to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                max_time=max(0.5, timeout_s),
            )
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text.strip()
