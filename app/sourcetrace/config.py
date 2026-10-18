"""Runtime configuration, resolved from ``SOURCETRACE_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_EMBED_ID = "BAAI/bge-small-en-v1.5"
BAKED_MODEL_DIR = "/models/model"
BAKED_EMBED_DIR = "/models/embed"

MODES = ("auto", "server", "standalone")
BACKENDS = ("qwen", "mock")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_model_path() -> str:
    explicit = os.environ.get("SOURCETRACE_MODEL_PATH")
    if explicit:
        return explicit
    return BAKED_MODEL_DIR if Path(BAKED_MODEL_DIR).is_dir() else DEFAULT_MODEL_ID


def _default_embed_path() -> str:
    explicit = os.environ.get("SOURCETRACE_EMBED_PATH")
    if explicit is not None:  # an explicit empty string disables dense retrieval
        return explicit
    return BAKED_EMBED_DIR if Path(BAKED_EMBED_DIR).is_dir() else ""


def _default_index_dir() -> str:
    return (
        os.environ.get("SOURCETRACE_INDEX_DIR")
        or os.environ.get("MC3_INDEX_DIR")
        or os.environ.get("MC2_INDEX_DIR")
        or "/app/index"
    )


def _default_output_dir() -> str:
    return (
        os.environ.get("SOURCETRACE_OUTPUT_DIR")
        or os.environ.get("MC3_OUTPUT_DIR")
        or os.environ.get("MC2_OUTPUT_DIR")
        or "/app/output"
    )


def _default_socket_path() -> str:
    explicit = os.environ.get("SOURCETRACE_SOCKET")
    if explicit:
        return explicit
    if os.name == "nt":
        import tempfile
        return str(Path(tempfile.gettempdir()) / "sourcetrace.sock")
    return "/tmp/sourcetrace.sock"


@dataclass(frozen=True)
class Settings:
    mode: str = "auto"
    backend: str = "qwen"
    model_path: str = DEFAULT_MODEL_ID
    embed_path: str = ""  # "" disables dense retrieval (BM25 only)
    socket_path: str = "/tmp/sourcetrace.sock"
    index_dir: str = "/app/index"
    output_dir: str = "/app/output"
    # Time budgets (seconds). The harness limits are 30 per question and 600 for startup.
    client_deadline_s: float = 27.0
    request_budget_s: float = 24.0
    index_budget_s: float = 420.0
    index_client_wait_s: float = 570.0
    file_timeout_s: float = 60.0
    # Retrieval / reasoning
    top_chunks: int = 10
    max_chunks_per_file: int = 3
    max_hops: int = 2
    max_new_tokens: int = 160
    vision_lookup: bool = True
    prune_citations: bool = True
    verify_answer: bool = True
    max_image_side: int = 1600
    vram_cap_gib: float = 44.0
    allow_cpu: bool = False
    trace_dir: str = ""
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            mode=_env_str("SOURCETRACE_MODE", d.mode),
            backend=_env_str("SOURCETRACE_BACKEND", d.backend),
            model_path=_default_model_path(),
            embed_path=_default_embed_path(),
            socket_path=_default_socket_path(),
            index_dir=_default_index_dir(),
            output_dir=_default_output_dir(),
            client_deadline_s=_env_float("SOURCETRACE_CLIENT_DEADLINE_S", d.client_deadline_s),
            request_budget_s=_env_float("SOURCETRACE_REQUEST_BUDGET_S", d.request_budget_s),
            index_budget_s=_env_float("SOURCETRACE_INDEX_BUDGET_S", d.index_budget_s),
            index_client_wait_s=_env_float(
                "SOURCETRACE_INDEX_CLIENT_WAIT_S", d.index_client_wait_s
            ),
            file_timeout_s=_env_float("SOURCETRACE_FILE_TIMEOUT_S", d.file_timeout_s),
            top_chunks=_env_int("SOURCETRACE_TOP_CHUNKS", d.top_chunks),
            max_chunks_per_file=_env_int("SOURCETRACE_MAX_CHUNKS_PER_FILE", d.max_chunks_per_file),
            max_hops=_env_int("SOURCETRACE_MAX_HOPS", d.max_hops),
            max_new_tokens=_env_int("SOURCETRACE_MAX_NEW_TOKENS", d.max_new_tokens),
            vision_lookup=_env_bool("SOURCETRACE_VISION_LOOKUP", d.vision_lookup),
            prune_citations=_env_bool("SOURCETRACE_PRUNE_CITATIONS", d.prune_citations),
            verify_answer=_env_bool("SOURCETRACE_VERIFY_ANSWER", d.verify_answer),
            max_image_side=_env_int("SOURCETRACE_MAX_IMAGE_SIDE", d.max_image_side),
            vram_cap_gib=_env_float("SOURCETRACE_VRAM_CAP_GIB", d.vram_cap_gib),
            allow_cpu=_env_bool("SOURCETRACE_ALLOW_CPU", d.allow_cpu),
            trace_dir=_env_str("SOURCETRACE_TRACE_DIR", d.trace_dir),
            log_level=_env_str("SOURCETRACE_LOG_LEVEL", d.log_level).upper(),
        )
