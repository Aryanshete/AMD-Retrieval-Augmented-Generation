# Your image MUST be built on the ROCm base image AMD mandates. This is checked
# by layer identity before your container is run -- we compare the lower layers
# of your image against the base itself, not the name you gave it -- and an
# image that is not built on it is disqualified.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Install dependencies while strictly protecting ROCm PyTorch from replacement
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app/ /app/
COPY models/ /models/

# Ensure required run, corpus, output, and persistence directories exist
RUN mkdir -p /app/corpus /app/output /app/index /app/run /models /models/llm /models/ocr

# Verify PyTorch installation retains ROCm build
RUN python3 -c "import torch; print('Container PyTorch version:', torch.__version__); assert 'rocm' in torch.__version__.lower(), 'ROCm PyTorch build was corrupted!'"

# Pre-download EasyOCR weights so it operates 100% offline under --network none
RUN python3 -c "import easyocr; easyocr.Reader(['en'], model_storage_directory='/models/ocr', download_enabled=True, gpu=False); print('EasyOCR weights downloaded.')"

# Download 100% free offline LLM weights and tokenizer (Qwen/Qwen2.5-1.5B-Instruct)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download( \
        repo_id='Qwen/Qwen2.5-1.5B-Instruct', \
        local_dir='/models/llm', \
        ignore_patterns=['*.bin', '*.pt', '*.onnx', '*.msgpack'], \
        local_dir_use_symlinks=False \
    ); print('Offline LLM weights downloaded successfully.')"

# Verify offline model config and tokenizer load without network
RUN python3 -c "from transformers import AutoConfig, AutoTokenizer; \
    AutoConfig.from_pretrained('/models/llm', local_files_only=True); \
    AutoTokenizer.from_pretrained('/models/llm', local_files_only=True); \
    print('Offline model and tokenizer verified successfully.')"

# Start background resident worker daemon and keep container alive
CMD ["bash", "-lc", "python3 /app/worker.py & exec sleep infinity"]
