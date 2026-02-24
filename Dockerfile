FROM vllm/vllm-openai:latest

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_PATH=/app/models/translate-gemma \
    MODEL_DISPLAY_NAME=Sharifsetup-Translator \
    TENSOR_PARALLEL_SIZE=1 \
    DTYPE=bfloat16 \
    MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.92

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY scripts /app/scripts
COPY models /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
