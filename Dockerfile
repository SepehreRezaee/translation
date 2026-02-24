FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_PATH=/app/models/sharifsetup-translate \
    MODEL_DISPLAY_NAME=Sharifsetup-Translator \
    VERBOSE_LOGS=false \
    MODEL_DEVICE=auto \
    DTYPE=bfloat16 \
    MAX_MODEL_LEN=4096 \
    TRUST_REMOTE_CODE=true

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY scripts /app/scripts
COPY models /app/models
RUN chmod +x /app/scripts/start.sh

EXPOSE 8000

CMD ["/app/scripts/start.sh"]
