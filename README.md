# Sharifsetup-Translator Offline API (`vLLM` + `FastAPI`)

Production-ready multilingual translation service that:
- uses display name `Sharifsetup-Translator` in API responses
- downloads `google/translategemma-4b-it` once into `./models/sharifsetup-translate`
- embeds that local model folder into the Docker image
- serves many-to-many translation offline at runtime (no model download at startup)

## 1. Download model once (local storage)

TranslateGemma may require Hugging Face access approval + token.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip huggingface-hub
python scripts/download_model.py \
  --model-id google/translategemma-4b-it \
  --output-dir models/sharifsetup-translate \
  --hf-token "$HF_TOKEN"
```

After this step, your model is stored inside this project and reused forever unless you replace it.

## 2. Build Docker image (copies model directory)

```bash
docker build -t gemma-translator:offline .
```

`Dockerfile` contains:
```dockerfile
COPY models /app/models
```
So the local model snapshot is inside the final image.

## 3. Run API container

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -e MODEL_PATH=/app/models/sharifsetup-translate \
  -e MODEL_DISPLAY_NAME=Sharifsetup-Translator \
  -e VERBOSE_LOGS=false \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  gemma-translator:offline
```

Or with compose:

```bash
docker compose up --build
```

### Logging flag

- `VERBOSE_LOGS=true`: show all levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `VERBOSE_LOGS=false`: show `ERROR` level only

## 4. API usage

Health:
```bash
curl http://localhost:8000/health
```

Expected model display:
`"model": "Sharifsetup-Translator"`

Translate (many-to-many):
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "source_language": "auto",
    "target_language": "French",
    "texts": [
      "Hello, how are you?",
      "Buenos dias, bienvenidos a nuestra plataforma."
    ],
    "preserve_formatting": true
  }'
```

You can translate any language pair by changing `source_language` and `target_language` (or keeping source as `auto`).

## Notes for production

- Requires NVIDIA GPU + NVIDIA Container Toolkit.
- Keep `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to guarantee offline runtime behavior.
- For larger Gemma variants, tune:
  - `TENSOR_PARALLEL_SIZE`
  - `MAX_MODEL_LEN`
  - `GPU_MEMORY_UTILIZATION`
