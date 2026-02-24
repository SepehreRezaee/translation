# Sharifsetup-Translator Offline API (`Transformers` + `FastAPI`)

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
  -e MODEL_DEVICE=auto \
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

### Gemma3 compatibility note

- This project uses `google/translategemma-4b-it` (`gemma3` architecture).
- If you see: `model type gemma3 ... Transformers does not recognize this architecture`, run:
```bash
pip install --upgrade --upgrade-strategy eager -r requirements.txt
```
- `TRUST_REMOTE_CODE=true` is enabled by default for compatibility.

## 4. API usage

Health:
```bash
curl http://localhost:8000/health
```

Expected model display:
`"model": "Sharifsetup-Translator"`

Translate text (`/translate/text`):
```bash
curl -X POST http://localhost:8000/translate/text \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hello, how are you?",
    "language": ["fr (French)", "de"]
  }'
```

`language` accepts list values in these formats:
- `fr`
- `French`
- `fr (French)`

Each translation item in response includes:
- `target_lang_code` (for example `fr`)
- `language` (for example `fr (French)`)
- `translated_text`

If your text contains control characters or raw newlines, you can use form data:
```bash
curl -X POST http://localhost:8000/translate/text \
  -F "language=fr (French)" \
  -F "language=de" \
  -F "content=Hello
Line 2"
```

Translate uploaded file (`/translate/file`):
```bash
curl -X POST http://localhost:8000/translate/file \
  -F "language=fr (French)" \
  -F "language=de" \
  -F "file=@./sample.txt"
```

Allowed file extensions for `/translate/file`:
- `.pdf`
- `.docx`
- `.txt`

The API now exposes exactly:
- `GET /health`
- `POST /translate/text`
- `POST /translate/file`

## Notes for production

- Requires NVIDIA GPU + NVIDIA Container Toolkit.
- Keep `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to guarantee offline runtime behavior.
- For larger variants, tune:
  - `MODEL_DEVICE`
  - `DTYPE`
  - `MAX_MODEL_LEN`
