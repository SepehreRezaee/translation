MODEL_ID ?= google/translategemma-4b-it
MODEL_DIR ?= models/sharifsetup-translate
IMAGE ?= gemma-translator:offline

.PHONY: download-model build run up

download-model:
	python scripts/download_model.py --model-id "$(MODEL_ID)" --output-dir "$(MODEL_DIR)" --hf-token "$$HF_TOKEN"

build:
	docker build -t "$(IMAGE)" .

run:
	docker run --gpus all --rm -p 8000:8000 -e MODEL_PATH=/app/models/sharifsetup-translate -e MODEL_DISPLAY_NAME=Sharifsetup-Translator -e MODEL_DEVICE="$${MODEL_DEVICE:-auto}" -e VERBOSE_LOGS="$${VERBOSE_LOGS:-false}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "$(IMAGE)"

up:
	docker compose up --build
