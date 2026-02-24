MODEL_ID ?= google/translategemma-4b-it
MODEL_DIR ?= models/translate-gemma
IMAGE ?= gemma-translator:offline

.PHONY: download-model build run up

download-model:
	python scripts/download_model.py --model-id "$(MODEL_ID)" --output-dir "$(MODEL_DIR)" --hf-token "$$HF_TOKEN"

build:
	docker build -t "$(IMAGE)" .

run:
	docker run --gpus all --rm -p 8000:8000 -e MODEL_PATH=/app/models/translate-gemma -e MODEL_DISPLAY_NAME=Sharifsetup-Translator -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "$(IMAGE)"

up:
	docker compose up --build
