import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import Settings
from .languages import LANGUAGES, normalize_language

logger = logging.getLogger(__name__)


class TranslatorEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_path = Path(settings.model_path).resolve()
        self._validate_model_path()

        self.torch, auto_model_cls, auto_processor_cls = self._load_transformer_backends()
        self.device = self._resolve_device(settings.model_device)
        self.dtype = self._resolve_dtype(settings.dtype, self.device)

        processor_kwargs: Dict[str, Any] = {
            "trust_remote_code": settings.trust_remote_code,
            "local_files_only": True,
            "fix_mistral_regex": True,
        }
        try:
            self.processor = auto_processor_cls.from_pretrained(
                str(self.model_path), **processor_kwargs
            )
        except TypeError:
            processor_kwargs.pop("fix_mistral_regex", None)
            self.processor = auto_processor_cls.from_pretrained(
                str(self.model_path), **processor_kwargs
            )

        model_kwargs: Dict[str, Any] = {
            "dtype": self.dtype,
            "trust_remote_code": settings.trust_remote_code,
            "local_files_only": True,
        }
        try:
            self.model = auto_model_cls.from_pretrained(str(self.model_path), **model_kwargs)
        except TypeError:
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
            self.model = auto_model_cls.from_pretrained(str(self.model_path), **fallback_kwargs)

        self.model.to(self.device)
        self.model.eval()
        self.model_name = settings.model_display_name

        logger.info(
            "Loaded model '%s' on device '%s' with dtype '%s'.",
            self.model_name,
            self.device,
            self.dtype,
        )

    @staticmethod
    def _load_transformer_backends() -> Tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Missing runtime dependencies. Install with: pip install -r requirements.txt"
            ) from exc

        return torch, AutoModelForImageTextToText, AutoProcessor

    def _validate_model_path(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {self.model_path}. Run scripts/download_model.py first."
            )

        if not (self.model_path / "config.json").exists():
            raise FileNotFoundError(
                f"Missing config.json in {self.model_path}. This directory is not a valid Hugging Face model snapshot."
            )

    def _resolve_device(self, model_device: str) -> str:
        device = model_device.strip().lower()
        if device == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"

        if device.startswith("cuda") and not self.torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"

        return model_device

    def _resolve_dtype(self, dtype_name: str, device: str) -> Any:
        normalized = dtype_name.strip().lower()
        mapping = {
            "float16": self.torch.float16,
            "fp16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "bf16": self.torch.bfloat16,
            "float32": self.torch.float32,
            "fp32": self.torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported DTYPE value: {dtype_name}")

        resolved = mapping[normalized]
        if device == "cpu" and resolved in (self.torch.float16, self.torch.bfloat16):
            logger.warning("Using float32 on CPU even though DTYPE=%s was requested.", dtype_name)
            return self.torch.float32
        return resolved

    @staticmethod
    def _normalize_detected_code(code: str) -> Optional[str]:
        normalized = code.strip().lower()
        alias = {
            "zh-cn": "zh",
            "zh-tw": "zh",
            "pt-br": "pt",
            "iw": "he",
        }
        normalized = alias.get(normalized, normalized)
        if normalized in LANGUAGES:
            return normalized
        if "-" in normalized:
            prefix = normalized.split("-", 1)[0]
            if prefix in LANGUAGES:
                return prefix
        return None

    def _detect_source_lang_code(self, text: str) -> str:
        try:
            from langdetect import detect
        except Exception:
            return "en"

        try:
            detected = detect(text)
        except Exception:
            return "en"

        normalized = self._normalize_detected_code(detected)
        return normalized if normalized is not None else "en"

    def _generation_kwargs(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.settings.default_max_new_tokens,
            "do_sample": False,
        }

    @staticmethod
    def _clean_translation(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()
        return cleaned

    def _prepare_inputs(self, text: str, source_lang_code: str, target_lang_code: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang_code,
                        "target_lang_code": target_lang_code,
                        "text": text,
                    }
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if hasattr(inputs, "to"):
            moved = inputs.to(self.device)
            if isinstance(moved, dict):
                return moved
            return dict(moved)

        if isinstance(inputs, dict):
            return {key: value.to(self.device) for key, value in inputs.items()}

        raise RuntimeError("Unsupported processor output type.")

    def _translate_by_codes(self, text: str, source_lang_code: str, target_lang_code: str) -> str:
        if not text.strip():
            return ""

        inputs = self._prepare_inputs(text, source_lang_code, target_lang_code)
        with self.torch.inference_mode():
            generation = self.model.generate(**inputs, **self._generation_kwargs())

        input_len = inputs["input_ids"].shape[1]
        output = self.processor.decode(generation[0][input_len:], skip_special_tokens=True)
        return self._clean_translation(output)

    def translate_single_text(self, text: str, target_language: str) -> str:
        target_code, _ = normalize_language(target_language)
        source_code = self._detect_source_lang_code(text)
        return self._translate_by_codes(
            text=text,
            source_lang_code=source_code,
            target_lang_code=target_code,
        )


__all__ = ["TranslatorEngine"]

