import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import Settings
from .languages import LANGUAGES, normalize_language

logger = logging.getLogger(__name__)
_CONTROL_MARKERS = (
    "<end_of_turn>",
    "<|end_of_turn|>",
    "<eot_id>",
    "<|eot_id|>",
    "<start_of_turn>",
    "<|start_of_turn|>",
    "<bos>",
    "<eos>",
    "</s>",
)
_CONTROL_MARKER_RE = re.compile(
    r"(?:<\|?end_of_turn\|?>|<\|?eot_id\|?>|<\|?start_of_turn\|?>|<bos>|<eos>|</s>)",
    flags=re.IGNORECASE,
)


class TranslatorEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_path = Path(settings.model_path).resolve()
        self._validate_model_path()

        self.torch, auto_model_cls, auto_processor_cls = self._load_transformer_backends()
        self.device = self._resolve_device(settings.model_device)
        self.dtype = self._resolve_dtype(settings.dtype, self.device)
        self._configure_warnings()
        self.processor = self._load_processor(auto_processor_cls)
        self._force_tokenizer_regex_fix()

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

        self.pad_token_id, self.eos_token_id = self._resolve_generation_token_ids()
        self.end_of_turn_token_id = self._resolve_end_of_turn_token_id()
        self._configure_generation_token_ids()
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

    def _configure_warnings(self) -> None:
        warnings.filterwarnings(
            "ignore",
            message=r"Using a slow image processor as `use_fast` is unset.*",
        )

    def _load_processor(self, auto_processor_cls: Any) -> Any:
        base_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.settings.trust_remote_code,
            "local_files_only": True,
        }

        candidates = [
            {
                **base_kwargs,
                "use_fast": self.settings.use_fast_processor,
                "fix_mistral_regex": self.settings.fix_mistral_regex,
                "tokenizer_kwargs": {"fix_mistral_regex": self.settings.fix_mistral_regex},
            },
            {
                **base_kwargs,
                "use_fast": self.settings.use_fast_processor,
                "tokenizer_kwargs": {"fix_mistral_regex": self.settings.fix_mistral_regex},
            },
            {
                **base_kwargs,
                "use_fast": self.settings.use_fast_processor,
                "fix_mistral_regex": self.settings.fix_mistral_regex,
            },
            {
                **base_kwargs,
                "use_fast": self.settings.use_fast_processor,
            },
            base_kwargs,
        ]

        last_error: Optional[Exception] = None
        for kwargs in candidates:
            try:
                return auto_processor_cls.from_pretrained(str(self.model_path), **kwargs)
            except TypeError as exc:
                # Older/newer transformers versions may not accept some kwargs.
                last_error = exc

        if last_error is not None:
            raise RuntimeError(
                "Failed to load model processor with the current transformers version."
            ) from last_error
        raise RuntimeError("Failed to load model processor.")

    def _force_tokenizer_regex_fix(self) -> None:
        if not self.settings.fix_mistral_regex:
            return

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return

        try:
            from transformers import AutoTokenizer

            fixed_tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=self.settings.trust_remote_code,
                local_files_only=True,
                fix_mistral_regex=True,
                use_fast=self.settings.use_fast_processor,
            )
            self.processor.tokenizer = fixed_tokenizer
        except Exception:
            # Keep loaded tokenizer if fast fix path isn't supported in current transformers build.
            pass

    def _resolve_generation_token_ids(self) -> Tuple[Optional[int], Optional[int]]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None, None

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id
            try:
                tokenizer.pad_token_id = pad_token_id
            except Exception:
                pass
            try:
                if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass

        return pad_token_id, eos_token_id

    def _configure_generation_token_ids(self) -> None:
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is None:
            return

        if self.eos_token_id is not None and getattr(generation_config, "eos_token_id", None) is None:
            generation_config.eos_token_id = self.eos_token_id

        if self.pad_token_id is not None and getattr(generation_config, "pad_token_id", None) is None:
            generation_config.pad_token_id = self.pad_token_id

    def _resolve_end_of_turn_token_id(self) -> Optional[int]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None

        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        candidates = ("<end_of_turn>", "<|end_of_turn|>", "<eot_id>", "<|eot_id|>")

        for token in candidates:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None

            if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
                return token_id

        for token in candidates:
            try:
                encoded = tokenizer.encode(token, add_special_tokens=False)
            except Exception:
                encoded = None

            if isinstance(encoded, list) and len(encoded) == 1:
                token_id = encoded[0]
                if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
                    return token_id

        return None

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
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.settings.default_max_new_tokens,
            "do_sample": False,
        }
        if self.pad_token_id is not None:
            kwargs["pad_token_id"] = self.pad_token_id
        eos_ids = []
        if self.eos_token_id is not None:
            eos_ids.append(self.eos_token_id)
        if self.end_of_turn_token_id is not None and self.end_of_turn_token_id not in eos_ids:
            eos_ids.append(self.end_of_turn_token_id)
        if eos_ids:
            kwargs["eos_token_id"] = eos_ids[0] if len(eos_ids) == 1 else eos_ids
        return kwargs

    @staticmethod
    def _clean_translation(text: str) -> str:
        cleaned = text.strip()

        lowered = cleaned.lower()
        cut_positions = [lowered.find(marker) for marker in _CONTROL_MARKERS if marker in lowered]
        cut_positions = [index for index in cut_positions if index >= 0]
        if cut_positions:
            cleaned = cleaned[: min(cut_positions)].rstrip()

        cleaned = _CONTROL_MARKER_RE.sub("", cleaned).strip()
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
