from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .config import Settings
from .schemas import TranslationItem, TranslationRequest, TranslationResponse


class TranslatorEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_path = Path(settings.model_path).resolve()
        self._validate_model_path()

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=settings.trust_remote_code,
            local_files_only=True,
        )
        self.llm = LLM(
            model=str(self.model_path),
            tokenizer=str(self.model_path),
            tensor_parallel_size=settings.tensor_parallel_size,
            dtype=settings.dtype,
            max_model_len=settings.max_model_len,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            enforce_eager=settings.enforce_eager,
            swap_space=settings.swap_space,
            trust_remote_code=settings.trust_remote_code,
        )
        self.model_name = settings.model_display_name

    def _validate_model_path(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {self.model_path}. Run scripts/download_model.py first."
            )

        if not (self.model_path / "config.json").exists():
            raise FileNotFoundError(
                f"Missing config.json in {self.model_path}. This directory is not a valid Hugging Face model snapshot."
            )

    def _build_system_prompt(
        self,
        source_language: str,
        target_language: str,
        preserve_formatting: bool,
        glossary: Optional[Dict[str, str]],
    ) -> str:
        lines = [
            "You are a high-precision professional translator.",
            "Return only the translated text.",
            "Do not include explanations, transliteration, metadata, or extra notes.",
            f"Target language: {target_language}.",
        ]

        if source_language.strip().lower() == "auto":
            lines.append("Detect the source language automatically.")
        else:
            lines.append(f"Source language: {source_language}.")

        if preserve_formatting:
            lines.append("Preserve line breaks, punctuation, and list/paragraph structure.")

        if glossary:
            lines.append("Mandatory glossary mappings:")
            for src_term, tgt_term in glossary.items():
                lines.append(f"- {src_term} => {tgt_term}")

        return "\n".join(lines)

    def _build_prompt(
        self,
        text: str,
        source_language: str,
        target_language: str,
        preserve_formatting: bool,
        glossary: Optional[Dict[str, str]],
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(
                    source_language=source_language,
                    target_language=target_language,
                    preserve_formatting=preserve_formatting,
                    glossary=glossary,
                ),
            },
            {"role": "user", "content": text},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        return (
            f"{messages[0]['content']}\n\n"
            f"Translate this text:\n{text}\n\n"
            "Translated text:"
        )

    @staticmethod
    def _clean_translation(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()
        return cleaned

    def translate(self, payload: TranslationRequest) -> TranslationResponse:
        prompts: List[str] = []
        indexes: List[int] = []
        translated_texts: List[str] = [""] * len(payload.texts)

        for idx, text in enumerate(payload.texts):
            if text.strip():
                prompts.append(
                    self._build_prompt(
                        text=text,
                        source_language=payload.source_language,
                        target_language=payload.target_language,
                        preserve_formatting=payload.preserve_formatting,
                        glossary=payload.glossary,
                    )
                )
                indexes.append(idx)

        if prompts:
            sampling_params = SamplingParams(
                temperature=(
                    payload.temperature
                    if payload.temperature is not None
                    else self.settings.default_temperature
                ),
                top_p=payload.top_p if payload.top_p is not None else self.settings.default_top_p,
                max_tokens=(
                    payload.max_new_tokens
                    if payload.max_new_tokens is not None
                    else self.settings.default_max_new_tokens
                ),
                repetition_penalty=(
                    payload.repetition_penalty
                    if payload.repetition_penalty is not None
                    else self.settings.default_repetition_penalty
                ),
            )
            outputs = self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

            for output_idx, output in enumerate(outputs):
                original_idx = indexes[output_idx]
                candidate = ""
                if output.outputs:
                    candidate = output.outputs[0].text
                translated_texts[original_idx] = self._clean_translation(candidate)

        items = [
            TranslationItem(source_text=source, translated_text=translated)
            for source, translated in zip(payload.texts, translated_texts)
        ]
        return TranslationResponse(
            model=self.model_name,
            source_language=payload.source_language,
            target_language=payload.target_language,
            translations=items,
        )
