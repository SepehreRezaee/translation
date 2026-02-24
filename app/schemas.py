from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TranslationRequest(BaseModel):
    source_language: str = Field(
        default="auto",
        min_length=2,
        description="Source language name or code. Use 'auto' for automatic detection.",
    )
    target_language: str = Field(
        ...,
        min_length=2,
        description="Target language name or ISO code (e.g., 'Spanish', 'es').",
    )
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="One or more texts to translate in a single request.",
    )
    preserve_formatting: bool = Field(
        default=True,
        description="Preserve line breaks, punctuation, and original text structure.",
    )
    glossary: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional term mapping (source term -> required target term).",
    )
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    max_new_tokens: Optional[int] = Field(default=None, ge=16, le=2048)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.8, le=2.0)

    @field_validator("texts")
    @classmethod
    def ensure_not_all_empty(cls, texts: List[str]) -> List[str]:
        if not any(text.strip() for text in texts):
            raise ValueError("At least one text must contain non-whitespace characters.")
        return texts


class TranslationItem(BaseModel):
    source_text: str
    translated_text: str


class TranslationResponse(BaseModel):
    model: str
    source_language: str
    target_language: str
    translations: List[TranslationItem]


class TextTranslationRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Input text to translate.")
    language: str = Field(
        ...,
        description=(
            "Target language as code, name, or formatted value "
            "(e.g., 'fr', 'French', 'fr (French)')."
        ),
    )

    @field_validator("content")
    @classmethod
    def ensure_text_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must contain non-whitespace characters.")
        return value

    @field_validator("language")
    @classmethod
    def ensure_language(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("language must be a non-empty string.")
        return normalized


class LanguageTranslationItem(BaseModel):
    target_lang_code: str
    language: str
    translated_text: str


class TextTranslationResponse(BaseModel):
    model: str
    source_language: str
    content: str
    translations: List[LanguageTranslationItem]


class FileTranslationResponse(BaseModel):
    model: str
    source_language: str
    filename: str
    translations: List[LanguageTranslationItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str
    model_path: str
