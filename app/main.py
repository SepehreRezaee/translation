from contextlib import asynccontextmanager
import importlib
from io import BytesIO
import json
import logging
from pathlib import Path
from typing import Any, List, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .languages import format_language, normalize_language
from .logging_config import configure_logging
from .schemas import (
    FileTranslationResponse,
    HealthResponse,
    TextTranslationRequest,
    TextTranslationResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    translator_module = importlib.import_module("app.translator")
    translator_cls = getattr(translator_module, "TranslatorEngine", None)
    if translator_cls is None:
        raise RuntimeError("TranslatorEngine symbol not found in app.translator")
    translator = translator_cls(settings)
    app.state.translator = translator
    app.state.settings = settings
    yield


settings = get_settings()
configure_logging(settings.verbose_logs)
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
)


def _decode_text_file(content: bytes) -> str:
    if not content:
        raise ValueError("Uploaded file is empty.")

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to decode text file with supported encodings.")


def _extract_text_from_upload(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension == ".txt":
        text = _decode_text_file(content)
    elif extension == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise ValueError("Missing PDF parser dependency. Install 'pypdf'.") from exc

        reader = PdfReader(BytesIO(content))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif extension == ".docx":
        try:
            from docx import Document
        except Exception as exc:
            raise ValueError("Missing DOCX parser dependency. Install 'python-docx'.") from exc

        document = Document(BytesIO(content))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    else:
        raise ValueError("Unsupported file type. Allowed extensions: .pdf, .docx, .txt")

    if not text or not text.strip():
        raise ValueError("No extractable text found in uploaded file.")
    return text


def _normalize_languages(language_input: Any) -> List[Tuple[str, str]]:
    raw_values: List[str] = []
    if isinstance(language_input, str):
        candidates = [part.strip() for part in language_input.split(",")]
        raw_values.extend([item for item in candidates if item])
    elif isinstance(language_input, list):
        for item in language_input:
            if isinstance(item, str):
                candidates = [part.strip() for part in item.split(",")]
                raw_values.extend([candidate for candidate in candidates if candidate])
    else:
        raise ValueError("language must be a string or list of strings.")

    if not raw_values:
        raise ValueError("language must contain at least one value.")

    normalized: List[Tuple[str, str]] = []
    seen = set()
    for value in raw_values:
        code, _ = normalize_language(value)
        formatted = format_language(code)
        if code not in seen:
            seen.add(code)
            normalized.append((code, formatted))

    return normalized


async def _parse_text_translation_request(request: Request) -> TextTranslationRequest:
    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        language_values = form.getlist("language")
        if not language_values:
            language_single = form.get("language")
            language_values = [language_single] if isinstance(language_single, str) else []
        payload_data = {
            "content": form.get("content", ""),
            "language": language_values,
        }
    else:
        raw_body = await request.body()
        if not raw_body:
            raise HTTPException(status_code=400, detail="Request body is empty.")

        try:
            payload_data = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            try:
                payload_data = json.loads(raw_body.decode("utf-8"), strict=False)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid JSON body. If your text contains unescaped control characters, "
                        "either escape them (for example '\\n') or send multipart form data."
                    ),
                ) from exc

    if "text" in payload_data and "content" not in payload_data:
        payload_data["content"] = payload_data.get("text")
    if "target_language" in payload_data and "language" not in payload_data:
        payload_data["language"] = payload_data.get("target_language")
    if isinstance(payload_data.get("language"), str):
        payload_data["language"] = [payload_data["language"]]

    try:
        return TextTranslationRequest.model_validate(payload_data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    translator = getattr(request.app.state, "translator", None)
    settings = request.app.state.settings
    return HealthResponse(
        status="ok" if translator is not None else "degraded",
        model_loaded=translator is not None,
        model=translator.model_name if translator is not None else settings.model_display_name,
        model_path=settings.model_path,
    )


@app.post(
    "/translate/text",
    response_model=TextTranslationResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["content", "language"],
                        "properties": {
                            "content": {"type": "string", "example": "Hello world"},
                            "language": {
                                "type": "array",
                                "items": {"type": "string"},
                                "example": ["fr (French)", "de"],
                            },
                        },
                    }
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["content", "language"],
                        "properties": {
                            "content": {"type": "string"},
                            "language": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    }
                },
            },
        }
    },
)
async def translate_text(request: Request) -> TextTranslationResponse:
    translator = request.app.state.translator
    try:
        payload = await _parse_text_translation_request(request)
        languages = _normalize_languages(payload.language)
        translations: List[dict] = []
        for target_code, target_label in languages:
            translated_text = await run_in_threadpool(
                translator.translate_single_text, payload.content, target_code
            )
            translations.append(
                {
                    "target_lang_code": target_code,
                    "language": target_label,
                    "translated_text": translated_text,
                }
            )
        return TextTranslationResponse(
            model=translator.model_name,
            source_language="auto",
            content=payload.content,
            translations=translations,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Text translation request failed.")
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc


@app.post("/translate/file", response_model=FileTranslationResponse)
async def translate_file(
    request: Request,
    file: UploadFile = File(...),
    language: List[str] = Form(...),
) -> FileTranslationResponse:
    translator = request.app.state.translator
    filename = file.filename or "uploaded_file"

    try:
        languages = _normalize_languages(language)
        file_bytes = await file.read()
        text = _extract_text_from_upload(filename, file_bytes)
        translations: List[dict] = []
        for target_code, target_label in languages:
            translated_text = await run_in_threadpool(
                translator.translate_single_text, text, target_code
            )
            translations.append(
                {
                    "target_lang_code": target_code,
                    "language": target_label,
                    "translated_text": translated_text,
                }
            )
        return FileTranslationResponse(
            model=translator.model_name,
            source_language="auto",
            filename=filename,
            translations=translations,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("File translation request failed.")
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc
