from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .logging_config import configure_logging
from .schemas import HealthResponse, TranslationRequest, TranslationResponse
from .translator import TranslatorEngine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    translator = TranslatorEngine(settings)
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


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: Request, payload: TranslationRequest) -> TranslationResponse:
    translator: TranslatorEngine = request.app.state.translator
    try:
        return await run_in_threadpool(translator.translate, payload)
    except Exception as exc:
        logger.exception("Translation request failed.")
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc
