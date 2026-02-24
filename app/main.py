from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .schemas import HealthResponse, TranslationRequest, TranslationResponse
from .translator import TranslatorEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    translator = TranslatorEngine(settings)
    app.state.translator = translator
    app.state.settings = settings
    yield


settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    translator = getattr(request.app.state, "translator", None)
    model_path = request.app.state.settings.model_path
    return HealthResponse(
        status="ok" if translator is not None else "degraded",
        model_loaded=translator is not None,
        model_path=model_path,
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: Request, payload: TranslationRequest) -> TranslationResponse:
    translator: TranslatorEngine = request.app.state.translator
    try:
        return await run_in_threadpool(translator.translate, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc

