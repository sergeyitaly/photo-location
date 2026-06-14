"""FastAPI application entry point"""
import logging
import os
from pathlib import Path

# Avoid tokenizer multiprocessing deadlocks after fork (uvicorn --reload / WatchFiles).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.routes import router
from app.api.gazetteer_routes import router as gazetteer_router
from app.inference.model_warmup import warmup_torch_models
from app.services.llm_detective import warmup_ollama_model
from app.services.gazetteer_autoload import start_gazetteer_autoload_background

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=(
        "Photo geolocation for worldwide use: neural fusion (GeoCLIP / StreetCLIP / CLIP-ZS), "
        "optional Wikipedia + DEM validation, OSM Nominatim naming at any coordinate. "
        "Self-host heavy dependencies (Nominatim, models) at scale."
    ),
    version=settings.app_version,
    debug=settings.debug,
)

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)
app.include_router(gazetteer_router)

# Serve static frontend files
frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    logger.info(f"Mounted frontend static files from {frontend_path}")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    """Browsers request `/favicon.ico` by default — return SVG (modern browsers accept it)."""
    fav = frontend_path / "favicon.svg"
    if frontend_path.exists() and fav.is_file():
        return FileResponse(fav, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="favicon not found")


@app.on_event("startup")
async def startup_event():
    """Initialize on application startup"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Model device: {settings.model_device}")
    try:
        report = warmup_torch_models(settings)
        if report.get("preload_requested") and not report.get("skipped"):
            logger.info(
                "Torch warmup: clip_base=%s smoke=%s geoclip=%s streetclip=%s warnings=%s",
                report.get("clip_base_loaded"),
                report.get("clip_smoke_ok"),
                report.get("geoclip_loaded"),
                report.get("streetclip_loaded"),
                len(report.get("warnings") or []),
            )
    except Exception as e:
        logger.warning("Torch warmup crashed (non-fatal): %s", e, exc_info=True)

    try:
        ollama_report = await warmup_ollama_model(settings)
        if ollama_report.get("ok"):
            logger.info("Ollama warmup: model %s loaded", ollama_report.get("model"))
        elif ollama_report.get("skipped"):
            logger.debug("Ollama warmup skipped: %s", ollama_report.get("skipped"))
        elif ollama_report.get("error"):
            logger.warning("Ollama warmup failed: %s", ollama_report.get("error"))
    except Exception as e:
        logger.warning("Ollama warmup crashed (non-fatal): %s", e, exc_info=True)

    try:
        start_gazetteer_autoload_background(settings)
    except Exception as e:
        logger.warning("Gazetteer autoload launcher failed (non-fatal): %s", e, exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down application")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the web UI (same app as /static/index.html)."""
    index = frontend_path / "index.html"
    if frontend_path.exists() and index.is_file():
        return FileResponse(index, media_type="text/html")
    return {
        "message": "Photo Geolocation System API",
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc",
        "ui": "/static/index.html",
    }


@app.get("/api", include_in_schema=False)
async def api_meta():
    """JSON discovery for tools; the browser UI is at ``/``."""
    return {
        "message": "Photo Geolocation System API",
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc",
        "ui": "/",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
