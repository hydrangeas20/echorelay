"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .config import Settings, get_settings
from .inference import InferenceRunner, Transcriber, build_transcriber
from .protocol import PROTOCOL_VERSION
from .transport import StreamRegistry, router

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, transcriber: Transcriber | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=settings.log_level.upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        model = transcriber or build_transcriber(settings)
        # Model loading is blocking (and may download weights): keep it off the loop.
        await asyncio.to_thread(model.load)
        runner = InferenceRunner(model, settings.max_concurrent_inference, settings.inference_timeout_s)
        app.state.settings = settings
        app.state.runner = runner
        app.state.registry = StreamRegistry(settings.max_concurrent_streams)
        log.info("echorelay ready: %s", model.info)
        try:
            yield
        finally:
            runner.shutdown()

    app = FastAPI(title="EchoRelay", version=__version__, lifespan=lifespan)
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict:
        info = app.state.runner.transcriber.info
        reg = app.state.registry
        return {
            "status": "ok",
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "model": {
                "backend": info.backend,
                "name": info.model,
                "device": info.device,
                "compute_type": info.compute_type,
            },
            "streams": {
                "active": len(reg.active),
                "max": reg.max_streams,
                "completed": reg.completed,
                "failed": reg.failed,
                "disconnected": reg.disconnected,
            },
            "inference": {
                "in_flight": app.state.runner.in_flight,
                "max_concurrent": settings.max_concurrent_inference,
            },
        }

    return app
