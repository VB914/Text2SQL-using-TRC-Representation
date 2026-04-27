from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.logging import setup_logging
from services.routes import router


settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(
    title="TRC Text2SQL Research Demo",
    description="Structured prompting pipeline: natural language to TRC, validation, SQL, and execution.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
