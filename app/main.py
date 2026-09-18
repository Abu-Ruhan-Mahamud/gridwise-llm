"""GridWise LLM - BUP CSE Fest 2026 Preliminary.

Single HTTP API service. Stage 1: health endpoint only.
Pipeline (added in later stages):
    Energy Data + Operator Notes -> LLM Interpreter -> Guardrail Validator
    -> Math Optimizer -> Final Validator -> API Response
"""

import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("gridwise")

APP_VERSION = "0.1.0"

app = FastAPI(
    title="GridWise LLM",
    version=APP_VERSION,
    description="LLM-assisted campus energy directive interpretation and 24h cost optimization.",
)


@app.middleware("http")
async def timing_and_error_guard(request: Request, call_next):
    """Never crash the process; always return JSON. Log latency for the <30s budget."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - fail safe, controlled error
        elapsed = (time.perf_counter() - start) * 1000
        log.exception("unhandled error on %s %s (%.0f ms)", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "Unhandled server error."},
        )
    elapsed = (time.perf_counter() - start) * 1000
    log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"service": "gridwise-llm", "version": APP_VERSION, "health": "/health"}
