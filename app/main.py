"""GridWise LLM - BUP CSE Fest 2026 Preliminary.

Single HTTP API service.

Pipeline:
    Energy Data + Operator Notes
        -> LLM Interpreter            (stage 5)
        -> Guardrail Validator        (stage 5)
        -> Math Optimizer             (stage 4)
        -> Final Validator / replay   (stage 4)
        -> API Response

Stage 3 (current): API contract locked down. /optimize-energy returns a
schema-valid, energy-rule-valid grid-only baseline with every note marked
no_op. The interpreter and optimizer replace those two pieces in place.
"""

import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .baseline import build_grid_only_plan, summarize
from .schemas import DirectiveInterpretation, OptimizeRequest, OptimizeResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("gridwise")

APP_VERSION = "0.3.0"

app = FastAPI(
    title="GridWise LLM",
    version=APP_VERSION,
    description="LLM-assisted campus energy directive interpretation and 24h cost optimization.",
)


@app.middleware("http")
async def timing_and_error_guard(request: Request, call_next):
    """Never crash the process; always return JSON. Logs latency against the
    30 s per-request budget. No request body or secret is ever logged."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - Section 08 SAFE FAILURE
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "Unhandled server error."},
        )
    elapsed = (time.perf_counter() - start) * 1000
    log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """Section 6.1: 400 for malformed JSON or a structurally invalid request.
    FastAPI's default is 422, which the spec lists only as OPTIONAL, so we
    return the code the spec actually requires."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_request",
            "detail": [
                {"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()
            ],
        },
    )


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "service": "gridwise-llm",
        "version": APP_VERSION,
        "endpoints": {"health": "GET /health", "optimize": "POST /optimize-energy"},
    }


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(req: OptimizeRequest):
    hours = req.hours_in_order()

    # --- STAGE 3 PLACEHOLDER ------------------------------------------------
    # Replaced by the LLM interpreter + guardrails in stage 5. Marking every
    # note no_op is schema-valid but scores 0 on interpretation; it must not
    # survive to submission.
    interpretation = [
        DirectiveInterpretation(
            note_index=i,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation="Interpreter not yet wired; placeholder response.",
        )
        for i in range(len(req.operator_notes))
    ]
    # --- END PLACEHOLDER ----------------------------------------------------

    plan = build_grid_only_plan(hours, req.battery)
    total_grid, total_cost, peak_grid = summarize(plan, hours)

    return OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=interpretation,
        hourly_plan=plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        plan_summary=(
            "Baseline grid-only schedule: solar is consumed on site up to demand each "
            "hour and the remainder is imported from the grid; the battery idles, so "
            "it ends the day at its starting energy."
        ),
    )
