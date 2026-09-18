"""GridWise LLM - BUP CSE Fest 2026 Preliminary.

Pipeline:
    Energy Data + Operator Notes
        -> LLM Interpreter            (stage 5 - NOT YET WIRED)
        -> Guardrail Validator        (stage 5 - NOT YET WIRED)
        -> Math Optimizer             app/optimizer.py
        -> Final Validator / replay   app/validator.py
        -> API Response
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .baseline import summarize
from .interpreter import interpret
from .llm import providers_configured
from .optimizer import solve_schedule
from .schemas import DirectiveInterpretation, OptimizeRequest, OptimizeResponse
from .validator import validate_plan

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("gridwise")

APP_VERSION = "0.5.0"


# --------------------------------------------------------------------------
# Keep-alive: a free host idles the service out after ~15 minutes, and the
# cold start that follows blows the 30 s request budget. Pinging our own
# public URL counts as inbound traffic and keeps the instance up. This is a
# SECOND layer only - it cannot wake a service that has already slept, so an
# external pinger remains the primary defence.
# --------------------------------------------------------------------------
async def _keepalive_loop(url: str, interval: int) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            await asyncio.sleep(interval)
            try:
                r = await client.get(url)
                log.debug("keepalive %s -> %s", url, r.status_code)
            except Exception as exc:  # noqa: BLE001 - never let this kill the app
                log.warning("keepalive failed: %s", type(exc).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    url = os.getenv("KEEPALIVE_URL", "").strip()
    interval = int(os.getenv("KEEPALIVE_INTERVAL_SECONDS", "600"))
    task = None
    if url:
        log.info("keepalive enabled every %ss", interval)
        task = asyncio.create_task(_keepalive_loop(url, interval))
    yield
    if task:
        task.cancel()


app = FastAPI(
    title="GridWise LLM",
    version=APP_VERSION,
    description="LLM-assisted campus energy directive interpretation and 24h cost optimization.",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing_and_error_guard(request: Request, call_next):
    """Never crash the process; always return JSON. Logs latency against the
    30 s budget. No request body, note text or secret is ever logged."""
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
    FastAPI defaults to 422, which the spec lists only as OPTIONAL."""
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
        # Provider and model identifiers only - never key material.
        "llm": providers_configured(),
    }


def _describe(plan, directives: List[Dict[str, Any]], method: str) -> str:
    """Deterministic plan_summary. The LLM is not used here - Section 02 is
    explicit that LLM-for-summary-only does not satisfy the requirement, so
    the model earns its place upstream in interpretation, not in this string."""
    charge_h = [p.hour for p in plan if p.battery_action == "charge"]
    discharge_h = [p.hour for p in plan if p.battery_action == "discharge"]
    names = sorted({d["directive_type"] for d in directives})
    bits = []
    if names:
        bits.append("Applied operator directives: " + ", ".join(names) + ".")
    else:
        bits.append("No operator directive constrained this schedule.")
    bits.append("On-site solar is consumed first each hour, up to its effective availability.")
    if charge_h or discharge_h:
        bits.append(
            f"The battery charges in hour(s) {charge_h} at lower tariffs and discharges in "
            f"hour(s) {discharge_h} at higher tariffs, returning to its starting energy by hour 23."
        )
    else:
        bits.append("The battery idles for the full horizon and ends at its starting energy.")
    if method.startswith("lp_dropped_"):
        bits.append(
            f"({method.rsplit('_', 1)[1]} directive(s) could not be satisfied together with "
            "the rest and were not applied.)"
        )
    elif method != "lp":
        bits.append(f"(Schedule produced by the {method} fallback path.)")
    return " ".join(bits)


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(req: OptimizeRequest):
    hours = req.hours_in_order()

    # --- Stage 1: LLM interpretation, then Stage 2: deterministic guardrails
    entries, meta = await interpret(req.operator_notes, req.battery.capacity_kwh)
    interpretation = [DirectiveInterpretation(**e) for e in entries]
    active: List[Dict[str, Any]] = [
        e for e in entries if e["applies"] and e["directive_type"] != "no_op"
    ]
    if not meta["llm_ok"]:
        log.error("interpretation degraded to all-no_op: %s", meta.get("error"))

    plan, method, applied = solve_schedule(hours, req.battery, active)
    if len(applied) != len(active):
        dropped = [d["directive_type"] for d in active if d not in applied]
        log.error("could not satisfy every directive; not applied: %s", dropped)
    total_grid, total_cost, peak_grid = summarize(plan, hours)

    # Section 08 final replay, at the API boundary. solve_schedule already
    # gates on this; repeating it here means nothing leaves the service
    # unchecked even if a future change bypasses the ladder.
    residual = validate_plan(
        hours,
        req.battery,
        plan,
        applied,
        {"total_grid_kwh": total_grid, "total_cost_bdt": total_cost, "peak_grid_kwh": peak_grid},
    )
    if residual:
        log.error("final replay found violations (%s): %s", method, residual[:3])

    return OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=interpretation,
        hourly_plan=plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        plan_summary=_describe(plan, applied, method),
    )
