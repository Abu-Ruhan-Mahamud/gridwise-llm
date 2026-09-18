"""Provider-agnostic LLM client with a hard time budget.

Provider order matters. Measured free-tier limits:
  Gemini (gemini-3.6-flash)  generous token budget -> PRIMARY
  Groq   (openai/gpt-oss-120b)  8,000 tokens/minute, and gpt-oss spends extra
         tokens on hidden reasoning, so a prompt of this size exhausts the
         minute in ~4 calls -> FALLBACK only

Two providers on separate quotas means a 429 on one does not cost us a case.
Everything is bounded by a deadline so the 30 s per-request limit cannot be
blown by retries.

Keys come from the environment only. They are never logged, never echoed in a
response, and never written to disk by this module.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx

log = logging.getLogger("gridwise.llm")

PER_CALL_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))
TOTAL_BUDGET_S = float(os.getenv("LLM_BUDGET_SECONDS", "20"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1200"))
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMUnavailable(RuntimeError):
    """Every configured provider failed. Caller must fail safe, not crash."""


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    api_key: str
    attempts: int

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)


def _chain(raw: str) -> List[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def _providers() -> List[Provider]:
    """Resolve an ordered chain of (provider, model) links.

    Free-tier quotas are metered PER MODEL, so naming several models builds
    several independent quota buckets rather than one. Measured on this
    account:

        groq   openai/gpt-oss-120b          8,000 tok/min, 1,000 req/day
        groq   openai/gpt-oss-20b           8,000 tok/min, 1,000 req/day
        groq   qwen/qwen3.8-27b             8,000 tok/min, 1,000 req/day
        groq   openai/gpt-oss-safeguard-20b 8,000 tok/min, 1,000 req/day
        gemini gemini-3.1-flash-lite        separate daily bucket
        gemini gemini-3.6-flash             20 req/day  <- last resort only

    Six links, roughly 32k tokens/minute combined. Each added model was first
    checked against the traps that actually broke this system tonight (the
    'one until three' daytime reading, 'charger isolated', and an inverted
    solar phrasing): a model that interprets badly would be worse than no
    model, since it is reached exactly when the good ones are throttled.

    Groq leads on request budget by two orders of magnitude, so it goes first.
    Gemini trails as genuine redundancy: a Groq-wide outage or token-rate
    exhaustion still leaves a working path.

    Keys are named per provider, so there is no primary/fallback slot to put a
    key into wrongly. The legacy generic LLM_* scheme is honoured only when no
    named key exists at all, so a stale dashboard variable can never override
    a correctly named key.
    """
    out: List[Provider] = []

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if groq_key:
        for model in _chain(os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-120b,openai/gpt-oss-20b,"
                "qwen/qwen3.8-27b,openai/gpt-oss-safeguard-20b",
            )):
            out.append(Provider("groq", model, groq_key, attempts=2 if not out else 1))
    if gemini_key:
        for model in _chain(os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite,gemini-3.6-flash")):
            out.append(Provider("gemini", model, gemini_key, attempts=1))
    if out:
        return out

    primary = Provider(
        name=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b").strip(),
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        attempts=2,
    )
    if primary.configured:
        out.append(primary)
    fallback = Provider(
        name=os.getenv("LLM_FALLBACK_PROVIDER", "gemini").strip().lower(),
        model=os.getenv("LLM_FALLBACK_MODEL", "gemini-3.1-flash-lite").strip(),
        api_key=os.getenv("LLM_FALLBACK_API_KEY", "").strip(),
        attempts=1,
    )
    if fallback.configured:
        out.append(fallback)
    return out


def providers_configured() -> List[str]:
    """Names only - safe to expose. Never returns key material."""
    return [f"{p.name}:{p.model}" for p in _providers()]


async def _post(client: httpx.AsyncClient, url: str, headers: dict, payload: dict) -> httpx.Response:
    r = await client.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r


async def _call(p: Provider, system: str, user: str, timeout: float) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        if p.name in ("gemini", "google"):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{p.model}:generateContent"
            r = await _post(
                client,
                url,
                {"x-goog-api-key": p.api_key, "Content-Type": "application/json"},
                {
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {
                        "temperature": 0,
                        "responseMimeType": "application/json",
                        "maxOutputTokens": MAX_OUTPUT_TOKENS,
                        # Measured: extended thinking costs 6-11 s per call on
                        # this model and buys nothing on a bounded extraction
                        # task. Disabling it takes the call to ~2 s, which is
                        # most of our margin against the 30 s limit.
                        "thinkingConfig": {"thinkingBudget": 0},
                    },
                },
            )
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts)

        base = {
            "groq": "https://api.groq.com/openai/v1",
            "openai": "https://api.openai.com/v1",
        }.get(p.name)
        if not base:
            raise ValueError(f"unknown provider '{p.name}'")
        payload = {
            "model": p.model,
            "temperature": 0,
            # temperature 0 alone is not reproducible on these endpoints; a
            # fixed seed materially reduces run-to-run variance on borderline
            # notes. Ignored by providers that do not support it.
            "seed": 20260918,
            "top_p": 1,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if "gpt-oss" in p.model:
            # Measured: default reasoning effort spends ~360 hidden tokens per
            # call against an 8k/min budget. "low" cuts it to ~100 with no
            # accuracy change on this task.
            payload["reasoning_effort"] = "low"
        r = await _post(
            client,
            f"{base}/chat/completions",
            {"Authorization": f"Bearer {p.api_key}", "Content-Type": "application/json"},
            payload,
        )
        return r.json()["choices"][0]["message"]["content"]


def _retry_after(exc: httpx.HTTPStatusError, default: float) -> float:
    raw = exc.response.headers.get("retry-after")
    try:
        return max(0.0, min(float(raw), 5.0))
    except (TypeError, ValueError):
        return default


async def complete_json(system: str, user: str, deadline: Optional[float] = None) -> str:
    """Return raw model text. Tries each provider, respecting a wall-clock deadline.

    Raises LLMUnavailable only when everything failed or the budget ran out;
    the caller turns that into a safe all-no_op interpretation, never a 500.
    """
    deadline = deadline or (time.monotonic() + TOTAL_BUDGET_S)
    provs = _providers()
    if not provs:
        raise LLMUnavailable("no LLM provider configured (set LLM_API_KEY)")

    last = "none"
    for p in provs:
        for attempt in range(1, p.attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                log.warning("LLM budget exhausted before %s", p.name)
                raise LLMUnavailable(f"budget exhausted (last: {last})")
            try:
                return await _call(p, system, user, min(PER_CALL_TIMEOUT_S, remaining))
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                last = f"{p.name} HTTP {code}"
                # Status only. Bodies can echo prompt content and headers carry
                # the key, so neither is logged.
                log.warning("provider %s returned HTTP %s (attempt %s)", p.name, code, attempt)
                if code == 429 or code not in RETRY_STATUS or attempt >= p.attempts:
                    break  # 429 here is a per-day quota; the next link has its own
                await asyncio.sleep(min(_retry_after(exc, 1.5), max(0.0, deadline - time.monotonic() - 1)))
            except Exception as exc:  # noqa: BLE001
                last = f"{p.name} {type(exc).__name__}"
                log.warning("provider %s failed: %s (attempt %s)", p.name, type(exc).__name__, attempt)
                if attempt >= p.attempts:
                    break
                await asyncio.sleep(0.5)

    raise LLMUnavailable(f"all providers failed (last: {last})")
