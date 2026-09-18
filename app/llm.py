"""Provider-agnostic LLM client.

Two providers on separate free quotas. A 429 or an outage on the primary is
the single failure most likely to cost us a hidden case, so the secondary is
tried automatically before we give up. Keys come from the environment only -
never a file in the repo, never a log line, never the API response.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import httpx

log = logging.getLogger("gridwise.llm")

DEFAULT_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))


class LLMUnavailable(RuntimeError):
    """Every configured provider failed. Caller must fail safe, not crash."""


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    api_key: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)


def _providers() -> List[Provider]:
    out = []
    primary = Provider(
        name=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b").strip(),
        api_key=os.getenv("LLM_API_KEY", "").strip(),
    )
    if primary.configured:
        out.append(primary)
    fallback = Provider(
        name=os.getenv("LLM_FALLBACK_PROVIDER", "gemini").strip().lower(),
        model=os.getenv("LLM_FALLBACK_MODEL", "gemini-2.5-flash").strip(),
        api_key=os.getenv("LLM_FALLBACK_API_KEY", "").strip(),
    )
    if fallback.configured:
        out.append(fallback)
    return out


def providers_configured() -> List[str]:
    """Names only - used by /health-style introspection. Never returns keys."""
    return [f"{p.name}:{p.model}" for p in _providers()]


async def _call_openai_compatible(
    p: Provider, base_url: str, system: str, user: str, timeout: float
) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {p.api_key}", "Content-Type": "application/json"},
            json={
                "model": p.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _call_gemini(p: Provider, system: str, user: str, timeout: float) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{p.model}:generateContent"
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            url,
            headers={"x-goog-api-key": p.api_key, "Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            },
        )
        r.raise_for_status()
        body = r.json()
        return body["candidates"][0]["content"]["parts"][0]["text"]


async def complete_json(system: str, user: str, timeout: Optional[float] = None) -> str:
    """Return raw model text (expected to be JSON). Tries each provider in turn.

    Raises LLMUnavailable only when every provider failed; the caller turns
    that into a safe all-no_op interpretation rather than a 500.
    """
    timeout = timeout or DEFAULT_TIMEOUT_S
    provs = _providers()
    if not provs:
        raise LLMUnavailable("no LLM provider configured (set LLM_API_KEY)")

    last = None
    for p in provs:
        try:
            if p.name in ("groq",):
                return await _call_openai_compatible(
                    p, "https://api.groq.com/openai/v1", system, user, timeout
                )
            if p.name in ("openai",):
                return await _call_openai_compatible(
                    p, "https://api.openai.com/v1", system, user, timeout
                )
            if p.name in ("gemini", "google"):
                return await _call_gemini(p, system, user, timeout)
            log.error("unknown LLM_PROVIDER '%s'", p.name)
        except httpx.HTTPStatusError as exc:
            # Status only. The body can echo prompt content; the key is in the
            # request headers. Neither goes to the log.
            last = exc
            log.warning("provider %s returned HTTP %s", p.name, exc.response.status_code)
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("provider %s failed: %s", p.name, type(exc).__name__)

    raise LLMUnavailable(f"all providers failed ({type(last).__name__ if last else 'none'})")
