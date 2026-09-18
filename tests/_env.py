"""Minimal .env loader for local test runs only.

Avoids a runtime dependency. Render and Docker supply real environment
variables, so nothing in app/ ever reads a .env file.
"""
from __future__ import annotations

import os


def load(path: str = None) -> list:
    path = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    found = []
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v:
                os.environ.setdefault(k, v)
                found.append(k)
    return found
