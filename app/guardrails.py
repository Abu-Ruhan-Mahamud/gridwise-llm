"""Deterministic guardrail layer (Section 08).

LLM output is untrusted structured data. Nothing reaches the optimizer without
passing through here. Every check below maps to a row of the Section 08 table:

  Allowed types      directive_type must be one of the six
  Note mapping       exactly one entry per note, note_index 0..N-1, no dupes
  Hours              unique ints 0..23, ascending
  Solar factor       0 <= factor <= 1
  Battery reserve    finite, non-negative, <= battery capacity
  Grid cap           finite, non-negative
  No invention       structurally impossible - only directive fields are read;
                     demand, tariff and battery values are never taken from
                     the model, they come from the request object
  applies semantics  applies == (directive_type != "no_op")

Repair-or-drop policy: anything unrecoverable degrades to no_op for that note.
A no_op scores 0 on that note but keeps the response valid, which is strictly
better than a 500 or an invented directive.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

log = logging.getLogger("gridwise.guardrails")

SUPPORTED = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}
HARD_TYPES = SUPPORTED - {"no_op"}

REQUIRED_KEYS = {
    "solar_reduction": ("hours", "factor"),
    "minimum_battery_reserve": ("hours", "minimum_energy_kwh"),
    "no_charge_window": ("hours",),
    "no_discharge_window": ("hours",),
    "max_grid_window": ("hours", "max_grid_kwh"),
}


def _no_op(note_index: int, why: str) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": why,
    }


def _clean_hours(raw: Any) -> List[int] | None:
    """Unique integers 0-23 in ascending order, or None if unusable."""
    if not isinstance(raw, (list, tuple)):
        return None
    out = set()
    for h in raw:
        if isinstance(h, bool):
            return None
        if isinstance(h, float) and not float(h).is_integer():
            return None
        try:
            hi = int(h)
        except (TypeError, ValueError):
            return None
        if 0 <= hi <= 23:
            out.add(hi)
        else:
            return None  # out-of-range hour means the window was misread
    return sorted(out) or None


def _finite_number(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def enforce(
    entries: Any, note_count: int, capacity_kwh: float
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (exactly note_count normalized entries in note_index order, repairs)."""
    repairs: List[str] = []
    by_index: Dict[int, Dict[str, Any]] = {}

    if not isinstance(entries, list):
        repairs.append("LLM output was not a list of directives")
        entries = []

    for item in entries:
        if not isinstance(item, dict):
            repairs.append("dropped a non-object entry")
            continue

        ni = item.get("note_index")
        if isinstance(ni, bool) or not isinstance(ni, (int, float)) or int(ni) != ni:
            repairs.append(f"dropped entry with non-integer note_index {ni!r}")
            continue
        ni = int(ni)
        if not 0 <= ni < note_count:
            repairs.append(f"dropped entry with out-of-range note_index {ni}")
            continue
        if ni in by_index:
            repairs.append(f"dropped duplicate entry for note_index {ni}")
            continue

        dtype = item.get("directive_type")
        explanation = item.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = "Interpreted from the operator note."

        if dtype not in SUPPORTED:
            repairs.append(f"note {ni}: unsupported directive_type {dtype!r} -> no_op")
            by_index[ni] = _no_op(ni, "Note did not map to a supported directive type.")
            continue

        if dtype == "no_op":
            by_index[ni] = _no_op(ni, explanation)
            continue

        adj = item.get("structured_adjustment")
        if not isinstance(adj, dict):
            repairs.append(f"note {ni}: {dtype} missing structured_adjustment -> no_op")
            by_index[ni] = _no_op(ni, "Directive was missing its structured adjustment.")
            continue

        hours = _clean_hours(adj.get("hours"))
        if hours is None:
            repairs.append(f"note {ni}: {dtype} had unusable hours {adj.get('hours')!r} -> no_op")
            by_index[ni] = _no_op(ni, "Directive hours could not be validated.")
            continue

        clean: Dict[str, Any] = {"hours": hours}

        if dtype == "solar_reduction":
            factor = _finite_number(adj.get("factor"))
            if factor is None or not 0.0 <= factor <= 1.0:
                repairs.append(f"note {ni}: solar factor {adj.get('factor')!r} outside [0,1] -> no_op")
                by_index[ni] = _no_op(ni, "Solar reduction factor failed validation.")
                continue
            clean["factor"] = factor

        elif dtype == "minimum_battery_reserve":
            val = _finite_number(adj.get("minimum_energy_kwh"))
            if val is None or val < 0:
                repairs.append(f"note {ni}: reserve {adj.get('minimum_energy_kwh')!r} invalid -> no_op")
                by_index[ni] = _no_op(ni, "Reserve value failed validation.")
                continue
            if val > capacity_kwh:
                # Section 08 forbids a reserve above capacity. Clamping keeps the
                # operator's intent (hold the battery high) where dropping the
                # directive entirely would fail the application check outright.
                repairs.append(f"note {ni}: reserve {val} > capacity {capacity_kwh}, clamped")
                val = capacity_kwh
            clean["minimum_energy_kwh"] = val

        elif dtype == "max_grid_window":
            val = _finite_number(adj.get("max_grid_kwh"))
            if val is None or val < 0:
                repairs.append(f"note {ni}: grid cap {adj.get('max_grid_kwh')!r} invalid -> no_op")
                by_index[ni] = _no_op(ni, "Grid cap failed validation.")
                continue
            clean["max_grid_kwh"] = val

        # Any extra keys the model invented are dropped here by construction:
        # `clean` is rebuilt from scratch, never copied from the model output.
        missing = [k for k in REQUIRED_KEYS[dtype] if k not in clean]
        if missing:
            repairs.append(f"note {ni}: {dtype} missing {missing} -> no_op")
            by_index[ni] = _no_op(ni, "Directive was structurally incomplete.")
            continue

        by_index[ni] = {
            "note_index": ni,
            "applies": True,  # Section 05: every non-no_op directive applies
            "directive_type": dtype,
            "structured_adjustment": clean,
            "explanation": explanation.strip(),
        }

    result = []
    for i in range(note_count):
        if i not in by_index:
            repairs.append(f"note {i}: no entry returned -> no_op")
            result.append(_no_op(i, "No directive was produced for this note."))
        else:
            result.append(by_index[i])

    if repairs:
        log.warning("guardrail repairs: %s", repairs)
    return result, repairs
