"""Final Validator - replays a finished schedule against every rule.

Section 08 requires a post-optimization replay proving each extracted directive
was actually followed; Section 11.3 lists the consistency checks the judge runs.
This module implements both, so it doubles as the local test oracle and as the
service's own last gate before it returns a response.

Pure functions, no I/O, no LLM. Returns a list of human-readable violations;
empty list means the plan passes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

TOL = 0.01  # Section 11.5


def effective_solar(hours: Sequence[Any], directives: Sequence[Dict[str, Any]]) -> Dict[int, float]:
    """Base solar with every solar_reduction factor applied (Section 5.3).

    Overlapping reductions multiply, which keeps the result order-independent.
    """
    eff = {h.hour: float(h.solar_kwh) for h in hours}
    for d in directives:
        if d.get("directive_type") != "solar_reduction":
            continue
        adj = d.get("structured_adjustment") or {}
        factor = float(adj.get("factor", 1.0))
        for hr in adj.get("hours", []):
            if hr in eff:
                eff[hr] *= factor
    return eff


def _hour_sets(directives: Sequence[Dict[str, Any]], dtype: str) -> set:
    out = set()
    for d in directives:
        if d.get("directive_type") == dtype:
            out.update((d.get("structured_adjustment") or {}).get("hours", []))
    return out


def reserve_by_hour(directives: Sequence[Dict[str, Any]], base_min: float) -> Dict[int, float]:
    """Per-hour floor: max(base reserve, any active directive reserve) - Section 5.3."""
    out: Dict[int, float] = {}
    for d in directives:
        if d.get("directive_type") != "minimum_battery_reserve":
            continue
        adj = d.get("structured_adjustment") or {}
        val = float(adj.get("minimum_energy_kwh", base_min))
        for hr in adj.get("hours", []):
            out[hr] = max(out.get(hr, base_min), val)
    return out


def grid_cap_by_hour(directives: Sequence[Dict[str, Any]]) -> Dict[int, float]:
    """Tightest cap wins if several max_grid_window directives overlap."""
    out: Dict[int, float] = {}
    for d in directives:
        if d.get("directive_type") != "max_grid_window":
            continue
        adj = d.get("structured_adjustment") or {}
        cap = float(adj.get("max_grid_kwh", math.inf))
        for hr in adj.get("hours", []):
            out[hr] = min(out.get(hr, math.inf), cap)
    return out


def validate_plan(
    hours: Sequence[Any],
    battery: Any,
    plan: Sequence[Any],
    directives: Sequence[Dict[str, Any]],
    totals: Dict[str, float] | None = None,
) -> List[str]:
    v: List[str] = []
    P = [p if isinstance(p, dict) else p.model_dump() for p in plan]

    # -- Section 11.3: exactly 24 unique hours 0..23 -------------------------
    plan_hours = [p["hour"] for p in P]
    if sorted(plan_hours) != list(range(24)):
        v.append(f"hourly_plan must contain each hour 0-23 exactly once (got {sorted(plan_hours)})")
        return v
    P.sort(key=lambda p: p["hour"])

    demand = {h.hour: float(h.demand_kwh) for h in hours}
    tariff = {h.hour: float(h.tariff_bdt_per_kwh) for h in hours}
    eff_solar = effective_solar(hours, directives)
    reserves = reserve_by_hour(directives, float(battery.minimum_energy_kwh))
    caps = grid_cap_by_hour(directives)
    no_charge = _hour_sets(directives, "no_charge_window")
    no_discharge = _hour_sets(directives, "no_discharge_window")

    energy = float(battery.initial_energy_kwh)

    for p in P:
        h = p["hour"]
        grid = float(p["grid_kwh"])
        solar_used = float(p["solar_used_kwh"])
        action = p["battery_action"]
        amount = float(p["battery_kwh"])
        after = float(p["battery_energy_after_kwh"])

        # finite + non-negative (Section 11.3)
        for name, val in (
            ("grid_kwh", grid),
            ("solar_used_kwh", solar_used),
            ("battery_kwh", amount),
            ("battery_energy_after_kwh", after),
        ):
            if not math.isfinite(val):
                v.append(f"h{h}: {name} is not finite ({val})")
            elif val < -TOL:
                v.append(f"h{h}: {name} is negative ({val})")

        if action not in ("charge", "discharge", "idle"):
            v.append(f"h{h}: battery_action '{action}' is not charge/discharge/idle")
            continue

        # Section 9.1 / 9.3
        if action == "idle":
            if abs(amount) > TOL:
                v.append(f"h{h}: battery_kwh must be 0 when idle (got {amount}) [9.1]")
            expected_after = energy
        elif action == "charge":
            if amount > float(battery.max_charge_kwh_per_hour) + TOL:
                v.append(
                    f"h{h}: charge {amount} exceeds max_charge_kwh_per_hour "
                    f"{battery.max_charge_kwh_per_hour} [9.3]"
                )
            expected_after = energy + amount
        else:
            if amount > float(battery.max_discharge_kwh_per_hour) + TOL:
                v.append(
                    f"h{h}: discharge {amount} exceeds max_discharge_kwh_per_hour "
                    f"{battery.max_discharge_kwh_per_hour} [9.3]"
                )
            expected_after = energy - amount

        if abs(after - expected_after) > TOL:
            v.append(
                f"h{h}: battery_energy_after_kwh {after} != expected {expected_after} "
                f"from {action} of {amount} on {energy} [9.1]"
            )

        # Section 9.2 + 5.3 reserve directive
        floor = reserves.get(h, float(battery.minimum_energy_kwh))
        if after < floor - TOL:
            v.append(f"h{h}: battery_energy_after_kwh {after} below required reserve {floor} [9.2]")
        if after > float(battery.capacity_kwh) + TOL:
            v.append(
                f"h{h}: battery_energy_after_kwh {after} above capacity "
                f"{battery.capacity_kwh} [9.2]"
            )

        # Section 9.4
        if solar_used > eff_solar.get(h, 0.0) + TOL:
            v.append(
                f"h{h}: solar_used_kwh {solar_used} exceeds effective solar "
                f"{eff_solar.get(h, 0.0)} [9.4]"
            )

        # Section 9.5
        charge_amt = amount if action == "charge" else 0.0
        discharge_amt = amount if action == "discharge" else 0.0
        lhs = grid + solar_used + discharge_amt
        rhs = demand[h] + charge_amt
        if abs(lhs - rhs) > TOL:
            v.append(f"h{h}: energy balance {lhs} != {rhs} [9.5]")

        # Section 5.3 window directives
        if h in no_charge and action == "charge" and amount > TOL:
            v.append(f"h{h}: charging {amount} inside no_charge_window [5.3]")
        if h in no_discharge and action == "discharge" and amount > TOL:
            v.append(f"h{h}: discharging {amount} inside no_discharge_window [5.3]")
        if h in caps and grid > caps[h] + TOL:
            v.append(f"h{h}: grid_kwh {grid} exceeds max_grid_kwh {caps[h]} [5.3]")

        energy = after

    # Section 9.6
    if abs(energy - float(battery.initial_energy_kwh)) > TOL:
        v.append(
            f"end-of-day battery {energy} != initial {battery.initial_energy_kwh} [9.6]"
        )

    # Section 11.3: reported totals must match a recalculation from the plan
    if totals is not None:
        exp_grid = sum(float(p["grid_kwh"]) for p in P)
        exp_cost = sum(float(p["grid_kwh"]) * tariff[p["hour"]] for p in P)
        exp_peak = max(float(p["grid_kwh"]) for p in P)
        for name, got, exp in (
            ("total_grid_kwh", totals.get("total_grid_kwh"), exp_grid),
            ("total_cost_bdt", totals.get("total_cost_bdt"), exp_cost),
            ("peak_grid_kwh", totals.get("peak_grid_kwh"), exp_peak),
        ):
            if got is None or abs(float(got) - exp) > TOL:
                v.append(f"{name} {got} != recalculated {exp} [11.3]")

    return v
