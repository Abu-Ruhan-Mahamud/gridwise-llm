"""Cost-minimizing 24-hour schedule (Section 5.2), solved as a linear program.

Model per hour h:
    g[h]  grid import            >= 0, <= max_grid_kwh if a cap applies
    s[h]  solar used             in [0, effective_solar[h]]
    c[h]  battery charge         in [0, max_charge], forced 0 in a no_charge_window
    d[h]  battery discharge      in [0, max_discharge], forced 0 in a no_discharge_window
    e[h]  battery energy after h in [max(base reserve, directive reserve), capacity]

    minimize  SUM g[h] * tariff[h]
    s.t.      g + s + d == demand + c            (9.5 energy balance)
              e[h] == e[h-1] + c[h] - d[h]       (9.1 battery state)
              e[23] == initial_energy_kwh        (9.6 end-of-day neutrality)

No binary variables are needed. An LP can return c[h] and d[h] both positive,
but that is always degenerate: it cancels out of both the balance equation and
the state equation, so netting it afterwards changes nothing except giving us a
single reportable battery_action. Keeping the model a pure LP keeps it in the
low milliseconds, which protects the 30 s request budget.

Validity outranks cost (Section 5.2), so every candidate schedule is replayed
through the final validator before it is accepted, and there is a fallback
ladder if the solver cannot produce a valid one.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import pulp

from .baseline import build_grid_only_plan
from .schemas import HourPlan
from .validator import (
    _hour_sets,
    effective_solar,
    grid_cap_by_hour,
    reserve_by_hour,
    validate_plan,
)

log = logging.getLogger("gridwise.optimizer")

ROUND_DP = 6
TOL = 0.01
SOLVER_TIME_LIMIT_S = 10


def _r(x: float) -> float:
    return round(float(x), ROUND_DP)


def _solve_lp(
    hours: Sequence[Any],
    battery: Any,
    directives: Sequence[Dict[str, Any]],
) -> List[HourPlan] | None:
    ordered = sorted(hours, key=lambda h: h.hour)
    idx = [h.hour for h in ordered]
    demand = {h.hour: float(h.demand_kwh) for h in ordered}
    tariff = {h.hour: float(h.tariff_bdt_per_kwh) for h in ordered}

    eff = effective_solar(ordered, directives)
    base_min = float(battery.minimum_energy_kwh)
    capacity = float(battery.capacity_kwh)
    reserves = reserve_by_hour(directives, base_min)
    caps = grid_cap_by_hour(directives)
    no_charge = _hour_sets(directives, "no_charge_window")
    no_discharge = _hour_sets(directives, "no_discharge_window")

    prob = pulp.LpProblem("gridwise", pulp.LpMinimize)

    g, s, c, d, e = {}, {}, {}, {}, {}
    for h in idx:
        g[h] = pulp.LpVariable(f"g_{h}", lowBound=0, upBound=caps.get(h))
        s[h] = pulp.LpVariable(f"s_{h}", lowBound=0, upBound=max(0.0, eff.get(h, 0.0)))
        c[h] = pulp.LpVariable(
            f"c_{h}", lowBound=0, upBound=0.0 if h in no_charge else float(battery.max_charge_kwh_per_hour)
        )
        d[h] = pulp.LpVariable(
            f"d_{h}",
            lowBound=0,
            upBound=0.0 if h in no_discharge else float(battery.max_discharge_kwh_per_hour),
        )
        # A reserve above capacity would make the bounds contradictory; the
        # guardrail layer rejects that upstream, this is belt-and-braces.
        floor = min(max(reserves.get(h, base_min), base_min), capacity)
        e[h] = pulp.LpVariable(f"e_{h}", lowBound=floor, upBound=capacity)

    prob += pulp.lpSum(g[h] * tariff[h] for h in idx)

    prev = float(battery.initial_energy_kwh)
    for h in idx:
        prob += g[h] + s[h] + d[h] == demand[h] + c[h], f"balance_{h}"
        prob += e[h] == prev + c[h] - d[h], f"state_{h}"
        prev = e[h]
    prob += e[idx[-1]] == float(battery.initial_energy_kwh), "end_of_day_neutral"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=SOLVER_TIME_LIMIT_S))
    if pulp.LpStatus[status] != "Optimal":
        log.warning("LP not optimal: %s", pulp.LpStatus[status])
        return None

    # ---- net the degenerate simultaneous charge/discharge, then rebuild the
    # ---- battery trajectory from the netted amounts so no float drift creeps in
    plan: List[HourPlan] = []
    energy = float(battery.initial_energy_kwh)
    for h in idx:
        net = (c[h].value() or 0.0) - (d[h].value() or 0.0)
        if net > TOL:
            action, amount = "charge", _r(net)
        elif net < -TOL:
            action, amount = "discharge", _r(-net)
        else:
            action, amount = "idle", 0.0

        # Free solar first: raising solar_used can only lower grid import, so it
        # never breaks a grid cap and never raises cost. The LP is indifferent
        # when tariff is 0, so this pins down a clean, obviously-valid plan.
        charge_amt = amount if action == "charge" else 0.0
        discharge_amt = amount if action == "discharge" else 0.0
        required = demand[h] + charge_amt - discharge_amt
        solar_used = max(0.0, min(eff.get(h, 0.0), required))
        grid = max(0.0, required - solar_used)

        energy = energy + charge_amt - discharge_amt
        plan.append(
            HourPlan(
                hour=h,
                grid_kwh=_r(grid),
                solar_used_kwh=_r(solar_used),
                battery_action=action,
                battery_kwh=amount,
                battery_energy_after_kwh=_r(energy),
            )
        )
    return plan


def solve_schedule(
    hours: Sequence[Any],
    battery: Any,
    directives: Sequence[Dict[str, Any]],
) -> Tuple[List[HourPlan], str]:
    """Return (plan, method). Never raises; always returns a usable schedule.

    Ladder, in order of how much we would rather not use it:
      lp                 - optimal under every directive. The expected path.
      lp_no_directives   - directives made the model infeasible or invalid.
                           Scores 0 on that case's directive application, but a
                           valid response beats a 500.
      baseline           - solver unavailable entirely.
    """
    try:
        plan = _solve_lp(hours, battery, directives)
        if plan is not None:
            viol = validate_plan(hours, battery, plan, directives)
            if not viol:
                return plan, "lp"
            log.error("LP produced an invalid plan, falling back: %s", viol[:3])
    except Exception:  # noqa: BLE001 - Section 08 SAFE FAILURE
        log.exception("LP solve failed")

    if directives:
        try:
            plan = _solve_lp(hours, battery, [])
            if plan is not None and not validate_plan(hours, battery, plan, []):
                log.warning("falling back to directive-free LP")
                return plan, "lp_no_directives"
        except Exception:  # noqa: BLE001
            log.exception("directive-free LP failed")

    log.warning("falling back to grid-only baseline")
    return build_grid_only_plan(list(hours), battery), "baseline"
