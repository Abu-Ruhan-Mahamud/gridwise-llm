"""Grid-only baseline schedule.

Battery idles for all 24 hours, so:
  * energy balance holds:            grid = demand - solar_used   (Section 9.5)
  * battery bounds hold trivially:   E_after == initial_energy    (Section 9.2)
  * end-of-day neutrality holds:     E_23 == initial_energy       (Section 9.6)

This is a genuinely feasible schedule under the base GridWise rules, not a
cosmetic placeholder. It serves two purposes: it proves the API contract
independent of the optimizer now, and it stays in the codebase as the
last-resort fallback if the solver fails or times out during judging.

It does NOT apply operator directives - minimum_battery_reserve and
max_grid_window can still be violated by it, so it is never a substitute for
the optimizer on a scored case.
"""

from __future__ import annotations

from typing import List, Tuple

from .schemas import BatterySpec, HourInput, HourPlan

ROUND_DP = 6


def _r(x: float) -> float:
    return round(float(x), ROUND_DP)


def build_grid_only_plan(hours: List[HourInput], battery: BatterySpec) -> List[HourPlan]:
    plan: List[HourPlan] = []
    energy = battery.initial_energy_kwh
    for h in sorted(hours, key=lambda x: x.hour):
        solar_used = max(0.0, min(h.solar_kwh, h.demand_kwh))
        grid = max(0.0, h.demand_kwh - solar_used)
        plan.append(
            HourPlan(
                hour=h.hour,
                grid_kwh=_r(grid),
                solar_used_kwh=_r(solar_used),
                battery_action="idle",
                battery_kwh=0.0,
                battery_energy_after_kwh=_r(energy),
            )
        )
    return plan


def summarize(plan: List[HourPlan], hours: List[HourInput]) -> Tuple[float, float, float]:
    """Recompute the three totals FROM the plan - Section 11.3 requires that
    total_grid_kwh, total_cost_bdt and peak_grid_kwh match a recalculation of
    the returned hourly_plan, so they are never tracked separately."""
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours}
    total_grid = sum(p.grid_kwh for p in plan)
    total_cost = sum(p.grid_kwh * tariff[p.hour] for p in plan)
    peak_grid = max((p.grid_kwh for p in plan), default=0.0)
    return _r(total_grid), _r(total_cost), _r(peak_grid)
