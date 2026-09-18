"""Optimizer in isolation, fed the PUBLIC REFERENCE directives.

This deliberately bypasses the LLM: it answers "given a perfect interpretation,
does the optimizer produce a valid, cost-competitive schedule?" so optimizer
bugs can never hide behind interpretation bugs (they are scored separately -
25 pts application vs 25 pts interpretation).
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.optimizer import solve_schedule  # noqa: E402
from app.schemas import BatterySpec, HourInput  # noqa: E402
from app.validator import validate_plan  # noqa: E402

PACK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
)


def main() -> int:
    pack = json.load(open(PACK, encoding="utf-8"))
    rows, failures = [], 0

    for case in pack["cases"]:
        inp, exp = case["input"], case["expected_output"]
        hours = [HourInput(**h) for h in inp["hours"]]
        battery = BatterySpec(**inp["battery"])
        directives = [e for e in exp["directive_interpretation"] if e.get("applies")]

        t0 = time.perf_counter()
        plan, method = solve_schedule(hours, battery, directives)
        ms = (time.perf_counter() - t0) * 1000

        tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours}
        cost = sum(p.grid_kwh * tariff[p.hour] for p in plan)
        totals = {
            "total_grid_kwh": sum(p.grid_kwh for p in plan),
            "total_cost_bdt": cost,
            "peak_grid_kwh": max(p.grid_kwh for p in plan),
        }
        viol = validate_plan(hours, battery, plan, directives, totals)

        ref = float(exp["total_cost_bdt"])
        gap = (cost - ref) / ref * 100 if ref else 0.0
        ok = not viol and method == "lp" and gap <= 0.01
        if not ok:
            failures += 1
        rows.append(
            (
                case["id"],
                method,
                "valid" if not viol else f"{len(viol)} VIOL",
                f"{cost:,.0f}",
                f"{ref:,.0f}",
                f"{gap:+.2f}%",
                f"{ms:.0f}ms",
                "PASS" if ok else "FAIL",
            )
        )
        for x in viol[:4]:
            print(f"    {case['id']}: {x}")

    w = [11, 18, 10, 11, 11, 9, 8, 6]
    hdr = ("case", "method", "plan", "our cost", "reference", "gap", "solve", "")
    print("\n" + "  ".join(h.ljust(x) for h, x in zip(hdr, w)))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(str(c).ljust(x) for c, x in zip(r, w)))
    print(f"\n{len(rows) - failures}/{len(rows)} cases pass\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
