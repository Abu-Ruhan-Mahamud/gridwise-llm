"""Team-authored tricky cases (X21-X32), run against the full local pipeline.

Checks both halves of the score separately: the extracted directive against the
case's expected interpretation, and the returned schedule against the same
expected directives (which is what the judge replays).

X30-feeder is infeasible by construction - at h18/h19 demand minus solar minus
the battery's max hourly discharge is 155/165 kWh, so no schedule can honour a
150 kWh cap. Correct behaviour there is to extract the directive correctly and
let the feasibility ladder drop it, so it is marked as a known exception rather
than left looking like a regression.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import load as _load_env  # noqa: E402

_load_env()

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schemas import BatterySpec, HourInput  # noqa: E402
from app.validator import validate_plan  # noqa: E402

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tricky_extra_cases.json")
KNOWN_INFEASIBLE = {"X30-feeder"}


def norm(a):
    if a is None:
        return None
    return {k: (sorted(v) if isinstance(v, list) else round(float(v), 4)) for k, v in a.items()}


def main() -> int:
    pack = json.load(open(CASES, encoding="utf-8"))
    client = TestClient(app)
    rows, fails = [], []

    for case in pack["cases"]:
        cid, inp, exp = case["id"], case["input"], case["expected"]
        r = client.post("/optimize-energy", json=inp)
        if r.status_code != 200:
            rows.append((cid, f"HTTP {r.status_code}", "-", "-"))
            fails.append(cid)
            continue
        b = r.json()
        got = b["directive_interpretation"]

        ok, detail = len(got) == len(exp), []
        for g, e in zip(got, exp):
            if g["directive_type"] != e["directive_type"]:
                ok = False
                detail.append(f"{g['directive_type']}!={e['directive_type']}")
            elif norm(g["structured_adjustment"]) != norm(e["structured_adjustment"]):
                ok = False
                detail.append(f"adj {g['structured_adjustment']}")
            if bool(g["applies"]) != bool(e["applies"]):
                ok = False
                detail.append("applies")

        hours = [HourInput(**h) for h in inp["hours"]]
        battery = BatterySpec(**inp["battery"])
        truth = [e for e in exp if e.get("applies")]
        totals = {k: b[k] for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh")}
        viol = validate_plan(hours, battery, b["hourly_plan"], truth, totals)

        plan_state = "valid"
        if viol:
            plan_state = f"{len(viol)} viol" + (" (expected)" if cid in KNOWN_INFEASIBLE else "")
        if not ok or (viol and cid not in KNOWN_INFEASIBLE):
            fails.append(cid)

        rows.append((cid, "PASS" if ok else "; ".join(detail)[:34], plan_state,
                     f"{b['total_cost_bdt']:,.0f}"))

    w = [16, 36, 20, 12]
    print("\n" + "  ".join(h.ljust(x) for h, x in zip(("case", "interpretation", "plan", "cost"), w)))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(str(c).ljust(x) for c, x in zip(r, w)))
    print(f"\n{len(rows) - len(fails)}/{len(rows)} pass" + (f"  failures: {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
