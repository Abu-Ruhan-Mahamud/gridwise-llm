"""Team-authored tricky pack v2 (TRICKY-01..20): adversarial interpretation traps.

Checks interpretation against the case's stated expectation, and separately
replays the returned schedule against those same expected directives.
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

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tricky_cases_v2.json")


def norm(a):
    if a is None:
        return None
    return {k: (sorted(v) if isinstance(v, list) else round(float(v), 4)) for k, v in a.items()}


def main() -> int:
    pack = json.load(open(CASES, encoding="utf-8"))
    client = TestClient(app)
    rows, fails = [], []

    for case in pack["cases"]:
        cid, inp = case["id"], case["input"]
        exp = case["expected_directive_interpretation"]
        r = client.post("/optimize-energy", json=inp)
        if r.status_code != 200:
            rows.append((cid, case.get("label", "")[:30], f"HTTP {r.status_code}", "-"))
            fails.append(cid)
            continue
        b = r.json()
        got = b["directive_interpretation"]

        ok, detail = len(got) == len(exp), []
        for g, e in zip(got, exp):
            if g["directive_type"] != e["directive_type"]:
                ok = False
                detail.append(f"got {g['directive_type']} want {e['directive_type']}")
            elif norm(g["structured_adjustment"]) != norm(e["structured_adjustment"]):
                ok = False
                detail.append(f"got {g['structured_adjustment']} want {e['structured_adjustment']}")
            elif bool(g["applies"]) != bool(e["applies"]):
                ok = False
                detail.append("applies")

        hours = [HourInput(**h) for h in inp["hours"]]
        battery = BatterySpec(**inp["battery"])
        truth = [e for e in exp if e.get("applies")]
        totals = {k: b[k] for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh")}
        viol = validate_plan(hours, battery, b["hourly_plan"], truth, totals)

        if not ok:
            fails.append(cid)
        rows.append((cid, case.get("label", "")[:30],
                     "PASS" if ok else "; ".join(detail)[:58],
                     "valid" if not viol else f"{len(viol)} viol"))

    w = [12, 32, 60, 10]
    print("\n" + "  ".join(h.ljust(x) for h, x in zip(("case", "label", "interpretation", "plan"), w)))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(str(c).ljust(x) for c, x in zip(r, w)))
    print(f"\n{len(rows) - len(fails)}/{len(rows)} interpretation pass"
          + (f"  failures: {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
