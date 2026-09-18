"""Local scoreboard against the 10 public sample cases.

These are public examples, NOT the hidden judge set - passing everything here
is necessary, not sufficient. Note wording, case IDs and reference numbers are
read from the JSON at runtime and never hard-coded into the service.

Usage:  python tests/run_local.py [path/to/sample_cases.json]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _env import load as _load_env  # noqa: E402

_loaded = _load_env()

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schemas import BatterySpec, HourInput  # noqa: E402
from app.validator import validate_plan  # noqa: E402

DEFAULT_PACK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "public_sample_cases.json"
)
TOL = 0.01


def norm_adj(d):
    """Comparable form of a structured_adjustment, tolerant of float noise."""
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        out[k] = list(v) if isinstance(v, list) else round(float(v), 4)
    return out


def interp_matches(got, exp):
    if got.get("note_index") != exp.get("note_index"):
        return False, "note_index"
    if bool(got.get("applies")) != bool(exp.get("applies")):
        return False, "applies"
    if got.get("directive_type") != exp.get("directive_type"):
        return False, f"type({got.get('directive_type')}!={exp.get('directive_type')})"
    if norm_adj(got.get("structured_adjustment")) != norm_adj(exp.get("structured_adjustment")):
        return False, f"adjustment{got.get('structured_adjustment')}"
    return True, ""


def main(path: str) -> int:
    pack = json.load(open(path, encoding="utf-8"))
    cases = pack["cases"]
    client = TestClient(app)

    rows = []
    hard_fail = 0

    for case in cases:
        inp = case["input"]
        exp = case["expected_output"]
        cid = case["id"]

        r = client.post("/optimize-energy", json=inp)
        if r.status_code != 200:
            rows.append((cid, f"HTTP {r.status_code}", "-", "-", "-"))
            hard_fail += 1
            continue
        body = r.json()

        # --- contract ------------------------------------------------------
        missing = [k for k in pack["_meta"]["schema_notes"]["output_required_fields"] if k not in body]
        contract = "ok" if not missing else f"missing {missing}"
        if body.get("scenario_id") != inp["scenario_id"]:
            contract = "scenario_id mismatch"
        if len(body.get("directive_interpretation", [])) != len(inp["operator_notes"]):
            contract = "interp count != note count"
        if [e["note_index"] for e in body.get("directive_interpretation", [])] != list(
            range(len(inp["operator_notes"]))
        ):
            contract = "interp not in note_index order"
        if contract != "ok":
            hard_fail += 1

        # --- interpretation vs public reference ----------------------------
        got_i = body.get("directive_interpretation", [])
        exp_i = exp["directive_interpretation"]
        n_ok = 0
        why = []
        for g, e in zip(got_i, exp_i):
            ok, reason = interp_matches(g, e)
            n_ok += ok
            if not ok:
                why.append(reason)
        interp = f"{n_ok}/{len(exp_i)}" + ("" if n_ok == len(exp_i) else " " + ",".join(why[:2]))

        # --- plan validity, replayed against the REFERENCE directives ------
        hours = [HourInput(**h) for h in inp["hours"]]
        battery = BatterySpec(**inp["battery"])
        ref_directives = [e for e in exp_i if e.get("applies")]
        totals = {k: body.get(k) for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh")}
        viol = validate_plan(hours, battery, body["hourly_plan"], ref_directives, totals)
        base_viol = validate_plan(hours, battery, body["hourly_plan"], [], totals)
        plan = "valid" if not viol else f"{len(viol)} viol"
        base = "valid" if not base_viol else f"{len(base_viol)} viol"

        # --- cost vs reference optimum -------------------------------------
        ref_cost = float(exp["total_cost_bdt"])
        got_cost = float(body.get("total_cost_bdt", 0))
        gap = (got_cost - ref_cost) / ref_cost * 100 if ref_cost else 0.0
        cost = f"{got_cost:,.0f} vs {ref_cost:,.0f} ({gap:+.1f}%)"

        rows.append((cid, contract, interp, f"{base}/{plan}", cost))
        if viol and os.getenv("VERBOSE"):
            for x in viol[:6]:
                print(f"    {cid}: {x}")

    w = [10, 22, 26, 22, 30]
    hdr = ("case", "contract", "interpretation", "base/with-directives", "cost vs reference")
    print("\n" + "  ".join(h.ljust(x) for h, x in zip(hdr, w)))
    print("  ".join("-" * x for x in w))
    for row in rows:
        print("  ".join(str(c).ljust(x) for c, x in zip(row, w)))
    print()
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PACK))
