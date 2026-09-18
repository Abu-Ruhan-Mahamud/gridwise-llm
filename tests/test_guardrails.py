"""Adversarial guardrail tests - no API key required.

Each case feeds the guardrail layer output a misbehaving model could plausibly
produce, and asserts the deterministic layer neutralises it. This is the part
of Section 08 that is scored whether or not the LLM behaves on the night.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.guardrails import enforce  # noqa: E402

CAP = 500.0
FAILS = []


def check(name, entries, note_count, expect):
    got, repairs = enforce(entries, note_count, CAP)
    ok = True
    if len(got) != note_count:
        ok = False
    for i, exp in enumerate(expect):
        for k, v in exp.items():
            if got[i].get(k) != v:
                ok = False
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
        print(f"  {status}  {name}\n        got={got}\n        want={expect}")
    else:
        print(f"  {status}  {name}")


print("\nguardrail behaviour under hostile LLM output\n" + "-" * 46)

check(
    "unsupported directive_type is neutralised",
    [{"note_index": 0, "applies": True, "directive_type": "shed_load",
      "structured_adjustment": {"hours": [1]}, "explanation": "x"}],
    1, [{"directive_type": "no_op", "applies": False, "structured_adjustment": None}])

check(
    "missing note gets a no_op so the 1:1 mapping holds",
    [{"note_index": 0, "applies": True, "directive_type": "no_charge_window",
      "structured_adjustment": {"hours": [3, 4]}, "explanation": "x"}],
    3, [{"directive_type": "no_charge_window"}, {"directive_type": "no_op"},
        {"directive_type": "no_op"}])

check(
    "duplicate note_index: first wins, second dropped",
    [{"note_index": 0, "applies": True, "directive_type": "no_charge_window",
      "structured_adjustment": {"hours": [3]}, "explanation": "a"},
     {"note_index": 0, "applies": True, "directive_type": "no_discharge_window",
      "structured_adjustment": {"hours": [5]}, "explanation": "b"}],
    1, [{"directive_type": "no_charge_window"}])

check(
    "solar factor above 1 is rejected, not clamped",
    [{"note_index": 0, "applies": True, "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12], "factor": 1.4}, "explanation": "x"}],
    1, [{"directive_type": "no_op"}])

check(
    "negative solar factor rejected",
    [{"note_index": 0, "applies": True, "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12], "factor": -0.2}, "explanation": "x"}],
    1, [{"directive_type": "no_op"}])

check(
    "reserve above capacity is clamped to capacity",
    [{"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
      "structured_adjustment": {"hours": [20], "minimum_energy_kwh": 900}, "explanation": "x"}],
    1, [{"directive_type": "minimum_battery_reserve",
         "structured_adjustment": {"hours": [20], "minimum_energy_kwh": CAP}}])

check(
    "hours are deduped and sorted ascending",
    [{"node": 1, "note_index": 0, "applies": True, "directive_type": "no_charge_window",
      "structured_adjustment": {"hours": [9, 3, 9, 5]}, "explanation": "x"}],
    1, [{"structured_adjustment": {"hours": [3, 5, 9]}}])

check(
    "out-of-range hour invalidates the window",
    [{"note_index": 0, "applies": True, "directive_type": "no_charge_window",
      "structured_adjustment": {"hours": [22, 24]}, "explanation": "x"}],
    1, [{"directive_type": "no_op"}])

check(
    "applies=false on a real directive is corrected to true",
    [{"note_index": 0, "applies": False, "directive_type": "max_grid_window",
      "structured_adjustment": {"hours": [7], "max_grid_kwh": 80}, "explanation": "x"}],
    1, [{"applies": True, "directive_type": "max_grid_window"}])

check(
    "applies=true on no_op is corrected to false",
    [{"note_index": 0, "applies": True, "directive_type": "no_op",
      "structured_adjustment": {"hours": [7]}, "explanation": "x"}],
    1, [{"applies": False, "structured_adjustment": None}])

check(
    "invented extra keys are stripped from structured_adjustment",
    [{"note_index": 0, "applies": True, "directive_type": "max_grid_window",
      "structured_adjustment": {"hours": [7], "max_grid_kwh": 80, "demand_kwh": 999,
                                "tariff_bdt_per_kwh": 0.1}, "explanation": "x"}],
    1, [{"structured_adjustment": {"hours": [7], "max_grid_kwh": 80}}])

check(
    "negative grid cap rejected",
    [{"note_index": 0, "applies": True, "directive_type": "max_grid_window",
      "structured_adjustment": {"hours": [7], "max_grid_kwh": -5}, "explanation": "x"}],
    1, [{"directive_type": "no_op"}])

check(
    "garbage payload still yields one no_op per note",
    "not a list", 2, [{"directive_type": "no_op"}, {"directive_type": "no_op"}])

check(
    "empty hours list is unusable",
    [{"note_index": 0, "applies": True, "directive_type": "no_discharge_window",
      "structured_adjustment": {"hours": []}, "explanation": "x"}],
    1, [{"directive_type": "no_op"}])

check(
    "out-of-range note_index dropped, mapping preserved",
    [{"note_index": 7, "applies": True, "directive_type": "no_charge_window",
      "structured_adjustment": {"hours": [1]}, "explanation": "x"}],
    1, [{"directive_type": "no_op"}])

print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}\n")
sys.exit(1 if FAILS else 0)
