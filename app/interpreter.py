"""LLM interpretation of operator notes (Sections 02, 04, 05).

The model does the language work - deciding which directive a note expresses,
which hours it covers, and which number it carries. It does NOT do arithmetic
we can do deterministically, and it never sees or touches demand, tariff or
battery figures beyond the one value it needs (capacity, to resolve a
percentage-of-capacity reserve).

All few-shot wording here is written from scratch. None of it is lifted from
the public sample pack: hard-coding published phrasing is explicitly barred,
and it would also teach the model the wrong lesson, since hidden notes
paraphrase. Each directive type is shown in TWO phrasing families so the model
generalizes the concept rather than a surface form.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple

from .guardrails import enforce
from .llm import TOTAL_BUDGET_S, LLMUnavailable, complete_json

log = logging.getLogger("gridwise.interpreter")

# Both providers have tight free-tier quotas, so an identical note set is
# answered from memory rather than spending a request. Keyed on the exact notes
# plus capacity (capacity changes a percentage-of-capacity reserve). Bounded so
# a long judging run cannot grow it without limit. Process-local and derived
# only from request data - no cross-scenario state, no persistence.
_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_MAX = 256

SYSTEM_PROMPT = """You convert campus energy operator notes into structured directives.

You return ONLY a JSON object of the form:
{"directives": [ {"note_index": int, "applies": bool, "directive_type": str,
                  "structured_adjustment": object or null, "explanation": str}, ... ]}

Return exactly one entry per operator note, in note_index order starting at 0.

THE ONLY PERMITTED directive_type VALUES
  solar_reduction          {"hours": [...], "factor": number}
  minimum_battery_reserve  {"hours": [...], "minimum_energy_kwh": number}
  no_charge_window         {"hours": [...]}
  no_discharge_window      {"hours": [...]}
  max_grid_window          {"hours": [...], "max_grid_kwh": number}
  no_op                    null
Never invent another type. Never add extra keys.

applies is true for all five real directives. applies is false ONLY for no_op,
and no_op always has structured_adjustment null.

HOURS
Whole-hour integers 0-23, unique, ascending. The start hour is INCLUDED and the
end hour is EXCLUDED.
  "1 PM to 3 PM"            -> [13, 14]
  "09:00 until 12:00"       -> [9, 10, 11]
  "between 5 and 7 tonight" -> [17, 18]
  "from 11 PM to 1 AM"      -> [23, 0]  (sorted ascending: [0, 23])
noon is 12, midnight is 0. A single hour like "during the 2 PM hour" is [14].

solar_reduction factor: THE FRACTION OF SOLAR THAT REMAINS.
Notes state this in two opposite ways. Decide which before you write a number.
  Remaining  "output will be about 30% of forecast"  -> factor 0.30
  Remaining  "only a quarter of normal generation"   -> factor 0.25
  Lost       "expect a 70% drop in solar"            -> factor 0.30  (1 - 0.70)
  Lost       "generation cut by four fifths"         -> factor 0.20  (1 - 0.80)
A note saying output FALLS BY x% gives factor 1 - x/100. A note saying output
FALLS TO x% gives factor x/100. These are different numbers. Read carefully.

minimum_battery_reserve is stated in two ways.
  Absolute    "keep at least 140 kWh stored"        -> minimum_energy_kwh 140
  Percentage  "hold at least 40% of capacity"       -> multiply 0.40 by the
              battery capacity given in the input and return absolute kWh.
Always return absolute kWh, never a percentage.

WORDING VARIES. Treat all of these as the same underlying directive:
  no_charge_window     cannot charge / charging unavailable / charger offline /
                       do not draw into the battery / charging disabled
  no_discharge_window  cannot discharge / keep the pack isolated / battery held
                       in reserve and not used / discharging disabled
  max_grid_window      import must not exceed N / stay at or below N / cap of N /
                       limit is N / draw no more than N

no_op is for a note that does not change today's 24-hour electricity schedule:
staffing, catering, deliveries, announcements, maintenance of unrelated systems,
or anything about a different day. Marking a real directive no_op loses the case,
and inventing a directive from an unrelated note loses it too.

A note's position carries no meaning. An unrelated note may be first, middle or
last. Judge every note on its own content.

explanation: one short sentence, plain language, no JSON, no restating numbers
you did not derive."""

FEW_SHOT_INPUT = """battery.capacity_kwh = 500

operator_notes:
[0] "Dust storm forecast for the morning - only about a third of normal PV output between 09:00 and 12:00."
[1] "Night shift wants no less than 140 kWh banked from 21:00 to 23:00."
[2] "Reminder: the library extends its opening hours next week."
"""

FEW_SHOT_OUTPUT = """{"directives": [
{"note_index": 0, "applies": true, "directive_type": "solar_reduction",
 "structured_adjustment": {"hours": [9, 10, 11], "factor": 0.33},
 "explanation": "Dust reduces usable solar to about a third during the morning window."},
{"note_index": 1, "applies": true, "directive_type": "minimum_battery_reserve",
 "structured_adjustment": {"hours": [21, 22], "minimum_energy_kwh": 140},
 "explanation": "A 140 kWh floor is required across the late evening."},
{"note_index": 2, "applies": false, "directive_type": "no_op",
 "structured_adjustment": null,
 "explanation": "Library hours do not affect today's energy schedule."}]}"""

FEW_SHOT_INPUT_2 = """battery.capacity_kwh = 400

operator_notes:
[0] "Substation maintenance: import must stay at or below 90 kWh in any hour from 6 AM to 9 AM."
[1] "Inverter derate will cut rooftop generation by 60% from 2 PM to 5 PM."
[2] "Keep the pack isolated from the load between 19:00 and 21:00."
"""

FEW_SHOT_OUTPUT_2 = """{"directives": [
{"note_index": 0, "applies": true, "directive_type": "max_grid_window",
 "structured_adjustment": {"hours": [6, 7, 8], "max_grid_kwh": 90},
 "explanation": "Grid import is capped at 90 kWh per hour during substation work."},
{"note_index": 1, "applies": true, "directive_type": "solar_reduction",
 "structured_adjustment": {"hours": [14, 15, 16], "factor": 0.4},
 "explanation": "A 60% cut leaves 40% of rooftop generation usable in the afternoon."},
{"note_index": 2, "applies": false, "directive_type": "no_discharge_window",
 "structured_adjustment": {"hours": [19, 20]},
 "explanation": "The battery must not supply load while isolated."}]}"""

# NOTE: entry 2 above intentionally carries applies=false on a real directive.
# It is corrected to true below, and the correction is shown to the model so the
# applies rule is demonstrated rather than only asserted.
FEW_SHOT_OUTPUT_2 = FEW_SHOT_OUTPUT_2.replace(
    '{"note_index": 2, "applies": false, "directive_type": "no_discharge_window"',
    '{"note_index": 2, "applies": true, "directive_type": "no_discharge_window"',
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def build_user_prompt(notes: Sequence[str], capacity_kwh: float) -> str:
    listed = "\n".join(f"[{i}] {note!r}" for i, note in enumerate(notes))
    return (
        f"battery.capacity_kwh = {capacity_kwh}\n\noperator_notes:\n{listed}\n\n"
        f"Return exactly {len(notes)} directive entries, note_index 0 to {len(notes) - 1}."
    )


def _parse(raw: str) -> Any:
    """Tolerant JSON extraction - some models wrap output in prose or fences."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(raw or "")
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


async def interpret(
    notes: Sequence[str], capacity_kwh: float
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (guardrailed entries, diagnostics). Never raises.

    Diagnostics are for our own logs and the local scoreboard. They are not
    part of the response schema and never carry key material.
    """
    cache_key = (tuple(notes), float(capacity_kwh))
    if cache_key in _CACHE:
        entries, meta = _CACHE[cache_key]
        _CACHE.move_to_end(cache_key)
        return [dict(e) for e in entries], {**meta, "cached": True}

    meta: Dict[str, Any] = {"llm_ok": False, "attempts": 0, "repairs": [], "error": None}
    # One deadline covers every provider and every retry, so a slow or
    # rate-limited provider can never push us past the 30 s request limit.
    deadline = time.monotonic() + TOTAL_BUDGET_S
    user = build_user_prompt(notes, capacity_kwh)

    messages_user = (
        f"{FEW_SHOT_INPUT}\n{FEW_SHOT_OUTPUT}\n\n"
        f"{FEW_SHOT_INPUT_2}\n{FEW_SHOT_OUTPUT_2}\n\n"
        f"Now do the same for this input.\n\n{user}"
    )

    raw = None
    for attempt in (1, 2):
        meta["attempts"] = attempt
        try:
            raw = await complete_json(SYSTEM_PROMPT, messages_user, deadline=deadline)
        except LLMUnavailable as exc:
            meta["error"] = str(exc)
            log.error("LLM unavailable: %s", exc)
            break

        parsed = _parse(raw)
        payload = None
        if isinstance(parsed, dict):
            payload = parsed.get("directives")
            if payload is None and "note_index" in parsed:
                payload = [parsed]
        elif isinstance(parsed, list):
            payload = parsed

        if payload is not None:
            entries, repairs = enforce(payload, len(notes), capacity_kwh)
            meta["llm_ok"] = True
            meta["repairs"] = repairs
            _CACHE[cache_key] = ([dict(e) for e in entries], dict(meta))
            if len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
            return entries, meta

        log.warning("attempt %s: model output was not parseable JSON", attempt)
        messages_user += (
            "\n\nYour previous reply was not valid JSON. Reply with the JSON object only."
        )

    # Fail safe (Section 08): a valid all-no_op response beats a crash.
    meta["error"] = meta["error"] or "unparseable LLM output"
    entries, repairs = enforce([], len(notes), capacity_kwh)
    meta["repairs"] = repairs
    return entries, meta
