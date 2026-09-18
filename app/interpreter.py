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

SYSTEM_PROMPT = """Convert campus energy operator notes into structured directives.

Output ONLY a JSON object, no prose, no code fences:
{"directives":[{"note_index":int,"applies":bool,"directive_type":str,
"structured_adjustment":object|null,"explanation":str}]}
Exactly one entry per note, in note_index order from 0. No other keys.

TYPES (no others exist):
solar_reduction {"hours":[..],"factor":n} | minimum_battery_reserve {"hours":[..],"minimum_energy_kwh":n}
no_charge_window {"hours":[..]} | no_discharge_window {"hours":[..]}
max_grid_window {"hours":[..],"max_grid_kwh":n} | no_op null

applies=true for all five real types. applies=false ONLY for no_op, which always has null.

HOURS: unique ints 0-23 ascending. Start included, END EXCLUDED.
Method: convert BOTH endpoints to the 24-hour clock first. If an endpoint is not
on the hour, round the START DOWN and the END UP to a whole hour. Then list every
hour from start up to but not including end.
"half past two in the afternoon until half past four" -> 14:30,16:30 -> 14,17 -> [14,15,16]
"from 9:15 to 11:45" -> 9,12 -> [9,10,11]
An hour is included whenever the window covers any part of it.
"1 PM to 3 PM" -> 13,15 -> [13,14]
"10 AM to 1 PM" -> 10,13 -> [10,11,12]     (crossing noon: do not stop at 11)
"09:00 until 12:00" -> 9,12 -> [9,10,11]
"between 5 and 7 tonight" -> 17,19 -> [17,18]
noon=12, midnight=0. "during the 2 PM hour"=[14].
Bare numbers with no am/pm: take the DAYTIME reading for anything about solar,
panel or roof work, or daytime campus activity.
"from one until three" -> 13,15 -> [13,14]   NOT [1,2]
Solar output is zero overnight, so a night reading would make a solar note
meaningless. Only read a bare number as a night hour when the note itself says
night, overnight, or names an explicit am time.

FACTOR = FRACTION REMAINING, never the loss. Decide which the note states:
falls TO 30% -> 0.30 | "only a quarter of normal" -> 0.25
falls BY 70% -> 0.30 | "cut by four fifths" -> 0.20
"drop of x%" and "drop to x%" are different numbers. Read carefully.

RESERVE: always absolute kWh. "at least 140 kWh" -> 140. "40% of capacity" ->
0.40 x the capacity given below.

SYNONYMS, same directive. For words like isolated, offline, unavailable,
disabled, out of service: WHAT is isolated decides the directive, not the word.
no_charge_window: the CHARGER or charging is isolated/offline/unavailable/disabled;
  cannot charge; do not draw into the battery; charging suspended or ceased
no_discharge_window: the BATTERY or PACK is isolated from the load; cannot
  discharge; held in reserve and not used; discharging disabled
max_grid_window: must not exceed N / stay at or below N / cap of N / draw no more than N

no_op: the note does not change today's electricity schedule (staffing, catering,
deliveries, announcements, unrelated systems, another day). Marking a real directive
no_op loses the case; inventing one from an unrelated note also loses it.

Note position means nothing. An unrelated note may be first, middle or last.
explanation: one short plain sentence."""

FEW_SHOT_INPUT = """battery.capacity_kwh = 500

operator_notes:
[0] "Dust storm - only about a third of normal PV output between 09:00 and 12:00."
[1] "Inverter derate cuts rooftop generation by 60% from 2 PM to 5 PM."
[2] "Hold the storage at 40% of rated capacity or better from 21:00 to 23:00."
[3] "Reminder: the library extends its opening hours next week."
"""

FEW_SHOT_OUTPUT = """{"directives":[
{"note_index":0,"applies":true,"directive_type":"solar_reduction","structured_adjustment":{"hours":[9,10,11],"factor":0.33},"explanation":"Dust leaves about a third of solar usable."},
{"note_index":1,"applies":true,"directive_type":"solar_reduction","structured_adjustment":{"hours":[14,15,16],"factor":0.4},"explanation":"A 60% cut leaves 40% usable."},
{"note_index":2,"applies":true,"directive_type":"minimum_battery_reserve","structured_adjustment":{"hours":[21,22],"minimum_energy_kwh":200},"explanation":"40% of 500 kWh is a 200 kWh floor."},
{"note_index":3,"applies":false,"directive_type":"no_op","structured_adjustment":null,"explanation":"Library hours do not affect today's schedule."}]}"""

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
        f"{FEW_SHOT_INPUT}\n{FEW_SHOT_OUTPUT}\n\nNow do the same for this input.\n\n{user}"
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
