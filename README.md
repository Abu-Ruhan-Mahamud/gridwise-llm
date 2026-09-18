# GridWise LLM

LLM-assisted campus energy directive interpretation and 24-hour cost optimization.
BUP CSE Fest 2026 Hackathon - Online Preliminary Round.

One HTTP service. Operator notes in natural language are interpreted by an LLM,
validated by deterministic code, applied as hard constraints to a linear
program, and the finished schedule is replayed against every rule before it is
returned.

```
Energy data + operator notes
    -> LLM interpreter        app/interpreter.py   (language -> structured directives)
    -> Guardrail validator    app/guardrails.py    (untrusted output -> safe directives)
    -> Math optimizer         app/optimizer.py     (LP over 24 hours, PuLP/CBC)
    -> Final validator        app/validator.py     (replay; nothing leaves unchecked)
    -> API response
```

The LLM sits in the interpretation path, which is where the challenge requires
it. `plan_summary` is generated deterministically in code, not by the model.

---

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/Abu-Ruhan-Mahamud/gridwise-llm.git
cd gridwise-llm
pip install -r requirements.txt

cp .env.example .env        # then paste your key(s) into .env
export $(grep -v '^#' .env | xargs)      # Linux/macOS

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Service is then on `http://localhost:8000`.

### Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | one key required | - | Groq key (console.groq.com). Starts `gsk_`. |
| `GEMINI_API_KEY` | one key required | - | Google AI Studio key (aistudio.google.com). |
| `GROQ_MODEL` | no | `openai/gpt-oss-120b,openai/gpt-oss-20b` | Comma-separated chain. |
| `GEMINI_MODEL` | no | `gemini-3.1-flash-lite,gemini-3.6-flash` | Comma-separated chain. |
| `LLM_TIMEOUT_SECONDS` | no | `8` | Per-call timeout. |
| `LLM_BUDGET_SECONDS` | no | `20` | Wall-clock budget across all providers and retries. |
| `LLM_MAX_OUTPUT_TOKENS` | no | `1200` | Output cap. |
| `KEEPALIVE_URL` | no | - | Own public `/health` URL; self-ping against idle spin-down. |
| `KEEPALIVE_INTERVAL_SECONDS` | no | `600` | Self-ping interval. |
| `LOG_LEVEL` | no | `INFO` | |
| `PORT` | no | `8000` | |

At least one of `GROQ_API_KEY` / `GEMINI_API_KEY` must be set. With both, the
service uses both. No key is ever logged, echoed in a response, or committed;
`.env` is gitignored.

A legacy generic scheme (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`,
`LLM_FALLBACK_*`) is still honoured, but **only when neither named key is
set**, so a stale variable cannot override a correct key.

### Models

Default chain, in order:

| # | Provider | Model | Free-tier limit (measured) |
|---|---|---|---|
| 1 | Groq | `openai/gpt-oss-120b` | 8,000 tokens/min, 1,000 req/day |
| 2 | Groq | `openai/gpt-oss-20b` | 8,000 tokens/min, 1,000 req/day |
| 3 | Google | `gemini-3.1-flash-lite` | separate daily bucket |
| 4 | Google | `gemini-3.6-flash` | 20 req/day |

Free-tier quota is metered per model, so the chain is four independent buckets,
not one. A 429 steps to the next link rather than retrying an exhausted one.

---

## Endpoints

### `GET /health`

```bash
curl -s https://gridwise-llm-vby1.onrender.com/health
```
```json
{"status": "ok"}
```

### `POST /optimize-energy`

```bash
curl -s -X POST https://gridwise-llm-vby1.onrender.com/optimize-energy \
  -H 'Content-Type: application/json' \
  -d '{
    "scenario_id": "DEMO-1",
    "operator_notes": [
      "Panel cleaning from 1 PM to 3 PM will leave about a fifth of normal solar output.",
      "Do not charge the battery between 2 PM and 4 PM.",
      "The cafeteria menu changes tomorrow."
    ],
    "hours": [
      {"hour": 0,  "demand_kwh": 90,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 6},
      {"hour": 1,  "demand_kwh": 85,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 6},
      {"hour": 2,  "demand_kwh": 80,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 5},
      {"hour": 3,  "demand_kwh": 80,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 5},
      {"hour": 4,  "demand_kwh": 85,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 5},
      {"hour": 5,  "demand_kwh": 95,  "solar_kwh": 0,   "tariff_bdt_per_kwh": 6},
      {"hour": 6,  "demand_kwh": 110, "solar_kwh": 5,   "tariff_bdt_per_kwh": 8},
      {"hour": 7,  "demand_kwh": 130, "solar_kwh": 20,  "tariff_bdt_per_kwh": 10},
      {"hour": 8,  "demand_kwh": 150, "solar_kwh": 50,  "tariff_bdt_per_kwh": 12},
      {"hour": 9,  "demand_kwh": 165, "solar_kwh": 90,  "tariff_bdt_per_kwh": 14},
      {"hour": 10, "demand_kwh": 175, "solar_kwh": 130, "tariff_bdt_per_kwh": 16},
      {"hour": 11, "demand_kwh": 180, "solar_kwh": 160, "tariff_bdt_per_kwh": 16},
      {"hour": 12, "demand_kwh": 185, "solar_kwh": 180, "tariff_bdt_per_kwh": 15},
      {"hour": 13, "demand_kwh": 180, "solar_kwh": 170, "tariff_bdt_per_kwh": 14},
      {"hour": 14, "demand_kwh": 170, "solar_kwh": 140, "tariff_bdt_per_kwh": 13},
      {"hour": 15, "demand_kwh": 165, "solar_kwh": 90,  "tariff_bdt_per_kwh": 14},
      {"hour": 16, "demand_kwh": 170, "solar_kwh": 45,  "tariff_bdt_per_kwh": 18},
      {"hour": 17, "demand_kwh": 185, "solar_kwh": 10,  "tariff_bdt_per_kwh": 22},
      {"hour": 18, "demand_kwh": 205, "solar_kwh": 0,   "tariff_bdt_per_kwh": 28},
      {"hour": 19, "demand_kwh": 215, "solar_kwh": 0,   "tariff_bdt_per_kwh": 30},
      {"hour": 20, "demand_kwh": 205, "solar_kwh": 0,   "tariff_bdt_per_kwh": 26},
      {"hour": 21, "demand_kwh": 175, "solar_kwh": 0,   "tariff_bdt_per_kwh": 18},
      {"hour": 22, "demand_kwh": 135, "solar_kwh": 0,   "tariff_bdt_per_kwh": 10},
      {"hour": 23, "demand_kwh": 105, "solar_kwh": 0,   "tariff_bdt_per_kwh": 7}
    ],
    "battery": {
      "capacity_kwh": 220, "initial_energy_kwh": 110, "minimum_energy_kwh": 40,
      "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50
    }
  }'
```

Abbreviated response:

```json
{
  "scenario_id": "DEMO-1",
  "directive_interpretation": [
    {"note_index": 0, "applies": true, "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
     "explanation": "Cleaning leaves about a fifth of solar usable."},
    {"note_index": 1, "applies": true, "directive_type": "no_charge_window",
     "structured_adjustment": {"hours": [14, 15]},
     "explanation": "Charging is not permitted in that window."},
    {"note_index": 2, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null,
     "explanation": "Catering does not affect today's schedule."}
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0,
     "battery_action": "idle", "battery_kwh": 0.0, "battery_energy_after_kwh": 110.0}
  ],
  "total_grid_kwh": 2430.0,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 215.0,
  "plan_summary": "Applied operator directives: no_charge_window, solar_reduction. ..."
}
```

### Status codes

| Code | When |
|---|---|
| 200 | Successful health or optimization response |
| 400 | Malformed JSON or structurally invalid request |
| 500 | Controlled internal error; no stack traces, no secrets |

FastAPI's default validation code is 422; it is overridden to 400 because the
specification requires 400 and lists 422 as optional.

---

## How it works

### 1. LLM interpreter (`app/interpreter.py`)

One call per scenario covering all 1-3 notes. The prompt states the six
permitted directive types and their exact shapes, the half-open hour convention
(start included, end excluded), and two rules that carry most of the difficulty:

- **`factor` is the fraction that REMAINS**, never the loss. "Falls to 30%"
  gives 0.30; "falls by 70%" also gives 0.30. These are stated as arithmetic
  rather than examples, because hidden notes paraphrase.
- **Reserves are always absolute kWh.** A note giving a percentage of capacity
  is converted using the capacity from the request.

Synonym families are listed per window directive, and the prompt states
explicitly that note position carries no meaning, so a distractor in first or
middle position is handled the same as one at the end. Few-shot examples were
written from scratch; no public sample wording is hard-coded anywhere.

Identical note sets are answered from a bounded in-process cache to conserve
free-tier quota.

### 2. Guardrails (`app/guardrails.py`)

Deterministic, pure functions. LLM output is untrusted structured data until it
passes:

- `directive_type` must be one of the six; anything else becomes `no_op`
- exactly one entry per note, `note_index` 0..N-1, duplicates dropped, missing
  entries filled with `no_op`
- hours: unique integers 0-23, deduplicated and sorted ascending; an
  out-of-range hour invalidates the window
- `factor` rejected unless 0 <= factor <= 1
- reserve must be finite and non-negative; above capacity it is clamped to
  capacity rather than discarded, preserving the operator's intent
- `max_grid_kwh` must be finite and non-negative
- `applies` is forced to `directive_type != "no_op"`
- demand, tariff and battery values are never read from model output -
  `structured_adjustment` is rebuilt from scratch, so invented keys cannot
  survive

Anything unrecoverable degrades that note to `no_op`. The service never crashes
and never invents a directive.

### 3. Optimizer (`app/optimizer.py`)

Linear program over 24 hours, solved with PuLP and the bundled CBC solver.

```
variables per hour h:
  g[h]  grid import       >= 0, <= max_grid_kwh where a cap applies
  s[h]  solar used        in [0, effective_solar[h]]
  c[h]  battery charge    in [0, max_charge], 0 in a no_charge_window
  d[h]  battery discharge in [0, max_discharge], 0 in a no_discharge_window
  e[h]  energy after h    in [max(base reserve, directive reserve), capacity]

minimize  SUM g[h] * tariff[h]
s.t.      g + s + d == demand + c
          e[h] == e[h-1] + c[h] - d[h]
          e[23] == initial_energy_kwh
```

No binary variables are needed. An LP can return `c[h]` and `d[h]` both
positive, but that cancels out of both the balance and state equations, so it
is netted afterwards into a single reportable `battery_action`. Keeping it a
pure LP holds solve time at roughly 4 ms.

Two deterministic post-processing steps: net simultaneous charge/discharge, and
raise `solar_used` to the maximum the hour can absorb (which only lowers grid
import, so it can never break a cap or raise cost).

**Feasibility ladder.** Scoring scenarios are guaranteed feasible, so an
infeasible model implies one of our own extractions is wrong. Rather than
discard every directive, the solver tries all of them, then every subset of
size n-1, then n-2, keeping as many as can be satisfied together. At most three
notes means at most eight solves. Last resort is a grid-only schedule, which is
feasible by construction because the battery idles.

### 4. Final validator (`app/validator.py`)

The finished schedule is replayed hour by hour before the response is built:
energy balance, battery transitions and bounds, rate limits, effective solar,
end-of-day neutrality, the reported totals recalculated from the plan, and each
applied directive checked directly against `hourly_plan`. Used both as the
service's own gate and as the local test oracle.

---

## Tests

```bash
python tests/test_guardrails.py    # 15 adversarial cases, no API key needed
python tests/test_optimizer.py     # optimizer vs reference optima, no API key needed
python tests/run_local.py          # full pipeline over the public pack, needs a key
python tests/test_paraphrase.py    # 10 unseen paraphrase cases, needs a key
```

Current results:

- **Public sample pack** - 10/10 cases, 18/18 notes interpreted correctly, every
  plan valid under replay, 0.00% cost gap against the published reference
  optimum on all ten.
- **Paraphrase stress set** - 10/10, using wording written independently of the
  public pack: words instead of digits, both solar phrasing directions, reserve
  as a percentage and as absolute kWh, window synonyms, distractors in first and
  middle position, single-hour and midnight-crossing windows.

The public cases are not the hidden judge set, and the stress set is our own
guess at how hidden notes vary.

---

## Docker

```bash
docker pull ghcr.io/abu-ruhan-mahamud/gridwise-llm:latest
docker run --rm -p 8000:8000 \
  -e GROQ_API_KEY=your_groq_key \
  -e GEMINI_API_KEY=your_gemini_key \
  ghcr.io/abu-ruhan-mahamud/gridwise-llm:latest
curl -s http://localhost:8000/health
```

Builds from `python:3.11-slim`, binds `0.0.0.0`, exposes 8000, honours `$PORT`.
No secrets are baked in; keys are passed at runtime.

To build locally:

```bash
docker build -t gridwise-llm:local .
docker run --rm -p 8000:8000 -e GROQ_API_KEY=... gridwise-llm:local
```

---

## Dependencies

`fastapi` 0.115.6 · `uvicorn[standard]` 0.34.0 · `pydantic` 2.10.4 ·
`httpx` 0.28.1 · `pulp` 2.9.0 (bundles the CBC solver; no system package
needed). Full pins in `requirements.txt`.

---

## Known limitations

- **Free-tier quota is the main operational risk.** Four independent model
  buckets and a response cache mitigate it, but sustained high request rates
  could still exhaust every link, in which case interpretation degrades to
  all-`no_op` and the schedule is still returned, valid, at base rules.
- **Degradation is silent to the caller by design.** A response never exposes
  provider errors. Degraded runs are visible in the service logs.
- **`gemini-3.6-flash` allows only 20 requests/day** on the free tier and is
  positioned last for that reason.
- **Overlapping `solar_reduction` directives multiply.** The specification does
  not define the interaction; multiplication was chosen because it is
  order-independent.
- **Overlapping reserves take the maximum and overlapping grid caps take the
  minimum**, i.e. the tightest constraint wins. Also not specified.
- **Free hosting sleeps when idle.** An external uptime pinger is the primary
  defence; the in-app self-ping is a second layer and cannot wake an instance
  that has already slept.
- Equal-cost schedules are not unique. The solver returns one optimal schedule;
  a different valid schedule at the same cost is equally correct.
