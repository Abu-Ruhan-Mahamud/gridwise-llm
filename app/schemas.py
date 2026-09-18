"""Request/response models mirroring Problem Statement Sections 07 and 10.

Design notes:
  * Input numerics are only required to be finite (allow_inf_nan=False), not
    non-negative. Rejecting a well-formed judge case scores 0 for that case,
    while accepting an odd one still lets us return a plan - the risk is
    asymmetric, so validation here is deliberately permissive.
  * structured_adjustment is typed as a plain dict at the API boundary. Its
    per-directive shape (Section 04) is enforced by the deterministic guardrail
    layer, which is where the spec says that check belongs (Section 08).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HOURS_IN_DAY = 24
TOLERANCE = 0.01  # Section 11.5

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]

SUPPORTED_DIRECTIVES = frozenset(
    {
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    }
)


# --------------------------------------------------------------------------
# Request (Section 07)
# --------------------------------------------------------------------------


class HourInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float
    solar_kwh: float
    tariff_bdt_per_kwh: float


class BatterySpec(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    capacity_kwh: float = Field(..., ge=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    scenario_id: str
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourInput] = Field(..., min_length=HOURS_IN_DAY, max_length=HOURS_IN_DAY)
    battery: BatterySpec

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, v: List[str]) -> List[str]:
        for i, note in enumerate(v):
            if not note or not note.strip():
                raise ValueError(f"operator_notes[{i}] must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _hours_cover_full_day(self) -> "OptimizeRequest":
        seen = [h.hour for h in self.hours]
        if sorted(seen) != list(range(HOURS_IN_DAY)):
            raise ValueError("hours must contain exactly one entry for each hour 0 through 23")
        return self

    def hours_in_order(self) -> List[HourInput]:
        """Hours sorted by hour index; the request is not required to be ordered."""
        return sorted(self.hours, key=lambda h: h.hour)


# --------------------------------------------------------------------------
# Response (Section 10)
# --------------------------------------------------------------------------


class DirectiveInterpretation(BaseModel):
    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[Dict[str, Any]] = None
    explanation: str


class HourPlan(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
