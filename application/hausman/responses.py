from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HausmanResponse:
    statistic: float
    p_value: float
    degrees_of_freedom: int
    alpha: float
    reject_null: bool
