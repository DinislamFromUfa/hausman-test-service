from dataclasses import dataclass
from enum import Enum


class HausmanDecision(Enum):
    DO_NOT_REJECT_H0 = "do_not_reject_h0"
    REJECT_H0 = "reject_h0"


@dataclass(frozen=True, slots=True)
class HausmanResult:
    statistic: float
    p_value: float
    degrees_of_freedom: int
    decision: HausmanDecision

    def __post_init__(self) -> None:
        if self.statistic < 0:
            raise ValueError("Hausman statistic cannot be negative")

        if not 0 <= self.p_value <= 1:
            raise ValueError("p-value must be between 0 and 1")

        if self.degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be positive")
