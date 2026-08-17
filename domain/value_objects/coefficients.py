from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class Coefficients:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("Coefficients cannot be empty")

        if not all(isfinite(value) for value in self.values):
            raise ValueError("Coefficients must contain only finite values")
