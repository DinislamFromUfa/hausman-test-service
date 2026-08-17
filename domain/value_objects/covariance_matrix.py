from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class CovarianceMatrix:
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("Covariance matrix cannot be empty")

        size = len(self.values)

        if any(len(row) != size for row in self.values):
            raise ValueError("Covariance matrix must be square")

        if not all(
            isfinite(value)
            for row in self.values
            for value in row
        ):
            raise ValueError(
                "Covariance matrix must contain only finite values"
            )
