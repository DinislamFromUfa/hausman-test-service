import pytest

from domain.value_objects.coefficients import Coefficients


def test_coefficients_can_be_created_with_valid_values() -> None:
    coefficients = Coefficients(
        values=(1.2, -0.5, 3.7),
    )

    assert coefficients.values == (1.2, -0.5, 3.7)


def test_coefficients_cannot_be_empty() -> None:
    with pytest.raises(ValueError, match="Coefficients cannot be empty"):
        Coefficients(values=())


@pytest.mark.parametrize(
    "values",
    [
        (float("nan"),),
        (float("inf"),),
        (float("-inf"),),
        (1.2, float("nan"), 3.4),
        (1.2, float("inf"), 3.4),
    ],
)
def test_coefficients_must_contain_only_finite_values(
    values: tuple[float, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="Coefficients must contain only finite values",
    ):
        Coefficients(values=values)


def test_coefficients_are_immutable() -> None:
    coefficients = Coefficients(
        values=(1.2, -0.5, 3.7),
    )

    with pytest.raises(AttributeError):
        coefficients.values = (10.0, 20.0, 30.0)
