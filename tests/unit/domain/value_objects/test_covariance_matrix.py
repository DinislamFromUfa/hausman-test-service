import pytest

from domain.value_objects.covariance_matrix import CovarianceMatrix


def test_covariance_matrix_can_be_created_with_valid_values() -> None:
    matrix = CovarianceMatrix(
        values=(
            (1.2, 0.3),
            (0.3, 0.8),
        ),
    )

    assert matrix.values == (
        (1.2, 0.3),
        (0.3, 0.8),
    )


def test_covariance_matrix_cannot_be_empty() -> None:
    with pytest.raises(
        ValueError,
        match="Covariance matrix cannot be empty",
    ):
        CovarianceMatrix(values=())


@pytest.mark.parametrize(
    "values",
    [
        (
            (1.0, 0.2),
            (0.3,),
        ),
        (
            (1.0, 0.2, 0.3),
            (0.2, 1.0),
        ),
        (
            (1.0,),
            (0.2, 1.0),
        ),
    ],
)
def test_covariance_matrix_must_be_square(
    values: tuple[tuple[float, ...], ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="Covariance matrix must be square",
    ):
        CovarianceMatrix(values=values)


@pytest.mark.parametrize(
    "values",
    [
        (
            (float("nan"), 0.0),
            (0.0, 1.0),
        ),
        (
            (float("inf"), 0.0),
            (0.0, 1.0),
        ),
        (
            (1.0, 0.0),
            (0.0, float("-inf")),
        ),
    ],
)
def test_covariance_matrix_must_contain_only_finite_values(
    values: tuple[tuple[float, ...], ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="Covariance matrix must contain only finite values",
    ):
        CovarianceMatrix(values=values)


def test_covariance_matrix_is_immutable() -> None:
    matrix = CovarianceMatrix(
        values=(
            (1.2, 0.3),
            (0.3, 0.8),
        ),
    )

    with pytest.raises(AttributeError):
        matrix.values = (
            (10.0, 20.0),
            (20.0, 30.0),
        )
