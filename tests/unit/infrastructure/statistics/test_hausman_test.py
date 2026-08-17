import pytest

from domain.value_objects.coefficients import Coefficients
from domain.value_objects.covariance_matrix import CovarianceMatrix
from domain.value_objects.hausman_result import HausmanDecision
from infrastructure.statistics.hausman import HausmanTest

from infrastructure.statistics.exceptions import CovarianceMatrixNotInvertible


def test_hausman_test_calculates_result() -> None:
    fe_coefficients = Coefficients(
        values=(1.2, 0.8),
    )

    re_coefficients = Coefficients(
        values=(1.0, 0.5),
    )

    fe_covariance = CovarianceMatrix(
        values=(
            (0.5, 0.0),
            (0.0, 0.5),
        ),
    )

    re_covariance = CovarianceMatrix(
        values=(
            (0.3, 0.0),
            (0.0, 0.3),
        ),
    )

    result = HausmanTest(alpha=0.05).calculate(
        fe_coefficients=fe_coefficients,
        re_coefficients=re_coefficients,
        fe_covariance=fe_covariance,
        re_covariance=re_covariance,
    )

    assert result.statistic == pytest.approx(0.65)
    assert result.p_value == pytest.approx(0.7225273536)
    assert result.degrees_of_freedom == 2
    assert result.decision is HausmanDecision.DO_NOT_REJECT_H0


@pytest.mark.parametrize(
    "alpha",
    [-0.1, 0.0, 1.0, 1.5],
)
def test_alpha_must_be_between_zero_and_one(alpha: float) -> None:
    with pytest.raises(
        ValueError,
        match="Alpha must be between 0 and 1",
    ):
        HausmanTest(alpha=alpha)


def test_coefficients_must_have_same_dimension() -> None:
    fe_coefficients = Coefficients(
        values=(1.2, 0.8),
    )

    re_coefficients = Coefficients(
        values=(1.0, 0.5, 0.2),
    )

    fe_covariance = CovarianceMatrix(
        values=(
            (0.5, 0.0),
            (0.0, 0.5),
        ),
    )

    re_covariance = CovarianceMatrix(
        values=(
            (0.3, 0.0),
            (0.0, 0.3),
        ),
    )

    with pytest.raises(
        ValueError,
        match="FE and RE coefficients must have the same dimension",
    ):
        HausmanTest(alpha=0.05).calculate(
            fe_coefficients=fe_coefficients,
            re_coefficients=re_coefficients,
            fe_covariance=fe_covariance,
            re_covariance=re_covariance,
        )


def test_fe_covariance_dimension_must_match_coefficients() -> None:
    fe_coefficients = Coefficients(
        values=(1.2, 0.8),
    )

    re_coefficients = Coefficients(
        values=(1.0, 0.5),
    )

    fe_covariance = CovarianceMatrix(
        values=(
            (0.5, 0.0, 0.1),
            (0.0, 0.5, 0.2),
            (0.1, 0.2, 0.7),
        ),
    )

    re_covariance = CovarianceMatrix(
        values=(
            (0.3, 0.0),
            (0.0, 0.3),
        ),
    )

    with pytest.raises(
        ValueError,
        match="FE covariance matrix dimension must match coefficients",
    ):
        HausmanTest(alpha=0.05).calculate(
            fe_coefficients=fe_coefficients,
            re_coefficients=re_coefficients,
            fe_covariance=fe_covariance,
            re_covariance=re_covariance,
        )


def test_hausman_test_rejects_h0_when_p_value_is_small() -> None:
    fe_coefficients = Coefficients(
        values=(3.0, 4.0),
    )

    re_coefficients = Coefficients(
        values=(1.0, 1.0),
    )

    fe_covariance = CovarianceMatrix(
        values=(
            (0.2, 0.0),
            (0.0, 0.2),
        ),
    )

    re_covariance = CovarianceMatrix(
        values=(
            (0.1, 0.0),
            (0.0, 0.1),
        ),
    )

    result = HausmanTest(alpha=0.05).calculate(
        fe_coefficients=fe_coefficients,
        re_coefficients=re_coefficients,
        fe_covariance=fe_covariance,
        re_covariance=re_covariance,
    )

    assert result.p_value < 0.05
    assert result.decision is HausmanDecision.REJECT_H0


def test_re_covariance_dimension_must_match_coefficients() -> None:
    fe_coefficients = Coefficients(
        values=(1.2, 0.8),
    )

    re_coefficients = Coefficients(
        values=(1.0, 0.5),
    )

    fe_covariance = CovarianceMatrix(
        values=(
            (0.5, 0.0),
            (0.0, 0.5),
        ),
    )

    re_covariance = CovarianceMatrix(
        values=(
            (0.3, 0.0, 0.1),
            (0.0, 0.3, 0.2),
            (0.1, 0.2, 0.7),
        ),
    )

    with pytest.raises(
        ValueError,
        match="RE covariance matrix dimension must match coefficients",
    ):
        HausmanTest(alpha=0.05).calculate(
            fe_coefficients=fe_coefficients,
            re_coefficients=re_coefficients,
            fe_covariance=fe_covariance,
            re_covariance=re_covariance,
        )


def test_hausman_test_fails_when_covariance_difference_is_singular() -> None:
    fe_coefficients = Coefficients(
        values=(1.2, 0.8),
    )

    re_coefficients = Coefficients(
        values=(1.0, 0.5),
    )

    fe_covariance = CovarianceMatrix(
        values=(
            (0.5, 0.0),
            (0.0, 0.5),
        ),
    )

    re_covariance = CovarianceMatrix(
        values=(
            (0.5, 0.0),
            (0.0, 0.5),
        ),
    )

    with pytest.raises(
        CovarianceMatrixNotInvertible,
    ):
        HausmanTest(alpha=0.05).calculate(
            fe_coefficients=fe_coefficients,
            re_coefficients=re_coefficients,
            fe_covariance=fe_covariance,
            re_covariance=re_covariance,
        )
