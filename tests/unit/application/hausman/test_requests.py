import pytest

from domain.value_objects.coefficients import Coefficients
from domain.value_objects.covariance_matrix import CovarianceMatrix
from application.hausman.requests import HausmanRequest


def test_hausman_request_stores_input_data() -> None:
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

    request = HausmanRequest(
        fe_coefficients=fe_coefficients,
        re_coefficients=re_coefficients,
        fe_covariance=fe_covariance,
        re_covariance=re_covariance,
        alpha=0.05,
    )

    assert request.fe_coefficients == fe_coefficients
    assert request.re_coefficients == re_coefficients
    assert request.fe_covariance == fe_covariance
    assert request.re_covariance == re_covariance
    assert request.alpha == 0.05


def test_hausman_request_is_immutable() -> None:
    request = HausmanRequest(
        fe_coefficients=Coefficients(values=(1.2, 0.8)),
        re_coefficients=Coefficients(values=(1.0, 0.5)),
        fe_covariance=CovarianceMatrix(
            values=(
                (0.5, 0.0),
                (0.0, 0.5),
            ),
        ),
        re_covariance=CovarianceMatrix(
            values=(
                (0.3, 0.0),
                (0.0, 0.3),
            ),
        ),
        alpha=0.05,
    )

    with pytest.raises(AttributeError):
        request.alpha = 0.01
