import pytest
from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.di import HausmanProvider
from presentation.hausman.router import router


def create_app() -> FastAPI:
    app = FastAPI()

    container = make_async_container(HausmanProvider())
    setup_dishka(container, app)

    app.include_router(router)

    return app


def test_calculate_hausman_does_not_reject_null():
    client = TestClient(create_app())

    response = client.post(
        "/hausman",
        json={
            "fe_coefficients": [1.0, 2.0],
            "re_coefficients": [0.9, 1.8],
            "fe_covariance": [
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            "re_covariance": [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            "alpha": 0.05,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["statistic"] == pytest.approx(0.05)
    assert data["p_value"] == pytest.approx(0.9753099120283327)
    assert data["degrees_of_freedom"] == 2
    assert data["alpha"] == 0.05
    assert data["reject_null"] is False


def test_calculate_hausman_rejects_null():
    client = TestClient(create_app())

    response = client.post(
        "/hausman",
        json={
            "fe_coefficients": [3.0, 5.0],
            "re_coefficients": [1.0, 1.0],
            "fe_covariance": [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            "re_covariance": [
                [0.5, 0.0],
                [0.0, 0.5],
            ],
            "alpha": 0.05,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["statistic"] == pytest.approx(40.0)
    assert data["p_value"] < 0.05
    assert data["degrees_of_freedom"] == 2
    assert data["alpha"] == 0.05
    assert data["reject_null"] is True


def test_calculate_hausman_returns_400_when_covariance_difference_is_singular():
    client = TestClient(create_app())

    response = client.post(
        "/hausman",
        json={
            "fe_coefficients": [1.0, 2.0],
            "re_coefficients": [0.9, 1.8],
            "fe_covariance": [
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            "re_covariance": [
                [2.0, 0.0],
                [0.0, 1.0],
            ],
            "alpha": 0.05,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Covariance difference matrix is not invertible",
    }


@pytest.mark.parametrize(
    "alpha",
    [-0.1, 0.0, 1.0, 1.5],
)
def test_calculate_hausman_returns_422_for_invalid_alpha(alpha: float):
    client = TestClient(create_app())

    response = client.post(
        "/hausman",
        json={
            "fe_coefficients": [1.0, 2.0],
            "re_coefficients": [0.9, 1.8],
            "fe_covariance": [
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            "re_covariance": [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            "alpha": alpha,
        },
    )

    assert response.status_code == 422


def test_calculate_hausman_returns_422_for_invalid_coefficients():
    client = TestClient(create_app())

    response = client.post(
        "/hausman",
        json={
            "fe_coefficients": ["invalid", 2.0],
            "re_coefficients": [0.9, 1.8],
            "fe_covariance": [
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            "re_covariance": [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            "alpha": 0.05,
        },
    )

    assert response.status_code == 422


def test_calculate_hausman_returns_422_for_missing_required_field():
    client = TestClient(create_app())

    response = client.post(
        "/hausman",
        json={
            "fe_coefficients": [1.0, 2.0],
            "re_coefficients": [0.9, 1.8],
            "fe_covariance": [
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            "re_covariance": [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        },
    )

    assert response.status_code == 422
