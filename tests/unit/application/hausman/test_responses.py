import pytest

from application.hausman.responses import HausmanResponse


def test_hausman_response_creation():
    response = HausmanResponse(
        statistic=12.5,
        p_value=0.01,
        degrees_of_freedom=3,
        alpha=0.05,
        reject_null=True,
    )

    assert response.statistic == 12.5
    assert response.p_value == 0.01
    assert response.degrees_of_freedom == 3
    assert response.alpha == 0.05
    assert response.reject_null is True


def test_hausman_response_is_frozen():
    response = HausmanResponse(
        statistic=12.5,
        p_value=0.01,
        degrees_of_freedom=3,
        alpha=0.05,
        reject_null=True,
    )

    with pytest.raises(AttributeError):
        response.statistic = 10.0
