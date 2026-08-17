import pytest

from domain.value_objects.hausman_result import HausmanDecision, HausmanResult


def test_hausman_result_can_be_created_with_valid_values() -> None:
    result = HausmanResult(
        statistic=12.43,
        p_value=0.002,
        degrees_of_freedom=3,
        decision=HausmanDecision.REJECT_H0,
    )

    assert result.statistic == 12.43
    assert result.p_value == 0.002
    assert result.degrees_of_freedom == 3
    assert result.decision is HausmanDecision.REJECT_H0


@pytest.mark.parametrize(
    "statistic",
    [-1.0, -0.001, -100.0],
)
def test_hausman_statistic_cannot_be_negative(
    statistic: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Hausman statistic cannot be negative",
    ):
        HausmanResult(
            statistic=statistic,
            p_value=0.05,
            degrees_of_freedom=3,
            decision=HausmanDecision.DO_NOT_REJECT_H0,
        )


@pytest.mark.parametrize(
    "p_value",
    [-0.1, -1.0, 1.1, 2.0],
)
def test_p_value_must_be_between_zero_and_one(
    p_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="p-value must be between 0 and 1",
    ):
        HausmanResult(
            statistic=10.0,
            p_value=p_value,
            degrees_of_freedom=3,
            decision=HausmanDecision.DO_NOT_REJECT_H0,
        )


@pytest.mark.parametrize(
    "degrees_of_freedom",
    [0, -1, -10],
)
def test_degrees_of_freedom_must_be_positive(
    degrees_of_freedom: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Degrees of freedom must be positive",
    ):
        HausmanResult(
            statistic=10.0,
            p_value=0.05,
            degrees_of_freedom=degrees_of_freedom,
            decision=HausmanDecision.DO_NOT_REJECT_H0,
        )


def test_p_value_can_be_zero() -> None:
    result = HausmanResult(
        statistic=10.0,
        p_value=0.0,
        degrees_of_freedom=3,
        decision=HausmanDecision.REJECT_H0,
    )

    assert result.p_value == 0.0


def test_p_value_can_be_one() -> None:
    result = HausmanResult(
        statistic=0.0,
        p_value=1.0,
        degrees_of_freedom=3,
        decision=HausmanDecision.DO_NOT_REJECT_H0,
    )

    assert result.p_value == 1.0


def test_hausman_result_is_immutable() -> None:
    result = HausmanResult(
        statistic=10.0,
        p_value=0.05,
        degrees_of_freedom=3,
        decision=HausmanDecision.DO_NOT_REJECT_H0,
    )

    with pytest.raises(AttributeError):
        result.p_value = 0.9


@pytest.mark.parametrize(
    "statistic",
    [float("nan"), float("inf"), float("-inf")],
)
def test_hausman_statistic_must_be_finite(
    statistic: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Hausman statistic must be finite",
    ):
        HausmanResult(
            statistic=statistic,
            p_value=0.05,
            degrees_of_freedom=3,
            decision=HausmanDecision.DO_NOT_REJECT_H0,
        )


@pytest.mark.parametrize(
    "p_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_p_value_must_be_finite(
    p_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="p-value must be finite",
    ):
        HausmanResult(
            statistic=10.0,
            p_value=p_value,
            degrees_of_freedom=3,
            decision=HausmanDecision.DO_NOT_REJECT_H0,
        )
