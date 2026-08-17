from application.hausman.interactor import HausmanInteractor
from application.hausman.requests import HausmanRequest
from domain.value_objects.coefficients import Coefficients
from domain.value_objects.covariance_matrix import CovarianceMatrix
from domain.value_objects.hausman_result import HausmanDecision, HausmanResult


class FakeHausmanCalculator:
    def __init__(self, result: HausmanResult) -> None:
        self.result = result
        self.called_with = None

    def calculate(
        self,
        fe_coefficients: Coefficients,
        re_coefficients: Coefficients,
        fe_covariance: CovarianceMatrix,
        re_covariance: CovarianceMatrix,
        alpha: float,
    ) -> HausmanResult:
        self.called_with = {
            "fe_coefficients": fe_coefficients,
            "re_coefficients": re_coefficients,
            "fe_covariance": fe_covariance,
            "re_covariance": re_covariance,
        }
        return self.result


def make_request() -> HausmanRequest:
    return HausmanRequest(
        fe_coefficients=Coefficients((1.0, 2.0)),
        re_coefficients=Coefficients((0.9, 1.8)),
        fe_covariance=CovarianceMatrix(((2.0, 0.0), (0.0, 2.0))),
        re_covariance=CovarianceMatrix(((1.0, 0.0), (0.0, 1.0))),
        alpha=0.05,
    )


def test_execute_returns_response():
    result = HausmanResult(
        statistic=5.2,
        p_value=0.02,
        degrees_of_freedom=2,
        decision=HausmanDecision.REJECT_H0,
    )
    calculator = FakeHausmanCalculator(result)
    interactor = HausmanInteractor(calculator)

    response = interactor.execute(make_request())

    assert response.statistic == 5.2
    assert response.p_value == 0.02
    assert response.degrees_of_freedom == 2
    assert response.alpha == 0.05
    assert response.reject_null is True


def test_execute_sets_reject_null_to_false_when_h0_is_not_rejected():
    result = HausmanResult(
        statistic=1.2,
        p_value=0.55,
        degrees_of_freedom=2,
        decision=HausmanDecision.DO_NOT_REJECT_H0,
    )
    calculator = FakeHausmanCalculator(result)
    interactor = HausmanInteractor(calculator)

    response = interactor.execute(make_request())

    assert response.reject_null is False


def test_execute_passes_request_data_to_calculator():
    request = make_request()
    result = HausmanResult(
        statistic=5.2,
        p_value=0.02,
        degrees_of_freedom=2,
        decision=HausmanDecision.REJECT_H0,
    )
    calculator = FakeHausmanCalculator(result)
    interactor = HausmanInteractor(calculator)

    interactor.execute(request)

    assert calculator.called_with == {
        "fe_coefficients": request.fe_coefficients,
        "re_coefficients": request.re_coefficients,
        "fe_covariance": request.fe_covariance,
        "re_covariance": request.re_covariance,
    }
