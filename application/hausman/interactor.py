from application.hausman.requests import HausmanRequest
from application.hausman.responses import HausmanResponse
from domain.ports.hausman_calculator import HausmanCalculatorPort
from domain.value_objects.hausman_result import HausmanDecision


class HausmanInteractor:
    def __init__(self, calculator: HausmanCalculatorPort) -> None:
        self._calculator = calculator

    def execute(self, request: HausmanRequest) -> HausmanResponse:
        result = self._calculator.calculate(
            fe_coefficients=request.fe_coefficients,
            re_coefficients=request.re_coefficients,
            fe_covariance=request.fe_covariance,
            re_covariance=request.re_covariance,
        )

        return HausmanResponse(
            statistic=result.statistic,
            p_value=result.p_value,
            degrees_of_freedom=result.degrees_of_freedom,
            alpha=request.alpha,
            reject_null=result.decision is HausmanDecision.REJECT_H0,
        )
