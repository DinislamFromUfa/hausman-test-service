from abc import ABC, abstractmethod

from domain.value_objects.coefficients import Coefficients
from domain.value_objects.covariance_matrix import CovarianceMatrix
from domain.value_objects.hausman_result import HausmanResult


class HausmanCalculatorPort(ABC):
    @abstractmethod
    def calculate(
        self,
        fe_coefficients: Coefficients,
        re_coefficients: Coefficients,
        fe_covariance: CovarianceMatrix,
        re_covariance: CovarianceMatrix,
        alpha: float,
    ) -> HausmanResult:
        pass
