from dataclasses import dataclass

from domain.value_objects.coefficients import Coefficients
from domain.value_objects.covariance_matrix import CovarianceMatrix


@dataclass(frozen=True, slots=True)
class HausmanRequest:
    fe_coefficients: Coefficients
    re_coefficients: Coefficients
    fe_covariance: CovarianceMatrix
    re_covariance: CovarianceMatrix
    alpha: float
