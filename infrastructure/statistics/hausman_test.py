from scipy.stats import chi2
import numpy as np

from domain.value_objects.coefficients import Coefficients
from domain.value_objects.covariance_matrix import CovarianceMatrix
from domain.value_objects.hausman_result import HausmanDecision, HausmanResult

from infrastructure.statistics.exceptions import CovarianceMatrixNotInvertible


class HausmanTest:
    def __init__(self, alpha: float) -> None:
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")

        self._alpha = alpha

    def calculate(
        self,
        fe_coefficients: Coefficients,
        re_coefficients: Coefficients,
        fe_covariance: CovarianceMatrix,
        re_covariance: CovarianceMatrix,
    ) -> HausmanResult:
        self._validate_dimensions(
            fe_coefficients,
            re_coefficients,
            fe_covariance,
            re_covariance,
        )

        fe = np.asarray(fe_coefficients.values)
        re = np.asarray(re_coefficients.values)

        fe_cov = np.asarray(fe_covariance.values)
        re_cov = np.asarray(re_covariance.values)

        coefficient_difference = fe - re
        covariance_difference = fe_cov - re_cov

        try:
            statistic = float(
                coefficient_difference
                @ np.linalg.solve(
                    covariance_difference,
                    coefficient_difference,
                )
            )
        except np.linalg.LinAlgError as error:
            raise CovarianceMatrixNotInvertible(
                "Covariance difference matrix is not invertible"
            ) from error

        degrees_of_freedom = len(fe_coefficients.values)

        p_value = float(
            chi2.sf(
                statistic,
                degrees_of_freedom,
            )
        )

        decision = (
            HausmanDecision.REJECT_H0
            if p_value <= self._alpha
            else HausmanDecision.DO_NOT_REJECT_H0
        )

        return HausmanResult(
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=degrees_of_freedom,
            decision=decision,
        )

    @staticmethod
    def _validate_dimensions(
        fe_coefficients: Coefficients,
        re_coefficients: Coefficients,
        fe_covariance: CovarianceMatrix,
        re_covariance: CovarianceMatrix,
    ) -> None:
        coefficients_count = len(fe_coefficients.values)

        if len(re_coefficients.values) != coefficients_count:
            raise ValueError(
                "FE and RE coefficients must have the same dimension"
            )

        if len(fe_covariance.values) != coefficients_count:
            raise ValueError(
                "FE covariance matrix dimension must match coefficients"
            )

        if len(re_covariance.values) != coefficients_count:
            raise ValueError(
                "RE covariance matrix dimension must match coefficients"
            )
