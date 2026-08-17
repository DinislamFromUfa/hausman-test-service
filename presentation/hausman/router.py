from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, HTTPException

from application.hausman.interactor import HausmanInteractor
from application.hausman.requests import HausmanRequest

from domain.value_objects.coefficients import Coefficients
from domain.value_objects.covariance_matrix import CovarianceMatrix

from infrastructure.statistics.exceptions import CovarianceMatrixNotInvertible
from presentation.hausman.schemas import (
    HausmanRequestSchema,
    HausmanResponseSchema,
)


router = APIRouter()


@router.post(
    "/hausman",
    response_model=HausmanResponseSchema,
)
@inject
def calculate_hausman(
    request: HausmanRequestSchema,
    interactor: FromDishka[HausmanInteractor],
) -> HausmanResponseSchema:
    hausman_request = HausmanRequest(
        fe_coefficients=Coefficients(tuple(request.fe_coefficients)),
        re_coefficients=Coefficients(tuple(request.re_coefficients)),
        fe_covariance=CovarianceMatrix(
            tuple(tuple(row) for row in request.fe_covariance)
        ),
        re_covariance=CovarianceMatrix(
            tuple(tuple(row) for row in request.re_covariance)
        ),
        alpha=request.alpha,
    )

    try:
        response = interactor.execute(hausman_request)
    except CovarianceMatrixNotInvertible as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return HausmanResponseSchema(
        statistic=response.statistic,
        p_value=response.p_value,
        degrees_of_freedom=response.degrees_of_freedom,
        alpha=response.alpha,
        reject_null=response.reject_null,
    )
