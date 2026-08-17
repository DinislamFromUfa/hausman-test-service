from pydantic import BaseModel, Field


class HausmanRequestSchema(BaseModel):
    fe_coefficients: list[float]
    re_coefficients: list[float]
    fe_covariance: list[list[float]]
    re_covariance: list[list[float]]
    alpha: float = Field(gt=0, lt=1)


class HausmanResponseSchema(BaseModel):
    statistic: float
    p_value: float
    degrees_of_freedom: int
    alpha: float
    reject_null: bool
