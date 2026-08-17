from pydantic import BaseModel, Field


class HausmanRequestSchema(BaseModel):
    fe_coefficients: list[float] = Field(min_length=1)
    re_coefficients: list[float] = Field(min_length=1)
    fe_covariance: list[list[float]] = Field(min_length=1)
    re_covariance: list[list[float]] = Field(min_length=1)
    alpha: float = Field(gt=0, lt=1)


class HausmanResponseSchema(BaseModel):
    statistic: float
    p_value: float
    degrees_of_freedom: int
    alpha: float
    reject_null: bool
