from typing import List, Optional
from pydantic import BaseModel, Field


class Coeffs(BaseModel):
    alpha: List[float] = Field(..., min_items=2, description='Коэффициенты \u03B1')
    lmbda: List[float] = Field(..., min_items=2, alias='lambda', description='Коэффициенты \u03BB')
    p: List[float] = Field(..., min_items=2, description='Коэффициенты p')
    Q_max: Optional[List[float]] = None
    rho_c: Optional[List[float]] = None


class ModelParameters(BaseModel):
    rho_max: float = Field(..., ge=0, description='Максимально допустимое значение плотности, [АТС/м]')
    omega_safe_min: float = Field(..., ge=0, description='Минимально допустимое значение параметра \u03C9, [м/с]')
    omega_safe_max: float = Field(..., ge=0, description='Максимально допустимое значение параметра \u03C9, [м/с]')
    mean: float = Field(..., description='1-ое значение для нормировки, служебный параметр, см. калибровку ФД')
    std: float = Field(..., description='2-ое значение для нормировки, служебный параметр, см. калибровку ФД')
    coeffs: Coeffs = \
        Field(..., description='Коэффициенты полиномов, аппроксимирующих зависимости параметров ФД от \u03C9')
