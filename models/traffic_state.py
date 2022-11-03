from typing import List
from pydantic import BaseModel, Field


class DistributionVertex(BaseModel):
    omega: float = Field(..., ge=0, description='Значение параметра \u03C9, [м/с]')
    rho: float = Field(..., ge=0, description='Значение плотности, [АТС/м]')


class Distribution(BaseModel):
    chain_id: int = Field(..., description='ID вершины')
    vertices: List[DistributionVertex] = Field(..., description='Вершины')


class TrafficState(BaseModel):
    distribution: List[Distribution] = Field(..., description='Распределение')
