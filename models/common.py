from typing import List, Optional, Union
from typing_extensions import Literal
from pydantic import BaseModel, Field

from .graph import Graph


class Phase(BaseModel):
    phase_id: int = Field(..., description='ID')
    phase_time: float = Field(..., ge=0, description='Длительность (без промтактов), [с]')


class ProgramState(BaseModel):
    controller_id: int = Field(..., description='ID контроллера')
    remain_phases: List[Phase] = Field(..., description='Фазы, проигрываемые с прошлого цикла')


class Plan(BaseModel):
    controller_id: int = Field(..., description='ID контроллера')
    shift: float = Field(..., ge=0, description='Сдвиг фаз, [с]')
    phases: List[Phase] = Field(..., description='Фазы')


class TrafficFlowConstant(BaseModel):
    chain_id: int = Field(..., description='ID вершины')
    density: float = Field(..., ge=0, description='Плотность потока, [АТС/м]')
    intensity: float = Field(..., ge=0, description='Интенсивность потока, [АТС/с]')


class TrafficFlowRecorded(BaseModel):
    chain_id: int = Field(..., description='ID вершины')
    time: List[float] = Field(..., description='Время записи показаний детекторов, [с]')
    density: List[float] = Field(..., description='Плотность потока, [АТС/м]')
    intensity: List[float] = Field(..., description='Интенсивность потока, [АТС/с]')


TrafficFlow = Union[TrafficFlowConstant, TrafficFlowRecorded]


class BaseRequestBody(BaseModel):
    simulation_time: int = Field(..., ge=0, description='Время симуляции, [с]')
    dx: float = Field(15., ge=0., description='Размер ячейки в графе')
    graph: Graph = Field(..., description='Граф сети')
    traffic_flows: List[TrafficFlow] = Field(..., description='Параметры входных потоков')


class BaseSuccessBody(BaseModel):
    success: Literal[True]
