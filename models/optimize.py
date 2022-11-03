from enum import Enum

from typing import List, Optional
from pydantic import BaseModel, Field

from .common import ProgramState, Plan, BaseRequestBody, BaseSuccessBody
from .traffic_state import TrafficState


class ModeEnum(str, Enum):
    free = 'free'
    strict_cycle = 'strict_cycle'
    multiple_cycle = 'multiple_cycle'
    bounded_cycle = 'bounded_cycle'


class ProgramBoundPhase(BaseModel):
    phase_id: int = Field(..., description='ID')
    phase_min: float = Field(..., ge=0, description='Минимальная длительность, [с]')
    phase_max: float = Field(..., ge=0, description='Максимальная длительность, [с]')


class ProgramBound(BaseModel):
    controller_id: int = Field(..., description='ID контроллера')
    mode: ModeEnum = Field(None, description='Тип плана')
    shift: float = Field(..., ge=0, description='Сдвиг, [с]')
    cycle_time: Optional[float] = Field(None, ge=0, description='Длительность цикла, [с]')
    cycle_min: Optional[float] = Field(None, ge=0, description='Минимальная длительность цикла, [с]')
    cycle_max: Optional[float] = Field(None, ge=0, description='Максимальная длительность цикла, [с]')
    phases: List[ProgramBoundPhase] = Field(..., description='Фазы')


class LocalSearchResult(BaseModel):
    plan: List[Plan]
    times: List[float]
    scores: List[float]
    best_scores: List[float]


class OptimizeRequestBody(BaseRequestBody):
    optimization_duration: int = Field(..., ge=0, description='Допустимая длительность оптимизации, [с]')
    programs_bounds: List[ProgramBound] = Field(..., description='Допустимые границы для фазных распределений')
    plan: List[Plan] = Field(..., description='Описание планов', alias='init_plan')
    programs_states: List[ProgramState] = Field(..., description='Начальные состояния программ на контроллерах')
    init_traffic_state: Optional[TrafficState] = Field(description='Начальное распределение АТС в сети')


class OptimizeResponseBody(BaseSuccessBody):
    plan: List[Plan] = Field(..., description='Новые значения оптимальных параметров')
    convergence: List[float] = Field(..., description='Значения целевой функции')
    local_search_result: Optional[LocalSearchResult]
