from optparse import Option
from typing import List, Optional, Any
from pydantic import BaseModel, Field

from .common import ProgramState, Plan, BaseRequestBody, BaseSuccessBody
from .traffic_state import TrafficState


class TrafficStateParams(BaseModel):
    is_binary: bool = Field(False, description='Формат ответа')


class OutputCarsParameters(BaseModel):
    period: int = Field(5, description='Время между измерениями, [с]')


class VelocitiesParameters(BaseModel):
    period: int = Field(5, description='Время между измерениями, [с]')


class TrafficSnapshotParameters(BaseModel):
    segment_length: int = Field(100, description='Длина сегмента, [м]')


# DEPRECATED
class VelocitiesPartitionedParameters(BaseModel):
    segment_length: int = Field(100, description='Длина сегмента, [м]')


class CarsParameters(BaseModel):
    period: int = Field(5, description='Время между измерениями, [с]')


class TrafficHistParameters(BaseModel):
    period: int = Field(5, description='Время между измерениями, [с]')


class PressureParameters(BaseModel):
    period: int = Field(5, description='Время между измерениями, [с]')


class CritDensityScoreParameters(BaseModel):
    period: int = Field(5, description='Время между измерениями, [с]')


class AverageIntensityScoreParameters(BaseModel):
    period: int = Field(5, description='Время между измерениями, [с]')


class StatsParameters(BaseModel):
    # TODO: подогнать интерфейс на фронте и сделать поле traffic_state опциональным
    traffic_state: TrafficStateParams = Field(TrafficStateParams(), description='Распределение АТС в сети')
    output_cars: Optional[OutputCarsParameters] = Field(description='Машины, покидающие область')
    velocities: Optional[VelocitiesParameters] = Field(description='Средние скорость потоков')
    traffic_snapshot_partitioned: Optional[TrafficSnapshotParameters] = Field(
        description='Параметры потоков с детальным разбиением по линкам'
    )
    # DEPRECATED
    velocities_partitioned: Optional[VelocitiesPartitionedParameters] = Field(
        description='Скорость потоков с детальным разбиением по линкам'
    )
    cars: Optional[CarsParameters] = Field(description='Число машин, покинувших ребро')
    traffic_hist: Optional[TrafficHistParameters] = Field(description='Параметры записи полной истории симуляции')
    pressure: Optional[PressureParameters]
    density: Optional[CritDensityScoreParameters]
    average_intensity_score: Optional[AverageIntensityScoreParameters]


class SimulateRequestBody(BaseRequestBody):
    plan: List[Plan] = Field(..., description='Описание планов')
    programs_states: List[ProgramState] = Field(..., description='Начальные состояния программ на контроллерах')
    init_traffic_state: Optional[TrafficState] = Field(description='Начальное распределение АТС в сети')
    stats_parameters: StatsParameters = Field(description='Параметры сбора статистики')


class Estimate(BaseModel):
    cars: float = Field(..., ge=0, description='Количество машин, [АТС]')
    time: float = Field(..., ge=0, description='Время, [с]')


class VelocityStats(BaseModel):
    chain_id: int = Field(..., description='Идентификатор линка')
    velocity: float = Field(..., ge=0, description='Скорость потока на линке, [м/с]')


class TrafficSnapshot(BaseModel):
    chain_id: int = Field(..., description='Идентификатор линка')
    offset: List[float] = Field(..., ge=0, description='Сдвиги сегментов, [м]')
    velocity: List[float] = Field(..., ge=0, description='Скорости потока на линке, [м/с]')
    occupancy: List[float] = Field(..., ge=0, description='Загруженность линка, [десятичная дробь]')
    intensity: List[float] = Field(..., ge=0, description='Интенсивность потока на линке, [атс/с]')


# DEPRECATED
class VelocityPartitionedStats(BaseModel):
    chain_id: int = Field(..., description='Идентификатор линка')
    offset: List[float] = Field(..., ge=0, description='Сдвиги сегментов, [м]')
    velocity: List[float] = Field(..., ge=0, description='Скорости потока на линке, [м/с]')


class CarsStats(BaseModel):
    chain_id: int = Field(..., description='Идентификатор линка')
    cars: float = Field(..., ge=0, description='Число машин, покинувших линк, [АТС]')


class LaneSnapshot(BaseModel):
    chain_id: int
    dx: float
    density: List[float]
    velocity: List[float]
    flow: List[float]


class ControllerSnapshot(BaseModel):
    controller_id: int
    phase_id: Optional[int]
    open_links: Optional[List[int]]


class InputFlowSnapshot(BaseModel):
    chain_id: int
    density: float
    intensity: Optional[float]
    velocity: Optional[float]
    omega: Optional[float]


class SystemSnapshot(BaseModel):
    time: float
    lanes: List[LaneSnapshot]
    controllers: List[ControllerSnapshot]
    input_flows: List[InputFlowSnapshot]


class PhasePressure(BaseModel):
    phase_id: int
    pressure: List[float]


class CtrlPressure(BaseModel):
    ctrl_id: int
    phases: List[PhasePressure]


class NetworkPressure(BaseModel):
    times: List[float]
    controllers: List[CtrlPressure]


class CritDensityScore(BaseModel):
    crit_density_score: float


class AverageIntensityScore(BaseModel):
    average_intensity: float


class SimulateResponseBody(BaseSuccessBody):
    traffic_state: Optional[TrafficState] = Field(description='Распределение АТС в сети')
    output_cars: Optional[List[Estimate]] = Field(description='Динамика числа АТС, покидающих область')
    velocities: Optional[List[VelocityStats]] = Field(description='Средние скорости потока на линках графа')
    traffic_snapshot_partitioned: Optional[List[TrafficSnapshot]] = Field(
        description='Параметры потока на линках графа, с разбиением на сегменты'
    )
    # DEPRECATED
    velocities_partitioned: Optional[List[VelocityPartitionedStats]] = Field(
        description='Параметры потока на линках графа, с разбиением на сегменты'
    )
    cars: Optional[List[CarsStats]] = Field(description='Число машин, покинувших линк')
    avg_velocity: Optional[float] = Field(description='Средняя скорость потока в сети, [м/с]')
    traffic_hist: Optional[List[SystemSnapshot]] = Field(description='Полная запись динамики системы')
    pressure: Optional[NetworkPressure]
    density: Optional[CritDensityScore]
    average_intensity_score: Optional[AverageIntensityScore]
