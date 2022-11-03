from typing import List
from pydantic import BaseModel, Field

from .model_parameters import ModelParameters


class Vertex(BaseModel):
    chain_id: int = Field(..., description='ID')
    length: float = Field(..., ge=0, description='Длина, [м]')


class Link(BaseModel):
    link_id: int = Field(..., description='ID')
    input_chain_id: int = Field(..., description='ID вершины, из которой выходит это ребро')
    output_chain_id: int = Field(..., description='ID вершины, в которую выходит это ребро')
    weight: float = Field(..., ge=0, le=1, description='Пропускная способность')


class Phase(BaseModel):
    phase_id: int = Field(..., description='ID')
    green_links: List[int] = Field(..., description='"Открытые" ребра в эту фазу')


class Controller(BaseModel):
    controller_id: int = Field(..., description='ID')
    phases: List[Phase] = Field(..., description='Фазы')


class Graph(BaseModel):
    vertices: List[Vertex] = Field(..., description='Вершины (участки дороги)')
    links: List[Link] = Field(..., description='Ребра (связи между вершинами)')
    controllers: List[Controller] = Field(..., description='Контроллеры (светофоры)')
    model_parameters: ModelParameters = Field(..., description='Параметры модели')
