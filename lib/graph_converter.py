import os
import sys
import xml.etree.ElementTree as ET
import pathlib

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib.net import Net

import re
import json
from collections import defaultdict
from typing import List, Set, Dict, Any, Tuple, Optional

from models.common import Plan, Graph

class GraphConverter:

    def __init__(self, net: Net, fd_path: Optional[os.PathLike] = None):
        # вершины графа
        self.chains = build_chains(net)
        # граничные ноды графа
        self.end_nodes = get_end_nodes(net)
        # ребра графа
        self.links = build_links(net, self.chains, self.end_nodes)
        # убираем болтающиеся дороги
        self.clean_dangling_chains()
        self.input_chains, self.output_chains = get_boundary_chains(net, self.chains, self.end_nodes)
        # контроллеры и программы
        self.controllers = build_controllers(net, self.links)
        self.programs, self.phases_index = build_plan(net, self.controllers)
        # загружаем фундаментальную диаграмму
        if fd_path is None:
            curr_dir = pathlib.Path(__file__).parent.resolve()
            fd_path = curr_dir.parent / 'data/macro/old_fundamental_diagram.json'
        with open(fd_path) as fd_file:
            self.fd = json.load(fd_file)


    def clean_dangling_chains(self):
        in_chains = set(link['input_chain_id'] for link in self.links.values())
        out_chains = set(link['output_chain_id'] for link in self.links.values())
        chains_dict = {chain['chain_id']: lane_id for lane_id, chain in self.chains.items()}
        dangling_chains = set(chains_dict.keys()) - in_chains - out_chains

        for dangling_chain_id in dangling_chains:
            lane_id = chains_dict[dangling_chain_id]
            del self.chains[lane_id]


    def get_graph(self) -> Graph:
        graph = Graph.parse_obj({
            'vertices': list(self.chains.values()),
            'links': list(self.links.values()),
            'controllers': list(self.controllers.values()),
            'model_parameters': self.fd,
        })
        return graph


    def to_dict(self) -> dict:
        graph_dict = self.__dict__
        graph_dict['end_nodes'] = list(graph_dict['end_nodes'])
        return graph_dict


    @classmethod
    def from_dict(cls, d: dict):
        graph_converter =  cls.__new__(cls)
        for key, value in d.items():
            setattr(graph_converter, key, value)
        graph_converter.end_nodes = set(graph_converter.end_nodes)
        return graph_converter


    def get_plan(self) -> List[Plan]:
        return [Plan.parse_obj(prog) for prog in self.programs.values()]



def build_chains(net: Net):
    """
    Построения списка вершин в графе
    """
    chains = {}
    counter = 1 # счетчик для численных id
    for edge in net.getEdges():
        # будем использовать только normal edges
        if edge.getFunction() not in ['', 'normal']:
            continue
        # формируем список полос
        for lane in edge.getLanes():
            chain_data = {
                'chain_id': counter,
                'length': lane.getLength(),
            }
            chains[lane.getID()] = chain_data
            counter += 1
    return chains


def get_end_nodes(net: Net) -> Set[str]:
    """
    Поиск граничных нод на графе
    """
    end_nodes = set(node.getID() for node in net.getNodes() if node.getParam('end_node'))
    return end_nodes


def get_boundary_chains(net: Net, chains: Dict[str, Any], end_nodes: Set[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    input_chains = {}
    output_chains = {}
    for edge in net.getEdges():

        from_node_id = edge.getFromNode().getID()
        if from_node_id in end_nodes:
            for lane in edge.getLanes():
                lane_id = lane.getID()
                if lane_id in chains:
                    input_chains[lane_id] = chains[lane_id]['chain_id']

        to_node_id = edge.getToNode().getID()
        if to_node_id in end_nodes:
            for lane in edge.getLanes():
                lane_id = lane.getID()
                if lane_id in chains:
                    output_chains[lane_id] = chains[lane_id]['chain_id']

    return input_chains, output_chains


def build_links(net, chains, end_nodes: Set[int]):
    """
    Построение списка линко в графе
    """
    links = {}
    counter = 1 # счетчик для численных id
    for node in net.getNodes():
        if node.getID() in end_nodes:
            continue
        for conn in node.getConnections():
            # conn_id = f'{node.getID()}_{conn.getTLLinkIndex()}'
            conn_id = f'{node.getID()}_{conn.getJunctionIndex()}'
            input_chain = chains[conn.getFromLane().getID()]
            output_chain = chains[conn.getToLane().getID()]
            link_data = {
                'link_id': counter,
                'input_chain_id': input_chain['chain_id'],
                'output_chain_id': output_chain['chain_id'],
                'weight': 1., # TODO: адекватный алгоритм для оценки весов поворотов
            }
            links[conn_id] = link_data
            counter += 1

    # WARN: временный костыль для корректировки веса линков
    # группируем линки
    input_chains_dict = defaultdict(list)
    for link in links.values():
        input_chains_dict[link['input_chain_id']].append(link)
    # корректируем веса
    for links_arr in input_chains_dict.values():
        weight = 1. / len(links_arr)
        for link in links_arr:
            link['weight'] = weight

    return links


def _get_green_links(node_id, state, links):
    green_links = []
    for link_idx, link_state in enumerate(state):
        if link_state not in ['g', 'G', 's']:
            continue
        link_id = f'{node_id}_{link_idx}'
        green_links.append(links.get(link_id)['link_id'])
    return green_links


# WARN: предполагается, что состояния основных тактов состоят только из флагов r,g,G,s
# main_tact_regex = re.compile(r'^[rsgG]*$')
main_tact_regex = re.compile(r'^\d+$')

def build_controllers(net: Net, links) -> Dict[str, Any]:
    controllers = {}
    ctrl_counter = 1
    phase_counter = 1
    for tls in net.getTrafficLights():
        node_id = tls.getID()
        phases = []
        for phase in tls.getPrograms()['default'].getPhases():
            # is_main_tact = bool(main_tact_regex.match(phase.state))
            is_main_tact = bool(main_tact_regex.match(phase.name))
            if is_main_tact:
                phases.append({
                    'phase_id': phase_counter,
                    'phase_number': int(re.findall(r'^\d+', phase.name)[0]),
                    'green_links': _get_green_links(node_id, phase.state, links),
                })
                phase_counter += 1
        controllers[node_id] = {
            'controller_id': ctrl_counter,
            'phases': phases,
        }
        ctrl_counter +=1
    return controllers


def _get_program_offset(prog_xml):
    root = ET.fromstring(prog_xml)
    return int(float(root.attrib['offset']))


def build_plan(net: Net, controllers, prog_name='default'):
    plan = {}
    phases_index = {}

    for tls in net.getTrafficLights():
        node_id = tls.getID()
        ctrl_data = controllers.get(node_id)

        prog = tls.getPrograms()[prog_name]
        shift = _get_program_offset(prog.toXML(node_id))

        phases_tmp = defaultdict(list)
        for phase in prog.getPhases():
            phase_number = int(re.findall(r'^\d+', phase.name)[0])
            phases_tmp[phase_number].append(phase)

        phases = []
        phase_idx = 0
        for phase_num_idx in range(len(phases_tmp)):
            phase_id = ctrl_data['phases'][phase_num_idx]['phase_id']
            sumo_phases = []
            phase_time = 0
            int_tact = 0
            for phase in phases_tmp.get(phase_num_idx + 1):
                # вычисляем параметры цикла
                phase_time += phase.duration if phase.duration is not None else 0
                # is_main_tact = bool(main_tact_regex.match(phase.state))
                is_main_tact = bool(main_tact_regex.match(phase.name))
                if not is_main_tact:
                    int_tact += phase.duration if phase.duration is not None else 0
                # строим соответстви phase_id -> [phase_idx]
                sumo_phases.append({
                    'sumo_phase_idx': phase_idx,
                    'is_main': is_main_tact,
                    'duration': phase.duration if phase.duration is not None else 0,
                })
                phase_idx += 1

            phases.append({
                'phase_id': phase_id,
                'int_tact': int_tact,
                'phase_time': phase_time,
            })
            phases_index[phase_id] = {
                'phase_id': phase_id,
                'int_tact': int_tact,
                'tacts': sumo_phases,
            }

        plan[node_id] = {
            'controller_id': ctrl_data['controller_id'],
            'shift': shift,
            'phases': phases,
        }
    return plan, phases_index

