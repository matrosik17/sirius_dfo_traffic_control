import os
import sys
import json
import pathlib
import timeit

import numpy as np
from typing import Any, Optional, List, Callable
from pydantic import BaseModel

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary
import sumolib

from lib.graph_converter import GraphConverter
from lib.sumo_env import SumoEnv, SumoSimulationStats, SumoTLControllers, SimulationRecorder
import lib.macro as macro
from lib.macro import MacroMonitoring
import lib.derivative_free_optimizaiton as dfo

from models.simulate import SimulateRequestBody, SimulateResponseBody
from models.optimize import OptimizeRequestBody, OptimizeResponseBody


# utils
def start_traci(configFile, gui=False, step=0.1):
    # Check SUMO has been set up properly
    sumo = "sumo-gui" if gui else "sumo"
    sumoBinary = checkBinary(sumo)

    # Start Simulation and step through
    traci.start([sumoBinary, "-c", configFile, "--step-length", str(step),
        "--collision.action", "none",
        "--start",
        # "--duration-log.statistics",
    ])


# Запуск симуляции

class SimulationConfig(BaseModel):
    name: Optional[str]
    # пути до конфигов симуляции
    sumo_config_path: str
    graph_path: str
    fundamental_diagram_path: str
    stats_path: Optional[str]
    # параметры симуляции
    simulation_step: float = 0.2
    simulation_duration: float = 3600
    time_window: float = 120
    graph_dx: float = 20


class SimulationResult(BaseModel):
    # пути до конфигов симуляции
    sumo_config_path: str
    graph_path: str
    fundamental_diagram_path: str
    # параметры симуляции
    simulation_step: float = 0.2
    simulation_duration: float = 3600
    time_window: float = 120
    graph_dx: float = 20
    # результаты симуляции
    graph: Any
    sumo_hist: Any
    macro_hist: Any
    stats: Optional[SumoSimulationStats]
    responses: Optional[List[SimulateResponseBody]]


def run_simulation(config: SimulationConfig, gui=False) -> SimulationResult:
    net = sumolib.net.readNet(config.graph_path, withPrograms=True)
    graph_converter = GraphConverter(net, fd_path=config.fundamental_diagram_path)

    # настройка + запуск симуляции
    start_traci(config.sumo_config_path, gui=gui, step=config.simulation_step)
    start_time = traci.simulation.getTime()

    env = SumoEnv(
        traci,
        config_path=config.sumo_config_path,
        time_step=config.simulation_step,
        simulation_time=config.simulation_duration,
        time_window=config.time_window
    )
    recorder = SimulationRecorder(traci, time_step=config.simulation_step, graph=graph_converter)
    try:
        while traci.simulation.getTime() < start_time + env.simulation_time:
            env.update(traci)
            recorder.update(env)
            traci.simulationStep()
            # костыль, чтобы корректно отдавать текущие показания детекторов
            env.loop_update_idx()
    finally:
        traci.close()

    try:
        if config.stats_path is None:
            raise ValueError("Can't find stats path")
        path = pathlib.Path(config.stats_path)
        with open(path, 'r') as file:
            stats_str = file.read()
        stats = SumoSimulationStats.parse_xml(stats_str)
    except Exception as e:
        print(f"Can't get simulation stats: {e}")
        stats = None

    # формируем входные потоки
    input_flows = []
    time = np.array(recorder.time)
    for lane_id, chain_id in graph_converter.input_chains.items():
        loop_id = f'{lane_id}_start_loop'
        loop_data = recorder.loops.get(loop_id)

        intensity = np.array(loop_data['intensity'])
        density = np.array(loop_data['occupancy']) * 0.2 / 100

        input_flows.append({
            'chain_id': chain_id,
            'time': list(time),
            'intensity': list(intensity),
            'density': list(density),
        })

    # начальные фазы
    plan = graph_converter.get_plan()
    init_progs_states = macro.build_init_trailing_programs(plan)

    simulation_request = SimulateRequestBody.parse_obj({
        "simulation_time": config.simulation_duration,
        "dx": config.graph_dx,
        "init_traffic_state": None,
        "traffic_flows": input_flows,
        "plan": plan,
        "programs_states":init_progs_states,
        "graph": graph_converter.get_graph(),
        "stats_parameters": {
            "traffic_state": {
                "is_binary": False,
            },
            "traffic_hist": {
                "period": 1,
            },
            "output_cars": None,
            "velocities": None,
            "cars": None,
            "velocities_partitioned": None,
            "pressure": {
                "period": 1,
            },
        },
    })

    # запуск macro-симуляции
    macro_hist = MacroMonitoring(graph_converter)
    sim_response = macro.run_simulation(simulation_request)
    macro_hist.update(sim_response.traffic_hist, config.simulation_duration, pressure=sim_response.pressure)

    # запись результата
    result = SimulationResult.parse_obj({
        # пути до конфигов симуляции
        'sumo_config_path': config.sumo_config_path,
        'graph_path': config.graph_path,
        'fundamental_diagram_path': config.fundamental_diagram_path,
         # параметры симуляции
        'simulation_step':config.simulation_step,
        'simulation_duration':config.simulation_duration,
        'time_window':config.time_window,
        'graph_dx':config.graph_dx,
        # результаты симуляции
        'graph': graph_converter.to_dict(),
        'sumo_hist': recorder.to_dict(),
        'macro_hist': macro_hist.to_dict(),
        'stats': stats and stats.dict(),
        'responses': None,
        # 'responses': [sim_response.dict(by_alias=True)],
    })

    return result


# Управление

class OptimizationConfig(BaseModel):
    name: Optional[str]
    # пути до конфигов симуляции
    sumo_config_path: str
    graph_path: str
    fundamental_diagram_path: str
    plan_constraints_path: str
    stats_path: Optional[str]
    # параметры симуляции
    simulation_step: float = 0.2
    simulation_duration: float = 3600
    time_window: float = 120
    graph_dx: float = 20
    # параметры оптимизации
    control_period: float = 120
    optimization_horizon: float = 600
    optimization_duration: float = 5



class OptimizationResult(BaseModel):
    # пути до конфигов симуляции
    sumo_config_path: str
    graph_path: str
    fundamental_diagram_path: str
    plan_constraints_path: str
    # параметры симуляции
    simulation_step: float = 0.2
    simulation_duration: float = 3600
    time_window: float = 120
    graph_dx: float = 20
    # параметры оптимизации
    control_period: float = 120
    optimization_horizon: float = 600
    optimization_duration: float = 5
    # результаты симуляции
    graph: Any
    sumo_hist: Any
    macro_hist: Any
    stats: Optional[SumoSimulationStats]
    optimization_requests: Optional[List[OptimizeRequestBody]]
    optimization_responses: Optional[List[OptimizeResponseBody]]


# Управление с фиксированным планом
# Отличия от симуляции: постоянные входные потоки, обновляемые каждые CONTROL_PERIOD секунд
def run_constant_control(config: OptimizationConfig, input_time_window: float = None, gui=False, plan = None) -> SimulationResult:
    # первичная настройка
    net = sumolib.net.readNet(config.graph_path, withPrograms=True)
    graph_converter = GraphConverter(net, fd_path=config.fundamental_diagram_path)

    # начальные входные потоки
    input_chains = graph_converter.input_chains
    init_traffic_flows = []
    intensity = 0 # [атс/c]
    speed = 0 # [м/c]
    density = 0.

    for chain_id in input_chains.values():
        traffic_flow = {
            'chain_id': chain_id,
            'density': density,
            'intensity': intensity,
        }
        init_traffic_flows.append(traffic_flow)

    # начальные фазы
    if plan is None:
        plan = graph_converter.get_plan()
    init_progs_states = macro.build_init_trailing_programs(plan)

    simulation_request = SimulateRequestBody.parse_obj({
        "simulation_time": config.control_period,
        "dx": config.graph_dx,
        "init_traffic_state": None,
        "traffic_flows": init_traffic_flows,
        "plan": plan,
        "programs_states":init_progs_states,
        "graph": graph_converter.get_graph(),
        "stats_parameters": {
            "traffic_state": {
                "is_binary": False,
            },
            "traffic_hist": {
                "period": 1,
            },
            "output_cars": None,
            "velocities": None,
            "cars": None,
            "velocities_partitioned": None,
            "pressure": {
                "period": 1,
            },
        },
    })

    # запуск симуляции
    start_traci(config.sumo_config_path, gui=gui, step=config.simulation_step)
    start_time = traci.simulation.getTime()

    env = SumoEnv(
        traci,
        config_path=config.sumo_config_path,
        time_step=config.simulation_step,
        simulation_time=config.simulation_duration,
        time_window=config.time_window
    )
    next_record_time = start_time + config.control_period
    macro_hist = MacroMonitoring(graph_converter)
    sumo_hist = SimulationRecorder(traci, time_step=config.simulation_step, graph=graph_converter)
    responses = []
    tl_controllers = SumoTLControllers(graph_converter.phases_index, graph_converter.controllers, start_time=start_time)
    tl_controllers.set_init_tacts(simulation_request.programs_states)
    tl_controllers.update_plan(config.control_period, simulation_request.plan)

    try:
        while traci.simulation.getTime() < start_time + env.simulation_time:
            curr_time = traci.simulation.getTime()
            env.update(traci)
            sumo_hist.update(env)
            traci.simulationStep()
            tl_controllers.update(curr_time, traci)
            if curr_time >= next_record_time:
                # запускаем симуляцию
                sim_response = macro.run_simulation(simulation_request)
                # записываем данные
                macro_hist.update(sim_response.traffic_hist, config.control_period, pressure=sim_response.pressure)
                responses.append(sim_response.dict(by_alias=True))
                # обновляем параметры запроса
                next_record_time += config.control_period
                init_progs_states, plan = macro.update_simulation_plan(simulation_request, config.control_period)
                simulation_request.init_traffic_state = sim_response.traffic_state
                simulation_request.programs_states = init_progs_states
                simulation_request.plan = plan
                if input_time_window is None:
                    simulation_request.traffic_flows = macro.estimate_input_flows(env, input_chains)
                else:
                    simulation_request.traffic_flows = macro.predict_mean_flows(sumo_hist, graph_converter.input_chains, input_time_window, config.control_period)
                # обновляем tl_controllers
                tl_controllers.update_plan(config.control_period, plan)
            # костыль, чтобы корректно отдавать текущие показания детекторов
            env.loop_update_idx()
    finally:
        traci.close()

    try:
        if config.stats_path is None:
            raise ValueError("Can't find stats path")
        path = pathlib.Path(config.stats_path)
        with open(path, 'r') as file:
            stats_str = file.read()
        stats = SumoSimulationStats.parse_xml(stats_str)
    except Exception as e:
        print(f"Can't get simulation stats: {e}")
        stats = None

    # запись результата
    result = SimulationResult.parse_obj({
        # пути до конфигов симуляции
        'sumo_config_path': config.sumo_config_path,
        'graph_path': config.graph_path,
        'fundamental_diagram_path': config.fundamental_diagram_path,
         # параметры симуляции
        'simulation_step':config.simulation_step,
        'simulation_duration':config.simulation_duration,
        'time_window':config.time_window,
        'graph_dx':config.graph_dx,
        # результаты симуляции
        'graph': graph_converter.to_dict(),
        'sumo_hist': sumo_hist.to_dict(),
        'macro_hist': macro_hist.to_dict(),
        'stats': stats and stats.dict(),
        'responses': responses,
    })

    return result


# Управление с адаптивным планом
def run_optimization(
    config: OptimizationConfig,
    input_time_window: float = None,
    optimizer: Callable[[OptimizeRequestBody], OptimizeResponseBody] = macro.run_optimization,
    gui=False
) -> OptimizationResult:
    # первичная настройка
    net = sumolib.net.readNet(config.graph_path, withPrograms=True)
    graph_converter = GraphConverter(net, fd_path=config.fundamental_diagram_path)
    with open(config.plan_constraints_path) as file:
        plan_constraints = json.load(file)
        for plan in plan_constraints:
            controller_id = plan["controller_id"]
            if isinstance(controller_id, str):
                controller_id = graph_converter.controllers[controller_id]["controller_id"]
                plan["controller_id"] = controller_id

    # начальные входные потоки
    input_chains = graph_converter.input_chains
    init_traffic_flows = []
    for chain_id in input_chains.values():
        traffic_flow = {
            'chain_id': chain_id,
            'density': 0.,
            'intensity': 0.,
        }
        init_traffic_flows.append(traffic_flow)

    # начальные фазы
    init_progs_states = macro.build_init_trailing_programs(graph_converter.get_plan())

    simulation_request = SimulateRequestBody.parse_obj({
        "simulation_time": config.control_period,
        "dx": config.graph_dx,
        "init_traffic_state": None,
        "traffic_flows": init_traffic_flows,
        "plan": graph_converter.get_plan(),
        "programs_states":init_progs_states,
        "graph": graph_converter.get_graph(),
        "stats_parameters": {
            "traffic_state": {
                "is_binary": False,
            },
            "traffic_hist": {
                "period": 1,
            },
            "output_cars": None,
            "velocities": None,
            "cars": None,
            "velocities_partitioned": None,
            "pressure": {
                "period": 1,
            },
        },
    })

    optimization_request = OptimizeRequestBody.parse_obj({
        "simulation_time": config.optimization_horizon,
        "dx":config.graph_dx,
        "optimization_duration": config.optimization_duration,
        "init_traffic_state": None,
        "traffic_flows":init_traffic_flows,
        "init_plan": graph_converter.get_plan(),
        "programs_states": [],
        "programs_bounds": plan_constraints,
        "graph": simulation_request.graph.copy()
    })

    # запуск управления
    start_traci(config.sumo_config_path, gui=gui, step=config.simulation_step)
    start_time = traci.simulation.getTime()

    env = SumoEnv(
        traci,
        config_path=config.sumo_config_path,
        time_step=config.simulation_step,
        simulation_time=config.simulation_duration,
        time_window=config.time_window
    )
    tl_controllers = SumoTLControllers(graph_converter.phases_index, graph_converter.controllers, start_time=start_time)
    tl_controllers.set_init_tacts(simulation_request.programs_states)
    tl_controllers.update_plan(config.control_period, simulation_request.plan)

    next_record_time = start_time + config.control_period
    macro_hist = MacroMonitoring(graph_converter)
    sumo_hist = SimulationRecorder(traci, time_step=config.simulation_step, graph=graph_converter)
    opt_requests = []
    opt_responses = []

    try:
        control_start = timeit.default_timer()
        print(f"Start control: {control_start}")
        while traci.simulation.getTime() < start_time + env.simulation_time:
            curr_time = traci.simulation.getTime()
            env.update(traci)
            sumo_hist.update(env)
            traci.simulationStep()
            tl_controllers.update(curr_time, traci)
            if curr_time >= next_record_time:
                # подбираем расписание
                print(f"RUN OPTIMIZER: curr_time = {curr_time}")
                opt_start = timeit.default_timer()
                opt_response = optimizer(optimization_request)
                opt_stop = timeit.default_timer()
                print(f"OPTIMIZATION IS DONE: duration = {opt_stop - opt_start}")
                opt_requests.append(optimization_request.dict(by_alias=True))
                opt_responses.append(opt_response.dict(by_alias=True))
                # прогнозируем распределение с этим расписанием
                simulation_request.plan = opt_response.plan
                sim_response = macro.run_simulation(simulation_request)
                # записываем результат
                macro_hist.update(sim_response.traffic_hist, config.control_period, pressure=sim_response.pressure)
                # обновляем параметры
                next_record_time += config.control_period
                init_progs_states = macro.build_trailing_programs(config.control_period, simulation_request.plan)
                prog_bounds = macro.align_bounds(init_progs_states, optimization_request.programs_bounds)
                plan = macro.align_plan(init_progs_states, simulation_request.plan)
                # plan = macro.align_plan(init_progs_states, optimization_request.plan)
                updated_flows = []
                if input_time_window is None:
                    updated_flows = macro.estimate_input_flows(env, input_chains)
                else:
                    updated_flows = macro.predict_mean_flows(sumo_hist, graph_converter.input_chains, input_time_window, config.control_period)
                # обновляем tl_controllers
                tl_controllers.update_plan(config.control_period, plan)
                # optimization_request
                optimization_request.init_traffic_state = sim_response.traffic_state.copy()
                init_plan = macro.align_plan(init_progs_states, optimization_request.plan)
                optimization_request.plan = plan
                optimization_request.programs_states = init_progs_states
                optimization_request.programs_bounds = prog_bounds
                optimization_request.traffic_flows = updated_flows
                # simulation_request
                simulation_request.init_traffic_state = sim_response.traffic_state.copy()
                simulation_request.programs_states = init_progs_states
                simulation_request.plan = plan
                simulation_request.traffic_flows = updated_flows
            # костыль, чтобы корректно отдавать текущие показания детекторов
            env.loop_update_idx()
    finally:
        traci.close()

    try:
        if config.stats_path is None:
            raise ValueError("Can't find stats path")
        path = pathlib.Path(config.stats_path)
        with open(path, 'r') as file:
            stats_str = file.read()
        stats = SumoSimulationStats.parse_xml(stats_str)
    except Exception as e:
        print(f"Can't get simulation stats: {e}")
        stats = None

    # запись результата
    result = OptimizationResult.parse_obj({
        # пути до конфигов симуляции
        'sumo_config_path': config.sumo_config_path,
        'graph_path': config.graph_path,
        'fundamental_diagram_path': config.fundamental_diagram_path,
        'plan_constraints_path': config.plan_constraints_path,
         # параметры симуляции
        'simulation_step':config.simulation_step,
        'simulation_duration':config.simulation_duration,
        'time_window':config.time_window,
        'graph_dx':config.graph_dx,
        # результаты симуляции
        'graph': graph_converter.to_dict(),
        'sumo_hist': sumo_hist.to_dict(),
        'macro_hist': macro_hist.to_dict(),
        'stats': stats and stats.dict(),
        'optimization_requests': opt_requests,
        'optimization_responses': opt_responses,
    })

    return result
