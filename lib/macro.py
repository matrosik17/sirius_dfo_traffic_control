import os
import sys
from collections import defaultdict
from typing import DefaultDict, Iterable, List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

import traffic_model

from models.common import Plan, ProgramState, Phase, TrafficFlowConstant, TrafficFlowRecorded
from models.simulate import NetworkPressure, SystemSnapshot, SimulateRequestBody, SimulateResponseBody
from models.optimize import ProgramBound, OptimizeRequestBody, OptimizeResponseBody

from lib.sumo_env import SumoEnv, SimulationRecorder
from lib.graph_converter import GraphConverter


def estimate_input_flows(env: SumoEnv, input_chains: Dict[str, int], rho_max=0.2) -> List[TrafficFlowConstant]:
    traffic_flows = []
    for lane_id, chain_id in input_chains.items():
        intensity = env.loop_intensity(f'{lane_id}_start_loop')
        density = rho_max * env.loop_occupancy(f'{lane_id}_start_loop') / 100

        traffic_flows.append(TrafficFlowConstant.parse_obj({
            'chain_id': chain_id,
            'density': density,
            'intensity': intensity,
        }))
    return traffic_flows


def predict_mean_flows(
    sumo_hist: SimulationRecorder,
    input_chains: Dict[str, int],
    time_window: float,
    control_period: float,
    dt: float = 0.2,
    curr_time: float = None
) -> List[TrafficFlowRecorded]:
    input_flows: List[TrafficFlowRecorded] = []
    time = np.array(sumo_hist.time)

    time_max = None
    if curr_time is None:
        time_max = time.max()
    else:
        time_max = curr_time

    mask = (time >= (time_max - time_window)) & (time < time_max)
    for lane_id, chain_id in input_chains.items():
        loop_id = f'{lane_id}_start_loop'
        loop_data = sumo_hist.loops.get(loop_id)

        intensity = np.array(loop_data['intensity'])
        density = np.array(loop_data['occupancy']) * 0.2 / 100

        masked_time = time[mask]
        masked_intensity = intensity[mask]
        masked_density = density[mask]

        clf = Ridge(alpha=0.5)
        y = np.column_stack((masked_intensity, masked_density))
        clf.fit(masked_time.reshape((-1, 1)), y)

        pred_time: np.ndarray = np.arange(time_max, time_max + control_period, dt)
        y_pred = clf.predict(pred_time.reshape((-1, 1)))
        pred_intensity = y_pred[:, 0]
        pred_density = y_pred[:, 1]
        pred_time -= time_max

        input_flows.append(TrafficFlowRecorded.parse_obj({
            'chain_id': chain_id,
            'time': list(pred_time),
            'intensity': list(pred_intensity),
            'density': list(pred_density),
        }))
    return input_flows


# методы перенесены из its.adaptive_control_process
def build_init_trailing_programs(plan: List[Plan]) -> List[ProgramState]:
    trailing_programs = []
    for program in plan:
        shift = program.shift
        if shift == 0:
              continue

        trailing_phases = [];
        phase_idx = len(program.phases) - 1
        while shift > 0:
            phase = program.phases[phase_idx]
            phase_time = min(phase.phase_time, shift)
            trailing_phases.append(Phase.parse_obj({
                'phase_id': phase.phase_id,
                'phase_time': phase_time,
            }));
            shift -= phase_time
            phase_idx -= 1

        trailing_programs.append(ProgramState.parse_obj({
            'controller_id': program.controller_id,
            'remain_phases': trailing_phases.reverse(),
        }))
    return trailing_programs


def build_trailing_programs(control_period: int, plan: List[Plan]) -> List[ProgramState]:
    trailing_programs = []
    for program in plan:
        cycle_time = 0
        for phase in program.phases:
            cycle_time += phase.phase_time
        # длительность последнего цикла, попавшая в control_period
        last_cycle_local_time = (control_period - program.shift) % cycle_time

        trailing_phases = [];
        if last_cycle_local_time > 0:
            local_time = 0
            for phase in program.phases:
                phase_time = phase.phase_time
                if local_time + phase_time < last_cycle_local_time:
                    pass
                elif local_time < last_cycle_local_time and local_time + phase_time > last_cycle_local_time:
                    phase_local_time = last_cycle_local_time - local_time
                    trailing_phases.append(Phase.parse_obj({
                        'phase_id': phase.phase_id,
                        'phase_time': abs(phase_time - phase_local_time),
                    }))
                else:
                    trailing_phases.append(phase)
                local_time += phase_time

        trailing_programs.append(ProgramState.parse_obj({
            'controller_id': program.controller_id,
            'remain_phases': trailing_phases,
        }))
    return trailing_programs


def calc_shift(trailing_program: ProgramState) -> int:
    shift = 0
    for phase in trailing_program.remain_phases:
        shift += phase.phase_time
    return shift


def align_plan(trailing_programs: List[ProgramState], plan: List[Plan]) -> List[Plan]:
    init_state_map = {state.controller_id: state for state in trailing_programs}
    for prog in plan:
        ctrl_id = prog.controller_id
        shift = calc_shift(init_state_map.get(ctrl_id))
        prog.shift = shift
    return plan


def update_simulation_plan(sim_req: SimulateRequestBody, control_period: int) -> Tuple[List[ProgramState], List[Plan]]:
    plan = sim_req.plan.copy()
    init_states = build_trailing_programs(control_period, plan)
    plan = align_plan(init_states, plan)
    return init_states, plan


def align_bounds(trailing_programs: List[ProgramState], programs_bounds: List[ProgramBound]) -> List[ProgramBound]:
    init_state_map = {state.controller_id: state for state in trailing_programs}
    for prog_bound in programs_bounds:
        ctrl_id = prog_bound.controller_id
        shift = calc_shift(init_state_map.get(ctrl_id))
        prog_bound.shift = shift
    return programs_bounds


def align_input_flows(
    input_flows: List[TrafficFlowRecorded],
    start_time: float,
    control_period: float
) -> List[TrafficFlowRecorded]:
    start_idx = None
    stop_idx = None
    aligned_input_flows = []
    for input_flow in input_flows:
        # определяем границы временной области
        time = np.array(input_flow.time)
        if start_idx is None:
            start_idx = np.argmax(time >= start_time)
        if stop_idx is None:
            stop_idx = np.argmin(time < start_time + control_period)
            if stop_idx == 0:
                stop_idx = len(input_flow.time)
        # задаем входные потоки
        aligned_time = time[start_idx:stop_idx]
        aligned_time -= aligned_time[0]
        aligned_input_flows.append(TrafficFlowRecorded.parse_obj({
            'chain_id': input_flow.chain_id,
            'time': list(aligned_time),
            'intensity': input_flow.intensity[start_idx:stop_idx],
            'density': input_flow.density[start_idx:stop_idx],
        }))
    return aligned_input_flows


def update_optimization_request(sim_req: SimulateRequestBody, control_period: int) -> Tuple[List[ProgramState], List[Plan]]:
    plan = sim_req.plan.copy()
    init_states = build_trailing_programs(control_period, plan)
    init_state_map = {state.controller_id: state for state in init_states}

    for prog in plan:
        ctrl_id = prog.controller_id
        shift = calc_shift(init_state_map.get(ctrl_id))
        prog.shift = shift

    return init_states, plan


def run_simulation(simulation_request: SimulateRequestBody) -> SimulateResponseBody:
    try:
        simulation_response = traffic_model.simulate(simulation_request.json(by_alias=True))
        return SimulateResponseBody.parse_raw(simulation_response)
    except:
        print(simulation_request.json(by_alias=True, indent=4))
        raise


def run_optimization(optimization_request: OptimizeRequestBody) -> OptimizeResponseBody:
    try:
        optimization_response = traffic_model.optimize(optimization_request.json(by_alias=True))
        return OptimizeResponseBody.parse_raw(optimization_response)
    except:
        print(optimization_request.json(by_alias=True, indent=4))
        raise


class MacroMonitoring:
    """
    Класс для хранения и отображения мониторинга макросимуляции
    """
    FIELDS_WITH_INT_KEY = {
        'intensities', 'densities', 'velocities',
        'ctrls_phases',
        'lanes_length',
        'input_densities', 'input_intensities', 'input_velocities', 'input_omegas',
        'ctrl_phase_list', 'phase_pressure',
    }

    def __init__(self, graph_converter: GraphConverter):
        self.lanes_ids: Dict[str, int] = {lane_id: lane['chain_id'] for lane_id, lane in graph_converter.chains.items()}
        self.input_lanes: Dict[str, int] = graph_converter.input_chains
        self.output_lanes: Dict[str, int] = graph_converter.output_chains

        self.base_time: float = 0.
        self.times: List[float] = []
        self.lanes_length: Dict[int, float] = {}
        # данные по полосам
        self.intensities: DefaultDict[int, List[float]] = defaultdict(list)
        self.densities: DefaultDict[int, List[float]] = defaultdict(list)
        self.velocities: DefaultDict[int, List[float]] = defaultdict(list)
        # данные по контроллерам
        self.ctrls_phases: DefaultDict[int, List[int]] = defaultdict(list)
        # данные по входным потокам
        self.input_densities: DefaultDict[int, List[float]] = defaultdict(list)
        self.input_intensities: DefaultDict[int, List[float]] = defaultdict(list)
        self.input_velocities: DefaultDict[int, List[float]] = defaultdict(list)
        self.input_omegas: DefaultDict[int, List[float]] = defaultdict(list)
        # данные о давлении
        self.pressure_times: Optional[List[float]] = None
        self.ctrl_phase_list: Optional[DefaultDict[int, List[int]]] = None
        self.phase_pressure: Optional[DefaultDict[int, List[float]]] = None


    def to_dict(self) -> dict:
        macro_hist_dict = self.__dict__
        return macro_hist_dict


    @classmethod
    def from_dict(cls, d: dict):
        macro_hist =  cls.__new__(cls)
        for key, value in d.items():
            if key in cls.FIELDS_WITH_INT_KEY:
                value = {int(k): v for k,v in value.items()}
            setattr(macro_hist, key, value)
        return macro_hist


    def update(
        self,
        traffic_hist: List[SystemSnapshot],
        control_period: int,
        pressure: Optional[NetworkPressure] = None
    ):
        for system_snapshot in traffic_hist:
            prev_time = self.times[-1] if len(self.times) > 0 else -1.
            curr_time = self.base_time + system_snapshot.time
            if prev_time >= curr_time:
                continue
            self.times.append(curr_time)
            for lane in system_snapshot.lanes:
                chain_id = lane.chain_id
                if chain_id not in self.lanes_length:
                    lane_len = lane.dx * len(lane.density)
                    self.lanes_length[chain_id] = lane_len
                self.densities[chain_id].append(lane.density)
                self.intensities[chain_id].append(lane.flow)
                self.velocities[chain_id].append(lane.velocity)
            for ctrl in system_snapshot.controllers:
                ctrl_id = ctrl.controller_id
                self.ctrls_phases[ctrl_id].append(ctrl.phase_id)
            for input_flow in system_snapshot.input_flows:
                chain_id = input_flow.chain_id
                self.input_intensities[chain_id].append(input_flow.intensity)
                self.input_densities[chain_id].append(input_flow.density)
                self.input_velocities[chain_id].append(input_flow.velocity)
                self.input_omegas[chain_id].append(input_flow.omega)
        if pressure is not None:
            times = np.array(pressure.times) + self.base_time
            if self.ctrl_phase_list is None:
                self.pressure_times = list(times)
                self.ctrl_phase_list = {}
                self.phase_pressure = defaultdict(list)
                # готовим ctrl_phase_list
                for ctrl in pressure.controllers:
                    ctrl_id = ctrl.ctrl_id
                    phase_ids = []
                    for phase in ctrl.phases:
                        phase_ids.append(phase.phase_id)
                        self.phase_pressure[phase.phase_id].extend(phase.pressure)
                    phase_ids.sort()
                    self.ctrl_phase_list[ctrl_id] = phase_ids
            else:
                self.pressure_times += list(times)
                for ctrl in pressure.controllers:
                    for phase in ctrl.phases:
                        self.phase_pressure[phase.phase_id].extend(phase.pressure)
        self.base_time += control_period


    def _time_smoothing(self, time: np.ndarray, records: Iterable[float], time_window: float) -> np.ndarray:
        dt = np.diff(time).mean()
        window_size = int(np.ceil(time_window / dt))
        records_series = pd.Series(records)
        return records_series.rolling(window_size, min_periods=1).mean().to_numpy()


    def get_lane_mean_intensity(self, lane_id: int, time_window: Optional[float] =None):
        chain_id = self.lanes_ids[lane_id]
        intensity = np.array(self.intensities[chain_id]).mean(axis=1)
        if time_window is None:
            return intensity
        else:
            time = np.array(self.times)
            intensity = self._time_smoothing(time, intensity, time_window)
            return intensity


    def get_lane_mean_velocity(self, lane_id: int, time_window: Optional[float] = None):
        chain_id = self.lanes_ids[lane_id]
        velocity = np.array(self.velocities[chain_id]).mean(axis=1)
        if time_window is None:
            return velocity
        else:
            time = np.array(self.times)
            velocity = self._time_smoothing(time, velocity, time_window)
            return velocity


    def get_lane_mean_density(self, lane_id: int, time_window: Optional[float] = None):
        chain_id = self.lanes_ids[lane_id]
        density = np.array(self.densities[chain_id]).mean(axis=1)
        if time_window is None:
            return density
        else:
            time = np.array(self.times)
            density = self._time_smoothing(time, density, time_window)
            return density


    def plot_lane_mean_intensity(
        self,
        lane_id,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        time_window=None,
        **kwargs
    ):
        time = np.array(self.times)
        intensity = self.get_lane_mean_intensity(lane_id, time_window=time_window) * 3600
        ax.plot(time, intensity, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Средняя интенсивность на полосе {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Интенсивность, АТС/ч', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_mean_velocity(
        self,
        lane_id,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        time_window=None,
        **kwargs
    ):
        time = np.array(self.times)
        velocity = self.get_lane_mean_velocity(lane_id, time_window=time_window) * 3.6
        ax.plot(time, velocity, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Средняя скорость на полосе {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Скорость, км/ч', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_mean_occupancy(
        self,
        lane_id,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        time_window=None,
        rho_max=0.2,
        occ_max=1.0,
        fontsize=14,
        **kwargs
    ):
        time = np.array(self.times)
        occupancy = self.get_lane_mean_density(lane_id, time_window=time_window) * 100 * occ_max / rho_max
        ax.plot(time, occupancy, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Средняя загруженность на полосе {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Загруженность, %', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_mean_data(
        self,
        lane_id,
        axes: List[plt.Axes],
        need_title=False,
        need_legend=True,
        time_window=None,
        fontsize=14,
        **kwargs
    ):
        self.plot_lane_mean_velocity(lane_id, axes[0], need_title=need_title, need_legend=need_legend, time_window=time_window, fontsize=fontsize, **kwargs)
        self.plot_lane_mean_occupancy(lane_id, axes[1], need_title=need_title, need_legend=need_legend, time_window=time_window, fontsize=fontsize, **kwargs)
        self.plot_lane_mean_intensity(lane_id, axes[2], need_title=need_title, need_legend=need_legend, time_window=time_window, fontsize=fontsize, **kwargs)


    def get_lane_position_intensity(self, lane_id, position, time_window=None):
        chain_id = self.lanes_ids[lane_id]
        flow = np.array(self.intensities[chain_id])

        lane_length = self.lanes_length[chain_id]
        dx = lane_length / flow.shape[1]
        position_idx = int(np.floor(position / dx))

        intensity = flow[:, position_idx]
        if time_window is None:
            return intensity
        else:
            time = np.array(self.times)
            intensity = self._time_smoothing(time, intensity, time_window)
            return intensity


    def get_lane_position_velocity(self, lane_id, position, time_window=None):
        chain_id = self.lanes_ids[lane_id]
        speed = np.array(self.velocities[chain_id])

        lane_length = self.lanes_length[chain_id]
        dx = lane_length / speed.shape[1]
        position_idx = int(np.floor(position / dx))

        velocity = speed[:, position_idx]
        if time_window is None:
            return velocity
        else:
            time = np.array(self.times)
            velocity = self._time_smoothing(time, velocity, time_window)
            return velocity


    def get_lane_position_density(self, lane_id, position, time_window=None):
        chain_id = self.lanes_ids[lane_id]
        density = np.array(self.densities[chain_id])

        lane_length = self.lanes_length[chain_id]
        dx = lane_length / density.shape[1]
        position_idx = int(np.floor(position / dx))

        density = density[:, position_idx]
        if time_window is None:
            return density
        else:
            time = np.array(self.times)
            density = self._time_smoothing(time, density, time_window)
            return density


    def plot_lane_position_intensity(
        self,
        lane_id,
        position,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        time_window=None,
        **kwargs
    ):
        time = np.array(self.times)
        intensity = self.get_lane_position_intensity(lane_id, position, time_window) * 3600

        ax.plot(time, intensity, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Интенсивность на полосе {lane_id} на отметке {position} м', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Интенсивность, АТС/ч', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_position_velocity(
        self,
        lane_id,
        position,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        time_window=None,
        **kwargs
    ):
        time = np.array(self.times)
        velocity = self.get_lane_position_velocity(lane_id, position, time_window) * 3.6

        ax.plot(time, velocity, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Скорость на полосе {lane_id} на отметке {position} м', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Скорость, км/ч', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_position_occupancy(
        self,
        lane_id,
        position,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        time_window=None,
        rho_max=0.2,
        occ_max=1.0,
        **kwargs
    ):
        time = np.array(self.times)
        occupancy = self.get_lane_position_density(lane_id, position, time_window) * 100. * occ_max / rho_max

        ax.plot(time, occupancy, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Загруженность на полосе {lane_id} на отметке {position} м', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Загруженность, %', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_position_data(
        self,
        lane_id,
        position,
        axes: List[plt.Axes],
        fontsize=14,
        **kwargs
    ):
        self.plot_lane_position_velocity(lane_id, position, axes[0], fontsize=fontsize, **kwargs)
        self.plot_lane_position_occupancy(lane_id, position, axes[1], fontsize=fontsize, **kwargs)
        self.plot_lane_position_intensity(lane_id, position, axes[2], fontsize=fontsize, **kwargs)


    def _get_phases_working_intervals(self, controller_id) -> Dict[int, List[Tuple[float, float]]]:
        """
        Cтроим набор интервалов, в которых работают фазы указанного контроллера
        """
        phases = defaultdict(list)
        start_time = None
        curr_phase = None
        for curr_time, phase in zip(self.times, self.ctrls_phases[controller_id]):
            if phase != curr_phase:
                if start_time is not None and curr_phase is not None:
                    phases[phase].append([start_time, curr_time])
                start_time = curr_time
                curr_phase = phase
        phases[curr_phase].append([start_time, self.times[-1]])
        return phases


    def plot_phase_switches(self, controller_id, ax: plt.Axes, alpha=0.2, **kwargs):
        phases = self._get_phases_working_intervals(controller_id)
        # строим подходящий colormap для фаз
        cm = plt.get_cmap('gist_rainbow')
        num_phases = len(phases)
        phase_colors = {phase: cm(1. * phase_idx / num_phases) for phase_idx, phase in enumerate(phases.keys())}
        # добавляем фазы на график
        for phase, time_intervals in phases.items():
            color = phase_colors[phase]
            for [start_time, end_time] in time_intervals:
                ax.axvspan(start_time, end_time, color=color, alpha=alpha, **kwargs)


    def plot_input_intensity(
        self,
        lane_id,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        **kwargs
    ):
        chain_id = self.lanes_ids[lane_id]
        time = np.array(self.times)
        intensity = np.array(self.input_intensities[chain_id]) * 3600
        ax.plot(time, intensity, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Интенсивность входного потока на полосе {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Интенсивность, АТС/ч', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_input_velocity(
        self,
        lane_id,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        show_omega=False,
        **kwargs
    ):
        chain_id = self.lanes_ids[lane_id]
        time = np.array(self.times)

        velocity = np.array(self.input_velocities[chain_id], dtype=float)
        velocity = np.nan_to_num(velocity)
        velocity *= 3.6
        ax.plot(time, velocity, **kwargs)

        if show_omega:
            omega = np.array(self.input_omegas[chain_id], dtype=float)
            omega = np.nan_to_num(omega)
            omega *= 3.6
            ax.plot(time, omega, '--', **kwargs)
            ax.grid(True)

        if need_title:
            ax.set_title(f'Скорость входного потока на полосе {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Скорость, км/ч', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_input_occupancy(
        self,
        lane_id,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        rho_max=0.2,
        occ_max=1.0,
        **kwargs
    ):
        chain_id = self.lanes_ids[lane_id]
        time = np.array(self.times)
        occupancy = np.array(self.input_densities[chain_id]) * 100 * occ_max / rho_max
        ax.plot(time, occupancy, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Загруженность входного потока на полосе {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Загруженность, %', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_input_data(self, lane_id, axes: List[plt.Axes], **kwargs):
        self.plot_input_velocity(lane_id, axes[0], **kwargs)
        self.plot_input_occupancy(lane_id, axes[1], **kwargs)
        self.plot_input_intensity(lane_id, axes[2], **kwargs)


    def heatmap_lane_intensity(
        self,
        lane_id,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        **kwargs
    ):
        chain_id = self.lanes_ids[lane_id]
        lane_length = self.lanes_length[chain_id]

        time = np.array(self.times)
        intensity = np.array(self.intensities[chain_id]) * 3600
        coordinates = np.linspace(0., lane_length, intensity.shape[1])

        heatmap = ax.pcolormesh(coordinates, time, intensity, vmin=0, vmax=intensity.max(), shading='auto', **kwargs)
        fig = ax.get_figure()
        fig.colorbar(heatmap, ax=ax)

        if need_title:
            ax.set_title(f'Интенсивность на полосе {lane_id}', fontsize=fontsize)
            ax.set_ylabel('Время, с', fontsize=fontsize)
            ax.set_xlabel('Координаты, м', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def heatmap_lane_velocity(
        self,
        lane_id,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        **kwargs
    ):
        chain_id = self.lanes_ids[lane_id]
        lane_length = self.lanes_length[chain_id]

        time = np.array(self.times)
        velocity = np.array(self.velocities[chain_id]) * 3.6
        coordinates = np.linspace(0., lane_length, velocity.shape[1])

        heatmap = ax.pcolormesh(coordinates, time, velocity, vmin=0, vmax=velocity.max(), shading='auto', **kwargs)
        fig = ax.get_figure()
        fig.colorbar(heatmap, ax=ax)

        if need_title:
            ax.set_title(f'Скорость на полосе {lane_id}', fontsize=fontsize)
            ax.set_ylabel('Время, с', fontsize=fontsize)
            ax.set_xlabel('Координаты, м', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def heatmap_lane_occupancy(
        self,
        lane_id,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        rho_max=0.2,
        occ_max=1.0,
        **kwargs
    ):
        chain_id = self.lanes_ids[lane_id]
        lane_length = self.lanes_length[chain_id]

        time = np.array(self.times)
        occupancy = np.array(self.densities[chain_id]) * 100 * occ_max / rho_max
        coordinates = np.linspace(0., lane_length, occupancy.shape[1])

        heatmap = ax.pcolormesh(coordinates, time, occupancy, vmin=0, vmax=100, shading='auto', **kwargs)
        fig = ax.get_figure()
        fig.colorbar(heatmap, ax=ax)

        if need_title:
            ax.set_title(f'Загруженность на полосе {lane_id}', fontsize=fontsize)
            ax.set_ylabel('Время, с', fontsize=fontsize)
            ax.set_xlabel('Координаты, м', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_pressure(
        self,
        ctrl_id,
        ax: plt.Axes,
        need_title=False,
        need_legend=True,
        fontsize=14,
        **kwargs
    ):
        phases_ids = self.ctrl_phase_list.get(ctrl_id)
        if phases_ids is None:
            raise ValueError(f"There's no phases for ctrl {ctrl_id}")
        times = self.pressure_times

        for phase_id in phases_ids:
            label = f'phase = {phase_id}'
            ax.plot(times, self.phase_pressure[phase_id], label=label)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Ctrl_id: {ctrl_id}', fontsize=fontsize)
            ax.set_ylabel('Давление', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)

        if need_legend:
            ax.legend()
