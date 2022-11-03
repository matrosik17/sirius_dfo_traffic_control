from typing import List
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel

import traci
import traci.constants as tc

from models.common import Plan, ProgramState

from lib.graph_converter import GraphConverter


class SumoEnv:
    """
    Класс, отвечающий за получение усредненных по времени данныз с детекторов на графе
    """

    def __init__(self, traci, config_path=None, time_step=0.2, time_window=300, simulation_time=3600) -> None:
        self.config_path = config_path
        self.dt = time_step
        self.window = int(np.ceil(time_window / self.dt))
        self.time_window = self.dt * self.window
        self.simulation_time = simulation_time

        self.curr_idx = 0

        # lanes
        self._lane_speed = {}
        self._lane_occupancy = {}
        self._lane_curr_cars = {}
        self._lane_input_cars_count = {}
        self._lane_output_cars_count = {}

        for lane_id in traci.lane.getIDList():
            traci.lane.subscribe(lane_id, [
                tc.LAST_STEP_MEAN_SPEED,
                tc.LAST_STEP_OCCUPANCY,
                tc.LAST_STEP_VEHICLE_ID_LIST
            ])

            self._lane_speed[lane_id] = np.zeros(self.window)
            self._lane_occupancy[lane_id] = np.zeros(self.window)
            self._lane_curr_cars[lane_id] = set()
            self._lane_input_cars_count[lane_id] = np.zeros(self.window)
            self._lane_output_cars_count[lane_id] = np.zeros(self.window)

        # detectors
        self._loop_counts = {}
        self._loop_speed = {}
        self._loop_occupancy = {}
        self._loop_curr_cars = {}
        self._loop_total_cars_output = {}

        for loop_id in traci.inductionloop.getIDList():
            traci.inductionloop.subscribe(loop_id, [
                tc.LAST_STEP_MEAN_SPEED,
                tc.LAST_STEP_OCCUPANCY,
                tc.LAST_STEP_VEHICLE_ID_LIST
            ])

            self._loop_counts[loop_id] = np.zeros(self.window)
            self._loop_speed[loop_id] = np.zeros(self.window)
            self._loop_occupancy[loop_id] = np.zeros(self.window)
            self._loop_curr_cars[loop_id] = set()
            self._loop_total_cars_output[loop_id] = 0


    def update(self, traci):
        # обновляем данные по полосам
        lane_data = traci.lane.getAllSubscriptionResults()
        for lane_id, data in lane_data.items():
            self._lane_speed[lane_id][self.curr_idx] = max(data[tc.LAST_STEP_MEAN_SPEED], 0)
            self._lane_occupancy[lane_id][self.curr_idx] = max(data[tc.LAST_STEP_OCCUPANCY], 0)

            cars = set(data[tc.LAST_STEP_VEHICLE_ID_LIST])
            prev_cars = self._lane_curr_cars[lane_id]
            self._lane_curr_cars[lane_id] = cars
            self._lane_input_cars_count[lane_id][self.curr_idx] = len(cars - prev_cars)
            self._lane_output_cars_count[lane_id][self.curr_idx] = len(prev_cars - cars)

        # обновляем данные детекторов
        loops_data = traci.inductionloop.getAllSubscriptionResults()
        for loop_id, data in loops_data.items():
            self._loop_speed[loop_id][self.curr_idx] = max(data[tc.LAST_STEP_MEAN_SPEED], 0)
            self._loop_occupancy[loop_id][self.curr_idx] = max(data[tc.LAST_STEP_OCCUPANCY], 0)

            cars = set(data[tc.LAST_STEP_VEHICLE_ID_LIST])
            prev_cars = self._loop_curr_cars[loop_id]
            self._loop_curr_cars[loop_id] = cars
            self._loop_counts[loop_id][self.curr_idx] = len(cars - prev_cars)
            self._loop_total_cars_output[loop_id] += len(prev_cars - cars)


    def lane_cars_counter(self, lane_id) -> float:
        return len(self._lane_curr_cars[lane_id])


    def lane_cars_input(self, lane_id) -> float:
        return self._lane_input_cars_count[lane_id][self.curr_idx]


    def lane_cars_output(self, lane_id) -> float:
        return self._lane_output_cars_count[lane_id][self.curr_idx]


    def lane_intensity(self, lane_id) -> float:
        return np.sum(self._lane_input_cars_count[lane_id]) / self.time_window


    def lane_speed(self, lane_id) -> float:
        speed = self._lane_speed[lane_id]
        mask = speed > 0
        if len(speed[mask]) == 0:
            return 0.
        return np.mean(speed[mask])


    def lane_occupancy(self, lane_id) -> float:
        occ = self._lane_occupancy[lane_id]
        # mask = occ > 0
        # if len(occ[mask]) == 0:
        #     return 0.
        return np.mean(occ)


    def loop_cars_left(self, loop_id) -> float:
        return self._loop_counts[loop_id][self.curr_idx]


    def loop_intensity(self, loop_id) -> float:
        return np.sum(self._loop_counts[loop_id]) / self.time_window


    def loop_speed(self, loop_id) -> float:
        speed = self._loop_speed[loop_id]
        mask = speed > 0
        if len(speed[mask]) == 0:
            return 0.
        return np.mean(speed[mask])


    def loop_occupancy(self, loop_id) -> float:
        occ = self._loop_occupancy[loop_id]
        # mask = occ > 0
        # if len(occ[mask]) == 0:
        #     return 0.
        return np.mean(occ)


    def loop_total_cars_output(self, loop_id) -> float:
        return self._loop_total_cars_output[loop_id]


    def loop_update_idx(self):
        self.curr_idx = (self.curr_idx + 1) % self.window


    # TODO: переделать на subscription'ах
    def ctrl_phase_idx(self, ctrl_id) -> int:
        return traci.trafficlight.getPhase(ctrl_id)


class SumoTLControllers:
    """
    Управление светофорами через traci
    """

    def __init__(self, phases_index, controllers, start_time: float = 0.):
        self.phases_index = phases_index
        self.controllers = {ctrl['controller_id']: ctrl_id for ctrl_id, ctrl in controllers.items()}
        # для обновления планов
        self.start_time = start_time
        self.session_stop_time = start_time
        self.ctrl_tacts = {}
        self.ctrl_stop_time = {}
        # управление
        self.ctrl_next_switch = {}
        self.ctrl_idx = {}


    def set_init_tacts(self, init_phases: List[ProgramState]):
        for init_state in init_phases:
            ctrl_id = init_state.controller_id
            self.ctrl_idx[ctrl_id] = 0
            self.ctrl_next_switch[ctrl_id] = None
            self.ctrl_tacts[ctrl_id] = []
            self.ctrl_stop_time[ctrl_id] = self.start_time
            for phase in init_state.remain_phases:
                phase_id = phase.phase_id
                phase_tacts = self.phases_index[phase_id]
                main_tact_duration = phase.phase_time - phase_tacts['int_tact']
                for tact_tmp in phase_tacts['tacts']:
                    tact = tact_tmp.copy()
                    if tact['is_main']:
                        tact['duration'] = main_tact_duration
                    self.ctrl_tacts[ctrl_id].append(tact)
                    self.ctrl_stop_time[ctrl_id] += tact['duration']
                    if self.ctrl_next_switch[ctrl_id] is None:
                        self.ctrl_next_switch[ctrl_id] = self.start_time + tact['duration']


    def update_plan(self, control_period: int, plan: List[Plan]):
        # print([prog.json(by_alias=True) for prog in plan])
        self.session_stop_time += control_period
        for prog in plan:
            ctrl_id = prog.controller_id
            if ctrl_id not in self.ctrl_stop_time:
                self.ctrl_idx[ctrl_id] = 0
                self.ctrl_next_switch[ctrl_id] = None
                self.ctrl_tacts[ctrl_id] = []
                self.ctrl_stop_time[ctrl_id] = self.start_time

            while self.ctrl_stop_time[ctrl_id] < self.session_stop_time:
                for phase in prog.phases:
                    phase_id = phase.phase_id
                    phase_tacts = self.phases_index[phase_id]
                    main_tact_duration = phase.phase_time - phase_tacts['int_tact']
                    for tact_tmp in phase_tacts['tacts']:
                        tact = tact_tmp.copy()
                        if tact['is_main']:
                            tact['duration'] = main_tact_duration
                        self.ctrl_tacts[ctrl_id].append(tact)
                        self.ctrl_stop_time[ctrl_id] += tact['duration']
                        if self.ctrl_next_switch[ctrl_id] is None:
                            self.ctrl_next_switch[ctrl_id] = self.start_time + tact['duration']


    def update(self, curr_time: float, traci):
        for ctrl_id, switch_time in self.ctrl_next_switch.items():
            if curr_time > switch_time:
                # обновляем параметры
                self.ctrl_idx[ctrl_id] += 1
                tact = self.ctrl_tacts[ctrl_id][self.ctrl_idx[ctrl_id]]
                self.ctrl_next_switch[ctrl_id] += tact['duration']
                # отправляем управляющий сигнал в traci
                tl_id = self.controllers[ctrl_id]
                traci.trafficlight.setPhase(tl_id, tact['sumo_phase_idx'])
                traci.trafficlight.setPhaseDuration(tl_id, tact['duration'])
                # if tl_id == '247379907':
                #     print(f"Set phase: {tl_id}: {tact['sumo_phase_idx']} - {tact['duration']} | curr_time = {curr_time}")



class SimulationRecorder:
    """
    Класс, отвечающий за сбор статистики симуляции
    """

    def __init__(self, _traci, graph: GraphConverter, time_step=0.2):
        self.dt = time_step
        self.curr_time = 0.
        self.time = []
        self.lanes = {}
        self.loops = {}
        self.controllers = {}

        lanes = list(graph.chains.keys())
        loops = traci.inductionloop.getIDList()
        controllers = traci.trafficlight.getIDList()

        for lane_id in lanes:
            self.lanes[lane_id] = {
                'intensity': [],
                'speed': [],
                'occupancy': [],
                'cars_input': [],
                'cars_output': [],
                'total_cars': [],
            }

        for loop_id in loops:
            detector_position = traci.inductionloop.getPosition(loop_id)
            self.loops[loop_id] = {
                'position': detector_position,
                'intensity': [],
                'speed': [],
                'occupancy': [],
                'cars_left': [],
            }

        for ctrl_id in controllers:
            self.controllers[ctrl_id] = {
                'phase_idx': [],
            }


    def to_dict(self) -> dict:
        sumo_hist_dict = self.__dict__
        return sumo_hist_dict


    @classmethod
    def from_dict(cls, d: dict):
        sumo_hist =  cls.__new__(cls)
        for key, value in d.items():
            setattr(sumo_hist, key, value)
        return sumo_hist


    def update(self, env: SumoEnv):
        for lane_id, lane_data in self.lanes.items():
            lane_data['intensity'].append(env.lane_intensity(lane_id))
            lane_data['speed'].append(env.lane_speed(lane_id))
            lane_data['occupancy'].append(env.lane_occupancy(lane_id))
            lane_data['cars_input'].append(env.lane_cars_input(lane_id))
            lane_data['cars_output'].append(env.lane_cars_output(lane_id))
            lane_data['total_cars'].append(env.lane_cars_counter(lane_id))

        for loop_id, loop_data in self.loops.items():
            loop_data['intensity'].append(env.loop_intensity(loop_id))
            loop_data['speed'].append(env.loop_speed(loop_id))
            loop_data['occupancy'].append(env.loop_occupancy(loop_id))
            loop_data['cars_left'].append(env.loop_cars_left(loop_id))

        for ctrl_id, ctrl_data in self.controllers.items():
            ctrl_data['phase_idx'].append(env.ctrl_phase_idx(ctrl_id))

        self.time.append(self.curr_time)
        self.curr_time += self.dt


    def get_detector_position(self, loop_id):
        return self.loops[loop_id]['position']


    def plot_lane_intensity(self, lane_id, ax: plt.Axes, need_title=True, need_legend=True, fontsize=20, **kwargs):
        ax.plot(self.time, np.array(self.lanes[lane_id]['intensity']) * 3600, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Интенсивность на полосе {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Интенсивность, АТС/ч', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_speed(self, lane_id, ax: plt.Axes, need_title=True, need_legend=True, fontsize=20, **kwargs):
        ax.plot(self.time, np.array(self.lanes[lane_id]['speed']) * 3.6, **kwargs)
        ax.grid(True)


        if need_title:
            ax.set_title(f'Средняя скорость на полосе {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Скорость, км/ч', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_occupancy(self, lane_id, ax: plt.Axes, need_title=True, need_legend=True, fontsize=20, **kwargs):
        ax.plot(self.time, 100 * np.array(self.lanes[lane_id]['occupancy']), **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Загруженность на полосе {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Загруженность, %', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_data(self, lane_id, axes: List[plt.Axes], need_title=True, need_legend=True, fontsize=20, **kwargs):
        self.plot_lane_speed(lane_id, axes[0], need_title=need_title, need_legend=need_legend, fontsize=fontsize, **kwargs)
        self.plot_lane_occupancy(lane_id, axes[1], need_title=need_title, need_legend=need_legend, fontsize=fontsize, **kwargs)
        self.plot_lane_intensity(lane_id, axes[2], need_title=need_title, need_legend=need_legend, fontsize=fontsize, **kwargs)


    def plot_lane_total_cars_input(self, lane_id, ax: plt.Axes, need_title=True, need_legend=True, fontsize=20, **kwargs):
        ax.plot(self.time, np.cumsum(self.lanes[lane_id]['cars_input']), **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Суммарное число машин покинувших полосу {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Число машин', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_total_cars_output(self, lane_id, ax: plt.Axes, need_title=True, need_legend=True, fontsize=20, **kwargs):
        ax.plot(self.time, np.cumsum(self.lanes[lane_id]['cars_output']), **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Суммарное число машин покинувших полосу {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Число машин', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_lane_cars_counter(self, lane_id, ax: plt.Axes, need_title=True, need_legend=True, fontsize=20, **kwargs):
        ax.plot(self.time, np.array(self.lanes[lane_id]['total_cars']), **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Текущее число машин на полосе {lane_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Число машин', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_loop_intensity(self, loop_id, ax: plt.Axes, need_title=True, need_legend=True, fontsize=20, **kwargs):
        ax.plot(self.time, np.array(self.loops[loop_id]['intensity']) * 3600, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Интенсивность на детекторе {loop_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Интенсивность, АТС/ч', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_loop_speed(self, loop_id, ax: plt.Axes, need_title=True, need_legend=True, fontsize=20, **kwargs):
        ax.plot(self.time, np.array(self.loops[loop_id]['speed']) * 3.6, **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Средняя скорость на детекторе {loop_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Скорость, км/ч', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_loop_occupancy(self, loop_id, ax: plt.Axes, need_title=True, need_legend=True, fontsize=20, **kwargs):
        ax.plot(self.time, np.array(self.loops[loop_id]['occupancy']), **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Загруженность на детекторе {loop_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Загруженность, %', fontsize=fontsize)

        if need_legend:
            ax.legend()


    def plot_loop_data(self, loop_id, axes: List[plt.Axes], need_title=True, need_legend=True, fontsize=20, **kwargs):
        self.plot_loop_speed(loop_id, axes[0], need_title=need_title, need_legend=need_legend, fontsize=fontsize, **kwargs)
        self.plot_loop_occupancy(loop_id, axes[1], need_title=need_title, need_legend=need_legend, fontsize=fontsize, **kwargs)
        self.plot_loop_intensity(loop_id, axes[2], need_title=need_title, need_legend=need_legend, fontsize=fontsize, **kwargs)


    def plot_loop_total_cars_flow(self, loop_id, ax: plt.Axes, need_title=True, need_legend=True, fontsize=20, **kwargs):
        ax.plot(self.time, np.cumsum(self.loops[loop_id]['cars_left']), **kwargs)
        ax.grid(True)

        if need_title:
            ax.set_title(f'Суммарное число машин проехавших под детектором {loop_id}', fontsize=fontsize)
            ax.set_xlabel('Время, с', fontsize=fontsize)
            ax.set_ylabel('Число машин', fontsize=fontsize)

        if need_legend:
            ax.legend()


class SumoVehiclesStats(BaseModel):
    loaded: float
    inserted: float
    running: float
    waiting: float


class SumoVehicleTripStats(BaseModel):
   route_length: float
   speed: float
   duration: float
   waiting_time: float
   time_loss: float
   depart_delay: float
   depart_delay_waiting: float
   total_travel_time:float
   total_depart_delay: float


class SumoSimulationStats(BaseModel):
    vehicles: SumoVehiclesStats
    vehicle_trip_stats: SumoVehicleTripStats

    @classmethod
    def parse_xml(cls, stats_string: str) -> "SumoSimulationStats":
        stats_data = ET.fromstring(stats_string)
        vehicles_xml = stats_data.find("vehicles")
        vehicle_trip_stats_xml = stats_data.find("vehicleTripStatistics")

        vehicles = SumoVehiclesStats(
            loaded=float(vehicles_xml.get('loaded')),
            inserted=float(vehicles_xml.get('inserted')),
            running=float(vehicles_xml.get('running')),
            waiting=float(vehicles_xml.get('waiting'))
        )
        vehicle_trip_stats = SumoVehicleTripStats(
            route_length=float(vehicle_trip_stats_xml.get("routeLength")),
            speed=float(vehicle_trip_stats_xml.get("speed")),
            duration=float(vehicle_trip_stats_xml.get("duration")),
            waiting_time=float(vehicle_trip_stats_xml.get("waitingTime")),
            time_loss=float(vehicle_trip_stats_xml.get("timeLoss")),
            depart_delay=float(vehicle_trip_stats_xml.get("departDelay")),
            depart_delay_waiting=float(vehicle_trip_stats_xml.get("departDelayWaiting")),
            total_travel_time=float(vehicle_trip_stats_xml.get("totalTravelTime")),
            total_depart_delay=float(vehicle_trip_stats_xml.get("totalDepartDelay"))
        )

        return SumoSimulationStats(vehicles=vehicles, vehicle_trip_stats=vehicle_trip_stats)
