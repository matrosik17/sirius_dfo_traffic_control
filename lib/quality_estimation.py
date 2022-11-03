import re

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error

from lib.sumo_env import SimulationRecorder
from lib.macro import MacroMonitoring


def interpolate_true_values(x_true, y_true, x_pred, y_pred):
    if x_true.max() < x_pred.max():
        mask_pred = x_pred <= x_true.max()
        x_pred = x_pred[mask_pred]
        y_pred = y_pred[mask_pred]
    interpolation = interp1d(x_true, y_true)
    return interpolation(x_pred), y_pred


def loops_error_metrics(
    sim_recorder: SimulationRecorder,
    macro_hist: MacroMonitoring,
    rho_max=0.2,
    time_window=None,
    intensity_max=0.6,
    velocity_max=15.27,
    density_max=0.13333
) -> pd.DataFrame:
    metrics = {
        'loops': [],
        'intensity_rmse': [],
        'velocity_rmse': [],
        'density_rmse': [],
        'intensity_medae': [],
        'velocity_medae': [],
        'density_medae': [],
        'intensity_mae': [],
        'velocity_mae': [],
        'density_mae': [],
        # relative error
        'intensity_re': [],
        'velocity_re': [],
        'density_re': [],
    }

    lane_time = np.array(sim_recorder.time)
    macro_time = np.array(macro_hist.times)
    for loop_id, loop in sim_recorder.loops.items():
        lane_id = re.findall('(.*)_(?:.*)_loop', loop_id)[0]
        position = loop['position']

        macro_intensity = macro_hist.get_lane_position_intensity(lane_id, position, time_window=time_window)
        macro_velocity = macro_hist.get_lane_position_velocity(lane_id, position, time_window=time_window)
        macro_density = macro_hist.get_lane_position_density(lane_id, position, time_window=time_window)

        loop_intensity = np.array(loop['intensity'])
        loop_velocity = np.array(loop['speed'])
        loop_density = np.array(loop['occupancy']) * rho_max / 100

        inter_loop_intensity, macro_intensity = interpolate_true_values(lane_time, loop_intensity, macro_time, macro_intensity)
        inter_loop_velocity, macro_velocity = interpolate_true_values(lane_time, loop_velocity, macro_time, macro_velocity)
        inter_loop_density, macro_density = interpolate_true_values(lane_time, loop_density, macro_time, macro_density)

        metrics['loops'].append(loop_id)
        metrics['intensity_rmse'].append(mean_squared_error(inter_loop_intensity, macro_intensity, squared=False))
        metrics['velocity_rmse'].append(mean_squared_error(inter_loop_velocity, macro_velocity, squared=False))
        metrics['density_rmse'].append(mean_squared_error(inter_loop_density, macro_density, squared=False))

        metrics['intensity_medae'].append(median_absolute_error(inter_loop_intensity, macro_intensity))
        metrics['velocity_medae'].append(median_absolute_error(inter_loop_velocity, macro_velocity))
        metrics['density_medae'].append(median_absolute_error(inter_loop_density, macro_density))

        metrics['intensity_mae'].append(mean_absolute_error(inter_loop_intensity, macro_intensity))
        metrics['velocity_mae'].append(mean_absolute_error(inter_loop_velocity, macro_velocity))
        metrics['density_mae'].append(mean_absolute_error(inter_loop_density, macro_density))

        metrics['intensity_re'].append(mean_absolute_error(inter_loop_intensity, macro_intensity) / intensity_max)
        metrics['velocity_re'].append(mean_absolute_error(inter_loop_velocity, macro_velocity) / velocity_max)
        metrics['density_re'].append(mean_absolute_error(inter_loop_density, macro_density) / density_max)

    return pd.DataFrame(metrics)


def lanes_error_metrics(
    sim_recorder: SimulationRecorder,
    macro_hist: MacroMonitoring,
    rho_max=0.2,
    time_window=None,
    intensity_max=0.6,
    velocity_max=15.27,
    density_max=0.13333
) -> pd.DataFrame:
    metrics = {
        'lanes': [],
        'intensity_rmse': [],
        'velocity_rmse': [],
        'density_rmse': [],
        'intensity_medae': [],
        'velocity_medae': [],
        'density_medae': [],
        # relative error
        'intensity_re': [],
        'velocity_re': [],
        'density_re': [],
    }

    lane_time = np.array(sim_recorder.time)
    macro_time = np.array(macro_hist.times)
    for lane_id, lane in sim_recorder.lanes.items():
        macro_intensity = macro_hist.get_lane_mean_intensity(lane_id, time_window=time_window)
        macro_velocity = macro_hist.get_lane_mean_velocity(lane_id, time_window=time_window)
        macro_density = macro_hist.get_lane_mean_density(lane_id, time_window=time_window)

        lane_intensity = np.array(lane['intensity'])
        lane_velocity = np.array(lane['speed'])
        lane_density = np.array(lane['occupancy']) * rho_max / 100

        inter_lane_intensity, macro_intensity = interpolate_true_values(lane_time, lane_intensity, macro_time, macro_intensity)
        inter_lane_velocity, macro_velocity = interpolate_true_values(lane_time, lane_velocity, macro_time, macro_velocity)
        inter_lane_density, macro_density = interpolate_true_values(lane_time, lane_density, macro_time, macro_density)

        metrics['lanes'].append(lane_id)
        metrics['intensity_rmse'].append(mean_squared_error(inter_lane_intensity, macro_intensity, squared=False))
        metrics['velocity_rmse'].append(mean_squared_error(inter_lane_velocity, macro_velocity, squared=False))
        metrics['density_rmse'].append(mean_squared_error(inter_lane_density, macro_density, squared=False))
        metrics['intensity_medae'].append(median_absolute_error(inter_lane_intensity, macro_intensity))
        metrics['velocity_medae'].append(median_absolute_error(inter_lane_velocity, macro_velocity))
        metrics['density_medae'].append(median_absolute_error(inter_lane_density, macro_density))
        metrics['intensity_re'].append(mean_absolute_error(inter_lane_intensity, macro_intensity) / intensity_max)
        metrics['velocity_re'].append(mean_absolute_error(inter_lane_velocity, macro_velocity) / velocity_max)
        metrics['density_re'].append(mean_absolute_error(inter_lane_density, macro_density) / density_max)

    return pd.DataFrame(metrics)