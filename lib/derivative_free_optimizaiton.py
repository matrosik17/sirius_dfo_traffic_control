"""
Код для проекта по безградиентной оптимизации

В работе используем только ограничения на длительности фаз (режим free_cycle)
"""

from typing import List, Optional
from enum import Enum
import timeit

import numpy as np
from numpy.typing import ArrayLike
import scipy.optimize as optimize
# from scipy.optimize import minimize

import torch
import torch.optim
import torch.nn as nn
from botorch.utils.sampling import sample_hypersphere

import lib.macro as macro

from models.common import Phase, Plan
from models.simulate import SimulateRequestBody, SimulateResponseBody, StatsParameters
from models.optimize import ModeEnum, OptimizeRequestBody, OptimizeResponseBody, ProgramBound, ProgramBoundPhase

def build_simulation_request(optimization_request: OptimizeRequestBody) -> SimulateRequestBody:
    return SimulateRequestBody(
        simulation_time=optimization_request.simulation_time,
        dx=optimization_request.dx,
        graph=optimization_request.graph,
        traffic_flows=optimization_request.traffic_flows,
        init_traffic_state=optimization_request.init_traffic_state,
        plan=optimization_request.plan,
        programs_states=optimization_request.programs_states,
        stats_parameters=StatsParameters.parse_obj({
            "traffic_state": {
                "is_binary": False,
            },
            "traffic_hist": None,
            "output_cars": {
                "period": 1,
            },
            "velocities": None,
            "cars": None,
            "velocities_partitioned": None,
            "pressure": None,
            "density": {
                "period": 1,
            },
            "average_intensity_score": {
                "period": 1,
            },
        }),
    )


def check_programs_bounds(prog_bounds: List[ProgramBound]):
    for prog_bound in prog_bounds:
        if prog_bound.mode != ModeEnum.free:
            raise RuntimeError(f"Incorrect program bound type ({prog_bound.mode}) for ctrl {prog_bound.controller_id}")


class TargetScore(Enum):
    OutputCars = "output_cars"
    CritDensityScore = "crit_density_score"
    AverageIntensityScore = "average_intensity_score"


def clip_phase_time(phase_time: float, phase_bound: ProgramBoundPhase):
    return min(phase_bound.phase_max, max(phase_time, phase_bound.phase_min))


class CostFunction:
    def __init__(self, optimization_request: OptimizeRequestBody, target: TargetScore = TargetScore.CritDensityScore):
        check_programs_bounds(optimization_request.programs_bounds)
        self.optimization_request = optimization_request
        self.simulation_request = build_simulation_request(optimization_request)
        self.target = target
        self.curr_idx = 0
        # лучший результат
        self.best_x = None
        self.best_target = None
        self.best_idx = None
        # история запусков
        self.args_hist = []
        self.oc_hist = [] # output cars
        self.cds_hist = [] # crit density score
        self.ais_hist = [] # average intensity score

    def params_to_plan(self, x: List[float]) -> List[Plan]:
        plan = []
        x_iter = iter(x)
        for prog_bound in self.optimization_request.programs_bounds:
            prog = Plan(
                controller_id=prog_bound.controller_id,
                shift=prog_bound.shift,
                phases=[Phase(phase_id=pb.phase_id, phase_time=clip_phase_time(next(x_iter), pb)) for pb in prog_bound.phases],
            )
            plan.append(prog)
        return plan

    def plan_to_params(self, plan: List[Plan]) -> List[float]:
        x = []
        for prog in plan:
            for phase in prog.phases:
                x.append(phase.phase_time)
        return x

    def sample_plan(self) -> List[Plan]:
        plan = []
        for prog_bound in self.optimization_request.programs_bounds:
            phase_min = np.array([pb.phase_min for pb in prog_bound.phases])
            phase_max = np.array([pb.phase_max for pb in prog_bound.phases])
            r = np.random.rand(len(phase_min))
            phase_times = phase_min + r * (phase_max - phase_min)

            prog = Plan(
                controller_id=prog_bound.controller_id,
                shift=prog_bound.shift,
                phases=[
                    Phase(
                        phase_id=pb.phase_id,
                        phase_time=phase_times[p_idx]
                    )
                    for p_idx, pb in enumerate(prog_bound.phases)]
            )
            plan.append(prog)
        return plan

    def get_target_score(self, simulate_response: SimulateResponseBody, target: TargetScore) -> float:
        if target == TargetScore.OutputCars:
            return -simulate_response.output_cars[-1].cars
        if target == TargetScore.CritDensityScore:
            return simulate_response.density.crit_density_score
        if target == TargetScore.AverageIntensityScore:
            return -simulate_response.average_intensity_score.average_intensity

    def drop_hist(self):
        self.best_x = None
        self.best_idx = None
        self.best_target = None
        self.args_hist.clear()
        self.oc_hist.clear()
        self.cds_hist.clear()
        self.ais_hist.clear()

    def __call__(self, x: List[float], target: Optional[TargetScore] = None) -> float:
        plan = self.params_to_plan(x)
        self.simulation_request.plan = plan
        simulate_response = macro.run_simulation(self.simulation_request)

        oc = self.get_target_score(simulate_response, target=TargetScore.OutputCars)
        cds = self.get_target_score(simulate_response, target=TargetScore.CritDensityScore)
        ais = self.get_target_score(simulate_response, target=TargetScore.AverageIntensityScore)

        self.args_hist.append(x)
        self.oc_hist.append(oc)
        self.cds_hist.append(cds)
        self.ais_hist.append(ais)

        if target is None:
            target = self.target

        target_score = self.get_target_score(simulate_response, target=target)
        if self.best_target is None or target_score < self.best_target:
            self.best_target = target_score
            self.best_x = x
            self.best_idx = self.curr_idx
        self.curr_idx += 1
        return target_score


# class AveragedCostFunction(CostFunction):
#     def __init__(self, optimization_request: OptimizeRequestBody, target: str = "output_cars", n_calls=10):
#         super().__init__(optimization_request, target)
#         self.n_calls = n_calls

#     def add_noise(self, x: List[float], eps: float = 1.) -> List[float]:
#         noise = eps * np.random.rand(len(x))
#         x_noised = np.array(x) + noise
#         return x_noised

#     def clip_bounds(self, x: List[float]) -> List[float]:
#         plan = self.params_to_plan(x)
#         for prog, prog_bound in zip(plan, self.optimization_request.programs_bounds):
#             for phase, phase_bound in zip(prog.phases, prog_bound.phases):
#                 phase.phase_time = min(phase_bound.phase_max, max(phase.phase_time, phase_bound.phase_min))
#         return self.plan_to_params(plan)

#     def __call__(self, x: List[float]) -> float:
#         ys = np.zeros(self.n_calls)
#         simulate_response = self.run(x)

#         if self.target == "output_cars":
#             ys[0] = -simulate_response.output_cars[-1].cars

#         for call_idx in range(1, self.n_calls):
#             x_noised = self.add_noise(x, eps=1.0)
#             x_noised = self.clip_bounds(x_noised)
#             simulate_response = self.run(x_noised)
#             if self.target == "output_cars":
#                 ys[call_idx] = -simulate_response.output_cars[-1].cars
#         return ys.mean()


class OptimizationTimeout(Exception):
    def __init__(self, xk: ArrayLike):
        self.xk = xk


class TimeoutCallback:
    def __init__(self, max_duration: float):
        self.start = timeit.default_timer()
        self.max_duration = max_duration

    def __call__(self, xk: ArrayLike) -> bool:
        curr_time = timeit.default_timer()
        if (curr_time - self.start) > self.max_duration:
            raise OptimizationTimeout(xk=xk)


def nelder_mead_optimization(
    optimization_request: OptimizeRequestBody,
    target=TargetScore.CritDensityScore
) -> OptimizeResponseBody:
    cost_func = CostFunction(optimization_request=optimization_request, target=target)

    bounds = []
    for prog_bound in optimization_request.programs_bounds:
        for phase in prog_bound.phases:
            bounds.append((phase.phase_min, phase.phase_max))

    init_plan = cost_func.optimization_request.plan
    x0 = cost_func.plan_to_params(init_plan)

    init_simplex = [x0]
    for i in range(len(x0)):
        plan = cost_func.sample_plan()
        xi = cost_func.plan_to_params(plan)
        init_simplex.append(xi)
    init_simplex = np.array(init_simplex)

    try:
        res = optimize.minimize(
            lambda x: cost_func(x),
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            # options={
            #     # 'adaptive': True,
            #     'xatol': 0.9,
            #     # 'initial_simplex': init_simplex,
            # },
            callback=TimeoutCallback(max_duration=optimization_request.optimization_duration)
        )
        # plan = cost_func.params_to_plan(res.x)
    except OptimizationTimeout as err:
        pass
        # plan = cost_func.params_to_plan(err.xk)

    plan = cost_func.params_to_plan(cost_func.best_x)
    optimize_response = OptimizeResponseBody(
        plan=plan,
        convergence=[0.],
        success=True,
        local_search_result=None,
    )
    return optimize_response


def powell_optimization(
    optimization_request: OptimizeRequestBody,
    target=TargetScore.AverageIntensityScore
) -> OptimizeResponseBody:
    cost_func = CostFunction(optimization_request=optimization_request, target=target)

    bounds = []
    for prog_bound in optimization_request.programs_bounds:
        for phase in prog_bound.phases:
            bounds.append((phase.phase_min, phase.phase_max))

    init_plan = cost_func.optimization_request.plan
    x0 = cost_func.plan_to_params(init_plan)

    try:
        res = optimize.minimize(
            lambda x: cost_func(x),
            x0,
            method="Powell",
            bounds=bounds,
            # options={
            #     # 'adaptive': True,
            #     'xatol': 0.9,
            #     # 'initial_simplex': init_simplex,
            # },
            # callback=TimeoutCallback(max_duration=optimization_request.optimization_duration)
        )
        plan = cost_func.params_to_plan(res.x)
    except OptimizationTimeout as err:
        plan = cost_func.params_to_plan(err.xk)

    optimize_response = OptimizeResponseBody(
        plan=plan,
        convergence=[0.],
        success=True,
        local_search_result=None,
    )
    return optimize_response


def direct_optimization(
    optimization_request: OptimizeRequestBody,
    target=TargetScore.CritDensityScore
) -> OptimizeResponseBody:
    cost_func = CostFunction(optimization_request=optimization_request, target=target)

    bounds = []
    for prog_bound in optimization_request.programs_bounds:
        for phase in prog_bound.phases:
            bounds.append((phase.phase_min, phase.phase_max))

    init_plan = cost_func.optimization_request.plan
    x0 = cost_func.plan_to_params(init_plan)

    try:
        res = optimize.direct(
            lambda x: cost_func(x),
            bounds=bounds,
            callback=TimeoutCallback(max_duration=optimization_request.optimization_duration)
        )
        # plan = cost_func.params_to_plan(res.x)
    except OptimizationTimeout as err:
        pass
        # plan = cost_func.params_to_plan(err.xk)

    plan = cost_func.params_to_plan(cost_func.best_x)
    optimize_response = OptimizeResponseBody(
        plan=plan,
        convergence=[0.],
        success=True,
        local_search_result=None,
    )
    return optimize_response


def stoch_grad(cost_func, x, tau, e, dim):
    x1 = x - tau * e
    x2 = x + tau * e
    f1 = cost_func(x1[0])
    f2 = cost_func(x2[0])
    return (dim / (2 * tau)) * (f2 - f1) * e


def smoothed_adam(
    optimization_request: OptimizeRequestBody,
    target=TargetScore.CritDensityScore
) -> OptimizeResponseBody:
    cost_func = CostFunction(optimization_request=optimization_request, target=target)

    x0 = cost_func.plan_to_params(cost_func.optimization_request.plan)
    x = torch.tensor(x0)
    x = nn.Parameter(x)

    x_min = []
    x_max = []
    for prog_bound in optimization_request.programs_bounds:
        for phase in prog_bound.phases:
            x_min.append(phase.phase_min)
            x_max.append(phase.phase_max)
    x_min = torch.tensor(x_min)
    x_max = torch.tensor(x_max)

    # n_steps = 250
    n_steps = 75
    # batch_size = 4
    batch_size = 4
    dim = x.shape[0]
    tau = 2.
    # tau = 2.
    lr = 5.

    # optimizer = torch.optim.SGD([x], lr=350.)
    optimizer = torch.optim.Adam([x], lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda l: 0.99)

    for i in range(n_steps):
        g = torch.zeros_like(x)
        for _ in range(batch_size):
            e = sample_hypersphere(d=dim)
            g_tmp = stoch_grad(cost_func, x, tau, e, dim)
            g += g_tmp[0]
        g /= batch_size
        x.grad = g
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()
        with torch.no_grad():
            x.clamp_(x_min, x_max)
    _last_target_score = cost_func(x)

    plan = cost_func.params_to_plan(cost_func.best_x)
    optimize_response = OptimizeResponseBody(
        plan=plan,
        convergence=[0.],
        success=True,
        local_search_result=None,
    )
    return optimize_response

