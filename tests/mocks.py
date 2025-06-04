"""Define mock classes and objects."""

import numpy as np


class MockLTIModel:
    def __init__(self) -> None:
        self.n = 3
        self.m = 2
        self.p = 3
        self.eps_max = 0.0
        self.x = np.zeros((self.n))

    def simulate_step(self, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.ones((self.p,))

    def simulate(self, U: np.ndarray, W: np.ndarray, steps: int) -> np.ndarray:
        return np.ones((steps, self.p))

    def get_initial_state_from_trajectory(
        self, U: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        return np.zeros((self.n,))

    def set_state(self, state: np.ndarray) -> None:
        self.x = state


class MockNonlinearModel:
    def __init__(self) -> None:
        self.n = 3
        self.m = 2
        self.p = 3
        self.eps_max = 0.0
        self.x = np.zeros((self.n))

    def simulate_step(self, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.ones((self.p,))

    def simulate(self, U: np.ndarray, W: np.ndarray, steps: int) -> np.ndarray:
        return np.ones((steps, self.p))


class MockLTIDDMPCController:
    def __init__(self) -> None:
        self.m = 2
        self.p = 3
        self.eps_max = 0.01
        self.u_s = np.zeros((self.m, 1))
        self.y_s = np.zeros((self.p, 1))
        self.n_mpc_step = 1

    def update_and_solve_data_driven_mpc(self) -> None:
        return

    def get_optimal_control_input_at_step(self, n_step: int) -> np.ndarray:
        return np.ones((self.m,))

    def store_input_output_measurement(
        self,
        u_current: np.ndarray,
        y_current: np.ndarray,
    ) -> None:
        return

    def get_optimal_cost_value(self) -> float:
        return 1.0


class MockNonlinearDDMPCController:
    def __init__(self) -> None:
        self.m = 2
        self.p = 3
        self.eps_max = 0.01
        self.y_r = np.zeros((self.p, 1))
        self.n_mpc_step = 1

    def update_and_solve_data_driven_mpc(self) -> None:
        return

    def get_optimal_control_input_at_step(self, n_step: int) -> np.ndarray:
        return np.ones((self.m,))

    def get_du_value_at_step(self, n_step: int) -> np.ndarray:
        return np.zeros((self.m,))

    def store_input_output_measurement(
        self,
        u_current: np.ndarray,
        y_current: np.ndarray,
        du_current: np.ndarray,
    ) -> None:
        return

    def get_optimal_cost_value(self) -> float:
        return 1.0
