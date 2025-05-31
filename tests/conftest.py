import numpy as np
import pytest

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCType,
    SlackVarConstraintType,
)
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    AlphaRegType,
)
from direct_data_driven_mpc.utilities.controller.controller_creation import (
    LTIDataDrivenMPCParams,
    NonlinearDataDrivenMPCParams,
)

from .mocks import (
    MockLTIDDMPCController,
    MockLTIModel,
    MockNonlinearDDMPCController,
    MockNonlinearModel,
)


@pytest.fixture
def mock_lti_model() -> MockLTIModel:
    return MockLTIModel()


@pytest.fixture
def mock_nonlinear_model() -> MockNonlinearModel:
    return MockNonlinearModel()


@pytest.fixture
def mock_lti_controller() -> MockLTIDDMPCController:
    return MockLTIDDMPCController()


@pytest.fixture
def mock_nonlinear_controller() -> MockNonlinearDDMPCController:
    return MockNonlinearDDMPCController()


@pytest.fixture
def dummy_lti_controller_data() -> tuple[
    LTIDataDrivenMPCParams, np.ndarray, np.ndarray
]:
    n = 2
    m = 2
    p = 3
    L = 4

    # Compute minimum required N for persistent excitation
    N_min = m * (L + 2 * n) + L + 2 * n - 1
    u_range = np.array([[-1, 1]] * m)

    lti_dd_mpc_params: LTIDataDrivenMPCParams = {
        "n": n,
        "N": N_min,
        "L": L,
        "Q": np.eye(p * L),
        "R": np.eye(m * L),
        "eps_max": 0.01,
        "lamb_alpha": 0.01,
        "lamb_sigma": 0.01,
        "c": 1.0,
        "U": np.array([[-1.0, 1.0]] * m),
        "u_range": u_range,
        "slack_var_constraint_type": SlackVarConstraintType.CONVEX,
        "controller_type": LTIDataDrivenMPCType.NOMINAL,
        "n_mpc_step": 1,
        "u_s": np.zeros((m, 1)),
        "y_s": np.ones((p, 1)),
    }

    # Generate randomized initial input-output data
    np_random = np.random.default_rng(0)
    u_d = np_random.uniform(u_range[:, 0], u_range[:, 1], (N_min, m))
    y_d = np_random.uniform(-1.0, 1.0, (N_min, p))
    # Note:
    # For simplicity, dummy `u_range` defines the same input bounds for every
    # input. This is not always the case.

    return (lti_dd_mpc_params, u_d, y_d)


@pytest.fixture
def dummy_nonlinear_controller_data() -> tuple[
    NonlinearDataDrivenMPCParams, np.ndarray, np.ndarray
]:
    n = 2
    m = 2
    p = 3
    L = 4

    # Compute minimum required N for persistent excitation
    N = 50
    u_range = np.array([[-1, 1]] * m)

    nonlinear_dd_mpc_params: NonlinearDataDrivenMPCParams = {
        "n": n,
        "N": N,
        "L": L,
        "Q": np.eye(p * (L + n + 1)),
        "R": np.eye(m * (L + n + 1)),
        "S": np.eye(p),
        "lamb_alpha": 0.01,
        "lamb_sigma": 0.01,
        "U": np.array([[-1.0, 1.0]] * m),
        "Us": np.array([[-0.9, 0.9]] * m),
        "u_range": u_range,
        "alpha_reg_type": AlphaRegType.APPROXIMATED,
        "lamb_alpha_s": 0.01,
        "lamb_sigma_s": 0.01,
        "y_r": np.ones((p, 1)),
        "ext_out_incr_in": False,
        "update_cost_threshold": None,
        "n_mpc_step": 1,
    }

    # Generate randomized initial input-output data
    np_random = np.random.default_rng(0)
    u = np_random.uniform(u_range[:, 0], u_range[:, 1], (N, m))
    y = np_random.uniform(-1.0, 1.0, (N, p))
    # Note:
    # For simplicity, dummy `u_range` defines the same input bounds for every
    # input. This is not always the case.

    return (nonlinear_dd_mpc_params, u, y)
