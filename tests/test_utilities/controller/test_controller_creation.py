from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCController,
)
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    NonlinearDataDrivenMPCController,
)
from direct_data_driven_mpc.utilities.controller.controller_creation import (
    create_lti_data_driven_mpc_controller,
    create_nonlinear_data_driven_mpc_controller,
)


@pytest.mark.parametrize(
    "u_shape, y_shape, expected_exception",
    [
        ((100, 2), (100, 2), None),  # Valid: Equal number of inputs-outputs
        ((100, 2), (90, 1), ValueError),  # Invalid: Mismatched sample count
        ((100, 1), (100, 2), None),  # Valid: Unequal number of inputs-outputs
    ],
)
def test_create_lti_data_driven_mpc_controller(
    u_shape: tuple[int, int],
    y_shape: tuple[int, int],
    expected_exception: type[Exception] | None,
    test_dd_mpc_controller_config: dict[str, Any],
) -> None:
    # Define test parameters
    np_random = np.random.default_rng(0)
    u_d = np_random.random(u_shape)
    y_d = np_random.random(y_shape)
    m = u_d.shape[1]  # Number of inputs
    p = y_d.shape[1]  # Number of outputs

    config: Any = test_dd_mpc_controller_config
    config["Q"] = np.eye(p * config["L"])
    config["R"] = np.eye(m * config["L"])
    config["u_s"] = np.zeros((m, 1))
    config["y_s"] = np.zeros((p, 1))
    config["U"] = np.array([[0, 1]] * m)

    # Check for exception or validate correct controller instantiation
    if expected_exception:
        with pytest.raises(expected_exception):
            create_lti_data_driven_mpc_controller(config, u_d, y_d)
    else:
        controller = create_lti_data_driven_mpc_controller(config, u_d, y_d)
        assert isinstance(controller, LTIDataDrivenMPCController)
        assert controller.m == u_shape[1]
        assert controller.p == y_shape[1]
        assert controller.get_optimal_cost_value() is not None


@pytest.mark.parametrize("ext_out_incr_in_status", [True, False])
@pytest.mark.parametrize(
    "u_shape, y_shape, expected_exception",
    [
        ((100, 2), (100, 2), None),  # Valid: Equal number of inputs-outputs
        ((100, 2), (90, 1), ValueError),  # Invalid: Mismatched sample count
        ((100, 1), (100, 2), None),  # Valid: Unequal number of inputs-outputs
    ],
)
@patch.object(NonlinearDataDrivenMPCController, "get_optimal_control_input")
@patch.object(NonlinearDataDrivenMPCController, "solve_alpha_sr_Lin_Dt")
def test_create_nonlinear_data_driven_mpc_controller(
    mock_controller_alpha_solve: Mock,
    mock_controller_get_optimal_input: Mock,
    u_shape: tuple[int, int],
    y_shape: tuple[int, int],
    expected_exception: type[Exception] | None,
    ext_out_incr_in_status: bool,
    test_dd_mpc_controller_config: dict[str, Any],
) -> None:
    # Define test parameters
    np_random = np.random.default_rng(0)
    u_d = np_random.random(u_shape)
    y_d = np_random.random(y_shape)
    m = u_d.shape[1]  # Number of inputs
    p = y_d.shape[1]  # Number of outputs

    config: Any = test_dd_mpc_controller_config
    matrix_order = config["L"] + config["n"] + 1

    if ext_out_incr_in_status:
        config["Q"] = np.eye((m + p) * (matrix_order))
    else:
        config["Q"] = np.eye(p * (matrix_order))

    config["R"] = np.eye(m * matrix_order)
    config["S"] = np.eye(p)
    config["U"] = np.array([[0, 1]] * m)
    config["Us"] = np.array([[0.1, 0.9]] * m)
    config["y_r"] = np.zeros((p, 1))
    config["ext_out_incr_in"] = ext_out_incr_in_status

    # Patch optimal control input retrieval to bypass solver status checks
    mock_controller_get_optimal_input.return_value = np.zeros((1,))

    # Patch `alpha` approximation to return a value with the correct shape
    N = u_shape[0]
    L = config["L"]
    n = config["n"]
    alpha_dim = (N - L - n, 1)
    mock_controller_alpha_solve.return_value = np.zeros(alpha_dim)

    # Check for exception or validate correct controller instantiation
    if expected_exception:
        with pytest.raises(expected_exception):
            create_nonlinear_data_driven_mpc_controller(config, u_d, y_d)
    else:
        controller = create_nonlinear_data_driven_mpc_controller(
            config, u_d, y_d
        )

        assert isinstance(controller, NonlinearDataDrivenMPCController)
        assert controller.m == u_shape[1]
        assert controller.p == y_shape[1]
        assert controller.get_optimal_cost_value() is not None
