from unittest.mock import Mock, patch

import numpy as np
import pytest

from direct_data_driven_mpc import (
    AlphaRegType,
    NonlinearDataDrivenMPCController,
)
from direct_data_driven_mpc.utilities.controller import (
    NonlinearDataDrivenMPCParams,
)


@pytest.mark.parametrize("ext_out_incr_in", [True, False])
@pytest.mark.parametrize(
    "alpha_reg_type",
    [
        AlphaRegType.APPROXIMATED,
        AlphaRegType.PREVIOUS,
        AlphaRegType.ZERO,
    ],
)
@patch.object(NonlinearDataDrivenMPCController, "_get_optimal_control_input")
def test_nonlinear_dd_mpc_controller_init(
    mock_controller_get_optimal_input: Mock,
    alpha_reg_type: AlphaRegType,
    ext_out_incr_in: bool,
    dummy_nonlinear_controller_data: tuple[
        NonlinearDataDrivenMPCParams, np.ndarray, np.ndarray
    ],
) -> None:
    # Define test parameters
    controller_params, u, y = dummy_nonlinear_controller_data
    m = u.shape[1]
    p = y.shape[1]

    # Override MPC matrices for `ext_out_incr_in = True`
    if ext_out_incr_in:
        L = controller_params["L"]
        n = controller_params["n"]
        controller_params["Q"] = np.eye((m + p) * (L + n + 1))

    # Patch optimal control input retrieval to bypass solver status checks
    mock_controller_get_optimal_input.return_value = np.ones((1,))

    # Test expected LTI data-driven MPC controller initialization
    controller = NonlinearDataDrivenMPCController(
        n=controller_params["n"],
        m=m,
        p=p,
        u=u,
        y=y,
        L=controller_params["L"],
        Q=controller_params["Q"],
        R=controller_params["R"],
        S=controller_params["S"],
        y_r=controller_params["y_r"],
        lamb_alpha=controller_params["lamb_alpha"],
        lamb_sigma=controller_params["lamb_sigma"],
        U=controller_params["U"],
        Us=controller_params["Us"],
        alpha_reg_type=alpha_reg_type,
        lamb_alpha_s=controller_params["lamb_alpha_s"],
        lamb_sigma_s=controller_params["lamb_sigma_s"],
        ext_out_incr_in=ext_out_incr_in,
        update_cost_threshold=controller_params["update_cost_threshold"],
        n_mpc_step=controller_params["n_mpc_step"],
    )

    # Verify controller instantiation with the MPC solution cost value
    assert controller.get_optimal_cost_value() is not None


@pytest.mark.parametrize(
    "case_value, expected_error_match",
    [
        # Case 1: Mismatched number of inputs
        ("invalid_m", "should match the number of inputs"),
        # Case 2: Input data not persistently exciting
        ("non_pers_exc_input", "rank of its induced Hankel"),
        # Case 3: Prediction horizon too short
        ("short_horizon", "prediction horizon"),
        # Case 4a: Invalid Q matrix
        ("invalid_Q", "Output weighting square matrix Q"),
        # Case 4b: Invalid R matrix
        ("invalid_R", "Input weighting square matrix R"),
        # Case 4c: Invalid S matrix
        ("invalid_S", "Output setpoint weighting square matrix S"),
    ],
)
def test_nonlinear_dd_mpc_controller_invalid_params(
    case_value: str,
    expected_error_match: str,
    dummy_nonlinear_controller_data: tuple[
        NonlinearDataDrivenMPCParams, np.ndarray, np.ndarray
    ],
) -> None:
    # Define test parameters
    controller_params, u, y = dummy_nonlinear_controller_data
    m = u.shape[1]
    p = y.shape[1]
    N = controller_params["N"]
    L = controller_params["L"]
    n = controller_params["n"]

    base_controller_kwargs = {
        "n": n,
        "m": m,
        "p": p,
        "u": u,
        "y": y,
        "L": L,
        "Q": controller_params["Q"],
        "R": controller_params["R"],
        "S": controller_params["S"],
        "y_r": controller_params["y_r"],
        "lamb_alpha": controller_params["lamb_alpha"],
        "lamb_sigma": controller_params["lamb_sigma"],
        "U": controller_params["U"],
        "Us": controller_params["Us"],
        "alpha_reg_type": controller_params["alpha_reg_type"],
        "lamb_alpha_s": controller_params["lamb_alpha_s"],
        "lamb_sigma_s": controller_params["lamb_sigma_s"],
        "ext_out_incr_in": controller_params["ext_out_incr_in"],
        "update_cost_threshold": controller_params["update_cost_threshold"],
        "n_mpc_step": controller_params["n_mpc_step"],
    }

    controller_kwargs = base_controller_kwargs.copy()

    # Override controller parameters based on the test case
    if case_value == "invalid_m":
        controller_kwargs["m"] = m + 1
    elif case_value == "non_pers_exc_input":
        controller_kwargs["u"] = np.zeros((N, m))
    elif case_value == "short_horizon":
        controller_kwargs["L"] = n - 1
    elif case_value == "invalid_Q":
        controller_kwargs["Q"] = np.eye(p * (L + n + 1) - 1)
    elif case_value == "invalid_R":
        controller_kwargs["R"] = np.eye(m * (L + n + 1) - 1)
    elif case_value == "invalid_S":
        controller_kwargs["S"] = np.eye(p + 1)

    # Run test
    with pytest.raises(ValueError, match=expected_error_match):
        NonlinearDataDrivenMPCController(**controller_kwargs)


@pytest.mark.parametrize("valid_dimensions", [True, False])
def test_nonlinear_store_input_output_measurement(
    valid_dimensions: bool,
    dummy_nonlinear_controller: NonlinearDataDrivenMPCController,
) -> None:
    # Get dummy nonlinear data-driven MPC controller
    controller = dummy_nonlinear_controller

    # Test input-output measurement storage
    if valid_dimensions:
        u_current = np.ones((controller.m,))
        y_current = np.ones((controller.p,))

        controller.store_input_output_measurement(u_current, y_current)

        assert np.allclose(controller.u[-1:], u_current)
        assert np.allclose(controller.y[-1:], y_current)
    else:
        u_current = np.ones((controller.m + 1,))
        y_current = np.ones((controller.p + 1,))

        with pytest.raises(ValueError, match="Incorrect dimensions"):
            controller.store_input_output_measurement(u_current, y_current)


@pytest.mark.parametrize("valid_dimensions", [True, False])
def test_nonlinear_set_input_output_data(
    valid_dimensions: bool,
    dummy_nonlinear_controller: NonlinearDataDrivenMPCController,
) -> None:
    # Get dummy nonlinear data-driven MPC controller
    controller = dummy_nonlinear_controller

    # Test input-output data setting
    if valid_dimensions:
        u = np.zeros(controller.u.shape)
        y = np.ones(controller.y.shape)

        controller.set_input_output_data(u, y)

        assert np.allclose(controller.u, u)
        assert np.allclose(controller.y, y)
    else:
        u = np.zeros((controller.u.shape[0] + 1, controller.u.shape[1]))
        y = np.ones((controller.y.shape[0] + 1, controller.y.shape[1]))

        with pytest.raises(ValueError, match="Incorrect dimensions"):
            controller.set_input_output_data(u, y)


@pytest.mark.parametrize("valid_dimensions", [True, False])
def test_nonlinear_set_output_setpoint(
    valid_dimensions: bool,
    dummy_nonlinear_controller: NonlinearDataDrivenMPCController,
) -> None:
    # Get dummy LTI data-driven MPC controller
    controller = dummy_nonlinear_controller

    # Test input-output setpoint setting
    if valid_dimensions:
        y_r = np.ones_like(controller.y_r)

        controller.set_output_setpoint(y_r)

        assert np.allclose(controller.y_r, y_r)

    else:
        y_r = np.ones((controller.y_r.shape[0] + 1, 1))

        with pytest.raises(ValueError, match="Incorrect dimensions"):
            controller.set_output_setpoint(y_r)
