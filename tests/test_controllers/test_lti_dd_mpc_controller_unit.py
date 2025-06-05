from unittest.mock import Mock, patch

import numpy as np
import pytest

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCController,
    LTIDataDrivenMPCType,
    SlackVarConstraintType,
)
from direct_data_driven_mpc.utilities.controller.controller_creation import (
    LTIDataDrivenMPCParams,
)


@pytest.mark.parametrize("use_terminal_constraints", [True, False])
@pytest.mark.parametrize(
    "slack_var_constraint_type",
    [SlackVarConstraintType.NONE, SlackVarConstraintType.CONVEX],
)
@pytest.mark.parametrize(
    "controller_type",
    [LTIDataDrivenMPCType.NOMINAL, LTIDataDrivenMPCType.ROBUST],
)
@patch.object(LTIDataDrivenMPCController, "get_optimal_control_input")
def test_lti_dd_mpc_controller_init(
    mock_controller_get_optimal_input: Mock,
    controller_type: LTIDataDrivenMPCType,
    slack_var_constraint_type: SlackVarConstraintType,
    use_terminal_constraints: bool,
    dummy_lti_controller_data: tuple[
        LTIDataDrivenMPCParams, np.ndarray, np.ndarray
    ],
) -> None:
    # Define test parameters
    controller_params, u_d, y_d = dummy_lti_controller_data
    m = u_d.shape[1]
    p = y_d.shape[1]

    # Patch optimal control input retrieval to bypass solver status checks
    mock_controller_get_optimal_input.return_value = np.ones((1,))

    # Test expected LTI data-driven MPC controller initialization
    controller = LTIDataDrivenMPCController(
        n=controller_params["n"],
        m=m,
        p=p,
        u_d=u_d,
        y_d=y_d,
        L=controller_params["L"],
        Q=controller_params["Q"],
        R=controller_params["R"],
        u_s=controller_params["u_s"],
        y_s=controller_params["y_s"],
        eps_max=controller_params["eps_max"],
        lamb_alpha=controller_params["lamb_alpha"],
        lamb_sigma=controller_params["lamb_sigma"],
        U=controller_params["U"],
        c=controller_params["c"],
        slack_var_constraint_type=slack_var_constraint_type,
        controller_type=controller_type,
        n_mpc_step=controller_params["n_mpc_step"],
        use_terminal_constraints=use_terminal_constraints,
    )

    # Verify controller instantiation with the MPC solution cost value
    assert controller.get_optimal_cost_value() is not None


def test_lti_dd_mpc_controller_robust_non_convex(
    dummy_lti_controller_data: tuple[
        LTIDataDrivenMPCParams, np.ndarray, np.ndarray
    ],
) -> None:
    # Define test parameters
    controller_params, u_d, y_d = dummy_lti_controller_data
    m = u_d.shape[1]
    p = y_d.shape[1]

    # Verify that a `NotImplementedError` is raised for robust
    # controllers with a non-convex slack variable constraint
    with pytest.raises(NotImplementedError):
        LTIDataDrivenMPCController(
            n=controller_params["n"],
            m=m,
            p=p,
            u_d=u_d,
            y_d=y_d,
            L=controller_params["L"],
            Q=controller_params["Q"],
            R=controller_params["R"],
            u_s=controller_params["u_s"],
            y_s=controller_params["y_s"],
            eps_max=controller_params["eps_max"],
            lamb_alpha=controller_params["lamb_alpha"],
            lamb_sigma=controller_params["lamb_sigma"],
            U=controller_params["U"],
            c=controller_params["c"],
            slack_var_constraint_type=SlackVarConstraintType.NON_CONVEX,
            controller_type=LTIDataDrivenMPCType.ROBUST,
            n_mpc_step=controller_params["n_mpc_step"],
            use_terminal_constraints=True,
        )


@pytest.mark.parametrize(
    "case_value, expected_error_match",
    [
        # Case 1: Mismatched number of inputs
        ("invalid_m", "should match the number of inputs"),
        # Case 2: Insufficient input data length
        ("short_input", "The required minimum N"),
        # Case 3: Input data not persistently exciting
        ("non_pers_exc_input", "rank of its induced Hankel"),
        # Case 4a: Prediction horizon too short for NOMINAL controller
        ("short_horizon_nominal", "prediction horizon"),
        # Case 4b: Prediction horizon too short for ROBUST controller
        ("short_horizon_robust", "prediction horizon"),
        # Case 5a: Invalid Q matrix
        ("invalid_Q", "Output weighting square matrix Q"),
        # Case 5b: Invalid R matrix
        ("invalid_R", "Input weighting square matrix R"),
    ],
)
def test_lti_dd_mpc_controller_invalid_params(
    case_value: str,
    expected_error_match: str,
    dummy_lti_controller_data: tuple[
        LTIDataDrivenMPCParams, np.ndarray, np.ndarray
    ],
) -> None:
    # Define test parameters
    controller_params, u_d, y_d = dummy_lti_controller_data
    m = u_d.shape[1]
    p = y_d.shape[1]
    L = controller_params["L"]
    n = controller_params["n"]

    # Compute minimum required N for persistent excitation
    N_min = m * (L + 2 * n) + L + 2 * n - 1

    base_controller_kwargs = {
        "n": n,
        "m": m,
        "p": p,
        "u_d": u_d,
        "y_d": y_d,
        "L": L,
        "Q": controller_params["Q"],
        "R": controller_params["R"],
        "u_s": controller_params["u_s"],
        "y_s": controller_params["y_s"],
        "eps_max": controller_params["eps_max"],
        "lamb_alpha": controller_params["lamb_alpha"],
        "lamb_sigma": controller_params["lamb_sigma"],
        "U": controller_params["U"],
        "c": controller_params["c"],
        "slack_var_constraint_type": SlackVarConstraintType.CONVEX,
        "controller_type": LTIDataDrivenMPCType.ROBUST,
        "n_mpc_step": controller_params["n_mpc_step"],
        "use_terminal_constraints": True,
    }

    controller_kwargs = base_controller_kwargs.copy()

    # Override controller parameters based on the test case
    if case_value == "invalid_m":
        controller_kwargs["m"] = m + 1
    elif case_value == "short_input":
        controller_kwargs["u_d"] = np.random.uniform(-1.0, 1.0, (N_min - 1, m))
    elif case_value == "non_pers_exc_input":
        controller_kwargs["u_d"] = np.zeros((N_min, m))
    elif case_value == "short_horizon_nominal":
        controller_kwargs["controller_type"] = LTIDataDrivenMPCType.NOMINAL
        controller_kwargs["L"] = n - 1
    elif case_value == "short_horizon_robust":
        controller_kwargs["controller_type"] = LTIDataDrivenMPCType.ROBUST
        controller_kwargs["L"] = 2 * n - 1
    elif case_value == "invalid_Q":
        controller_kwargs["Q"] = np.eye(p * L - 1)
    elif case_value == "invalid_R":
        controller_kwargs["R"] = np.eye(m * L - 1)

    # Run test
    with pytest.raises(ValueError, match=expected_error_match):
        LTIDataDrivenMPCController(**controller_kwargs)


@pytest.mark.parametrize("valid_dimensions", [True, False])
def test_lti_store_input_output_measurement(
    valid_dimensions: bool,
    dummy_lti_controller: LTIDataDrivenMPCController,
) -> None:
    # Get dummy LTI data-driven MPC controller
    controller = dummy_lti_controller

    # Test input-output measurement storage
    if valid_dimensions:
        u_current = np.ones((controller.m, 1))
        y_current = np.ones((controller.p, 1))

        controller.store_input_output_measurement(u_current, y_current)

        assert np.allclose(controller.u_past[-controller.m :], u_current)
        assert np.allclose(controller.y_past[-controller.p :], y_current)
    else:
        u_current = np.ones((controller.m + 1, 1))
        y_current = np.ones((controller.p + 1, 1))

        with pytest.raises(ValueError, match="Incorrect dimensions"):
            controller.store_input_output_measurement(u_current, y_current)


@pytest.mark.parametrize("valid_dimensions", [True, False])
def test_lti_set_past_input_output_data(
    valid_dimensions: bool,
    dummy_lti_controller: LTIDataDrivenMPCController,
) -> None:
    # Get dummy LTI data-driven MPC controller
    controller = dummy_lti_controller

    # Test past input-output data setting
    if valid_dimensions:
        u_past = np.zeros((controller.n * controller.m, 1))
        y_past = np.ones((controller.n * controller.p, 1))

        controller.set_past_input_output_data(u_past, y_past)

        assert np.allclose(controller.u_past, u_past)
        assert np.allclose(controller.y_past, y_past)
    else:
        u_past = np.zeros((controller.n * controller.m + 1, 1))
        y_past = np.ones((controller.n * controller.p + 1, 1))

        with pytest.raises(ValueError, match="Incorrect dimensions"):
            controller.set_past_input_output_data(u_past, y_past)


@pytest.mark.parametrize("valid_dimensions", [True, False])
def test_lti_set_input_output_setpoints(
    valid_dimensions: bool,
    dummy_lti_controller: LTIDataDrivenMPCController,
) -> None:
    # Get dummy LTI data-driven MPC controller
    controller = dummy_lti_controller

    # Test input-output setpoint setting
    if valid_dimensions:
        u_s = np.ones_like(controller.u_s)
        y_s = np.ones_like(controller.y_s)

        controller.set_input_output_setpoints(u_s, y_s)

        assert np.allclose(controller.u_s, u_s)
        assert np.allclose(controller.y_s, y_s)

    else:
        u_s = np.ones((controller.u_s.shape[0] + 1, 1))
        y_s = np.ones((controller.y_s.shape[0] + 1, 1))

        with pytest.raises(ValueError, match="Incorrect dimensions"):
            controller.set_input_output_setpoints(u_s, y_s)
