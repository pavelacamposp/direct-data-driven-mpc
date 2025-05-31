from unittest.mock import Mock

import numpy as np
import pytest

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCController,
)
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    NonlinearDataDrivenMPCController,
)
from direct_data_driven_mpc.utilities.controller.data_driven_mpc_sim import (
    print_mpc_step_info,
    simulate_lti_data_driven_mpc_control_loop,
    simulate_nonlinear_data_driven_mpc_control_loop,
)
from direct_data_driven_mpc.utilities.models.lti_model import LTIModel
from direct_data_driven_mpc.utilities.models.nonlinear_model import (
    NonlinearSystem,
)


def test_sim_lti_dd_mpc_control_loop(
    mock_lti_model: LTIModel,
    mock_lti_controller: LTIDataDrivenMPCController,
) -> None:
    # Define test parameters
    m = mock_lti_model.m
    p = mock_lti_model.p
    n_steps = 10
    np_random = np.random.default_rng(0)

    # Test LTI data-driven MPC closed-loop simulation
    u, y = simulate_lti_data_driven_mpc_control_loop(
        mock_lti_model, mock_lti_controller, n_steps, np_random, verbose=0
    )

    # Verify correct shapes of output
    assert u.shape == (n_steps, m)
    assert y.shape == (n_steps, p)


def test_sim_nonlinear_dd_mpc_control_loop(
    mock_nonlinear_model: NonlinearSystem,
    mock_nonlinear_controller: NonlinearDataDrivenMPCController,
) -> None:
    # Define test parameters
    m = mock_nonlinear_model.m
    p = mock_nonlinear_model.p
    n_steps = 10
    np_random = np.random.default_rng(0)

    # Test LTI data-driven MPC closed-loop simulation
    u, y = simulate_nonlinear_data_driven_mpc_control_loop(
        mock_nonlinear_model,
        mock_nonlinear_controller,
        n_steps,
        np_random,
        verbose=0,
    )

    # Verify correct shapes of output
    assert u.shape == (n_steps, m)
    assert y.shape == (n_steps, p)


@pytest.mark.parametrize("verbose", [0, 1, 2])
@pytest.mark.parametrize("include_inputs", [True, False])
def test_print_mpc_step_info(
    verbose: int, include_inputs: bool, capsys: pytest.CaptureFixture
) -> None:
    # Define test parameters
    step = 5
    mpc_cost_val = 1.0
    y_s = np.array([1.0, 2.0])
    y_sys_k = np.array([0.5, 1.5])

    if include_inputs:
        u_s = np.array([0.0, 1.0])
        u_sys_k = np.array([-0.5, 0.5])
    else:
        u_s = None
        u_sys_k = None

    # Create a mock progress bar
    mock_progress_bar = Mock()

    # Test print function
    print_mpc_step_info(
        verbose=verbose,
        step=step,
        mpc_cost_val=mpc_cost_val,
        u_s=u_s,
        u_sys_k=u_sys_k,
        y_s=y_s,
        y_sys_k=y_sys_k,
        progress_bar=mock_progress_bar,
    )

    # Capture printed output
    out, _ = capsys.readouterr()

    # Verify behavior based on the verbosity level
    if verbose == 0:
        assert out == ""
        mock_progress_bar.set_description.assert_not_called()
    elif verbose == 1:
        mock_progress_bar.set_description.assert_called_once()
        mock_progress_bar.update.assert_called_once()
        assert out == ""
    elif verbose == 2:
        assert f"MPC cost value: {mpc_cost_val:>8.4f}" in out
        assert "y_1e" in out
        if include_inputs:
            assert "u_1e" in out
        else:
            assert "u_1e" not in out
