import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from direct_data_driven_mpc import (
    AlphaRegType,
)
from direct_data_driven_mpc.utilities.controller import (
    create_nonlinear_data_driven_mpc_controller,
    generate_initial_input_output_data,
    get_nonlinear_data_driven_mpc_controller_params,
    simulate_nonlinear_data_driven_mpc_control_loop,
)
from direct_data_driven_mpc.utilities.models import (
    NonlinearSystem,
)
from direct_data_driven_mpc.utilities.visualization import (
    plot_input_output,
    plot_input_output_animation,
    save_animation,
)

matplotlib.use("Agg")  # Prevent GUI backend

# Define test configuration file paths
TEST_NONLINEAR_DD_MPC_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "config/controllers/test_nonlinear_dd_mpc_params.yaml",
)
TEST_NONLINEAR_DD_MPC_PARAMS_KEY = "test_nonlinear_dd_mpc_params"


@pytest.mark.integration
@pytest.mark.nonlinear_integration
@pytest.mark.parametrize("n_n_mpc_step", [True, False])
@pytest.mark.parametrize("ext_out_incr_in", [True, False])
@pytest.mark.parametrize(
    "alpha_reg_type",
    [AlphaRegType.APPROXIMATED, AlphaRegType.PREVIOUS, AlphaRegType.ZERO],
)
def test_nonlinear_dd_mpc_integration(
    alpha_reg_type: AlphaRegType,
    ext_out_incr_in: bool,
    n_n_mpc_step: bool,
    test_nonlinear_system_model: NonlinearSystem,
    tmp_path: Path,
) -> None:
    """
    Integration test for nonlinear data-driven MPC controllers across multiple
    configurations.
    """
    # Define test parameters
    np_random = np.random.default_rng(0)
    n_steps = 50
    verbose = 0

    # Define system model
    system_model = test_nonlinear_system_model

    # Load Data-Driven MPC controller parameters from configuration file
    m = system_model.m  # Number of inputs
    p = system_model.p  # Number of outputs
    dd_mpc_config = get_nonlinear_data_driven_mpc_controller_params(
        config_file=TEST_NONLINEAR_DD_MPC_CONFIG_PATH,
        controller_key=TEST_NONLINEAR_DD_MPC_PARAMS_KEY,
        m=m,
        p=p,
        verbose=verbose,
    )

    # Override controller parameters based on parameterized params
    dd_mpc_config["alpha_reg_type"] = alpha_reg_type

    if ext_out_incr_in:
        dd_mpc_config["ext_out_incr_in"] = ext_out_incr_in
        L = dd_mpc_config["L"]
        n = dd_mpc_config["n"]
        dd_mpc_config["Q"] = np.eye((m + p) * (L + n + 1))

    dd_mpc_config["n_mpc_step"] = dd_mpc_config["n"] if n_n_mpc_step else 1

    # Adjust parameters for specific controller
    # configurations to ensure CVXPY solutions are feasible
    # Note:
    # Different combinations of controller parameters may require different
    # tuning to avoid solver errors. For example, a controller with
    # `ext_out_incr_in = True` will have different parameter values than one
    # with `ext_out_incr_in = False`.
    if (
        alpha_reg_type == AlphaRegType.APPROXIMATED
        and ext_out_incr_in
        and not n_n_mpc_step
    ):
        dd_mpc_config["lamb_alpha_s"] = 1e-4

    # Generate initial input-output data
    u, y = generate_initial_input_output_data(
        system_model=system_model,
        controller_config=dd_mpc_config,
        np_random=np_random,
    )

    # Create nonlinear data-driven MPC controller
    dd_mpc_controller = create_nonlinear_data_driven_mpc_controller(
        controller_config=dd_mpc_config, u=u, y=y
    )

    # Verify controller MPC solution on initialization
    assert dd_mpc_controller.get_problem_solve_status() == "optimal"
    assert dd_mpc_controller.get_optimal_cost_value() is not None

    # Simulate data-driven MPC control system
    u_sys, y_sys = simulate_nonlinear_data_driven_mpc_control_loop(
        system_model=system_model,
        data_driven_mpc_controller=dd_mpc_controller,
        n_steps=n_steps,
        np_random=np_random,
        verbose=verbose,
    )

    # Verify input constraint satisfaction
    U = dd_mpc_config["U"]

    assert np.all(u_sys >= U[:, 0])
    assert np.all(u_sys <= U[:, 1])

    # Verify system reached stabilization
    # (input and outputs are close to their setpoints)
    y_r = dd_mpc_config["y_r"].flatten()

    # Only assert convergence for alpha regularization types
    # APPROXIMATED and PREVIOUS, since ZERO tends to underperform
    if alpha_reg_type != AlphaRegType.ZERO:
        np.testing.assert_allclose(y_sys[-1], y_r, rtol=1e-1)

    # Change controller setpoint
    new_y_r = dd_mpc_config["y_r"] + 0.5

    dd_mpc_controller.set_output_setpoint(new_y_r)

    np.testing.assert_equal(dd_mpc_controller.y_r, new_y_r)

    # Simulate data-driven MPC control system for the new setpoint
    n_steps_setpoint_change = 75
    _, y_change = simulate_nonlinear_data_driven_mpc_control_loop(
        system_model=system_model,
        data_driven_mpc_controller=dd_mpc_controller,
        n_steps=n_steps_setpoint_change,
        np_random=np_random,
        verbose=verbose,
    )

    # Verify system reached stabilization for the new setpoint
    if alpha_reg_type != AlphaRegType.ZERO:
        np.testing.assert_allclose(y_change[-1], new_y_r.flatten(), rtol=2e-1)

    # Test control data plotting
    y_r_data = np.tile(dd_mpc_config["y_r"].T, (n_steps, 1))
    U = dd_mpc_config["U"]
    u_bounds_list = U.tolist() if U is not None else None

    plot_input_output(
        u_k=u_sys,
        y_k=y_sys,
        y_s=y_r_data,
        u_bounds_list=u_bounds_list,
        dpi=100,
    )

    # Plot and save control data animation
    N = dd_mpc_config["N"]
    anim_fps = 50
    anim_points_per_frame = 5

    anim = plot_input_output_animation(
        u_k=u_sys,
        y_k=y_sys,
        y_s=y_r_data,
        u_bounds_list=u_bounds_list,
        initial_steps=N,
        interval=1000.0 / anim_fps,
        points_per_frame=anim_points_per_frame,
        dpi=100,
    )

    # Save input-output animation as a GIF
    data_length = N + n_steps
    anim_frames = math.ceil((data_length - 1) / anim_points_per_frame) + 1
    anim_bitrate = 2000
    anim_path = os.path.join(tmp_path, "anim.gif")

    save_animation(
        animation=anim,
        total_frames=anim_frames,
        fps=anim_fps,
        bitrate=anim_bitrate,
        file_path=anim_path,
    )

    # Assert animation file exists (animation was created)
    assert os.path.isfile(anim_path)
