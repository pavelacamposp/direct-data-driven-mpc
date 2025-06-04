import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCType,
    SlackVarConstraintType,
)
from direct_data_driven_mpc.utilities.controller.controller_creation import (
    create_lti_data_driven_mpc_controller,
)
from direct_data_driven_mpc.utilities.controller.controller_params import (
    get_lti_data_driven_mpc_controller_params,
)
from direct_data_driven_mpc.utilities.controller.data_driven_mpc_sim import (
    simulate_lti_data_driven_mpc_control_loop,
)
from direct_data_driven_mpc.utilities.controller.initial_data_generation import (  # noqa: E501
    generate_initial_input_output_data,
    randomize_initial_system_state,
)
from direct_data_driven_mpc.utilities.data_visualization import (
    plot_input_output,
    plot_input_output_animation,
    save_animation,
)
from direct_data_driven_mpc.utilities.models.lti_model import LTISystemModel

matplotlib.use("Agg")  # Prevent GUI backend

# Define test configuration file paths
TEST_LTI_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "config/models/test_lti_model.yaml"
)
TEST_MODEL_KEY = "test_lti_model"

TEST_LTI_DD_MPC_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "config/controllers/test_lti_dd_mpc_params.yaml",
)
TEST_LTI_DD_MPC_PARAMS_KEY = "test_lti_dd_mpc_params"


@pytest.mark.lti_integration
@pytest.mark.parametrize("n_n_mpc_step", [True, False])
@pytest.mark.parametrize(
    "controller_type, slack_var_constraint_type",
    [
        (LTIDataDrivenMPCType.NOMINAL, SlackVarConstraintType.NONE),
        (LTIDataDrivenMPCType.ROBUST, SlackVarConstraintType.NONE),
        (LTIDataDrivenMPCType.ROBUST, SlackVarConstraintType.CONVEX),
    ],
)
@pytest.mark.parametrize("bound_input", [True, False])
def test_lti_dd_mpc_integration(
    bound_input: bool,
    controller_type: LTIDataDrivenMPCType,
    slack_var_constraint_type: SlackVarConstraintType,
    n_n_mpc_step: bool,
    tmp_path: Path,
) -> None:
    """
    Integration test for LTI data-driven MPC controllers across multiple
    configurations.
    """
    # Define test parameters
    np_random = np.random.default_rng(0)
    n_steps = 50
    verbose = 0

    # Define system model
    system_model = LTISystemModel(
        config_file=TEST_LTI_MODEL_PATH,
        model_key=TEST_MODEL_KEY,
        verbose=verbose,
    )

    # Load Data-Driven MPC controller parameters from configuration file
    m = system_model.m  # Number of inputs
    p = system_model.p  # Number of outputs
    dd_mpc_config = get_lti_data_driven_mpc_controller_params(
        config_file=TEST_LTI_DD_MPC_CONFIG_PATH,
        controller_key=TEST_LTI_DD_MPC_PARAMS_KEY,
        m=m,
        p=p,
        verbose=verbose,
    )

    # Override controller parameters based on parameterized params
    if bound_input:
        dd_mpc_config["U"] = np.array([[-5, 5]] * m)
    else:
        dd_mpc_config["U"] = None

    dd_mpc_config["controller_type"] = controller_type
    dd_mpc_config["slack_var_constraint_type"] = slack_var_constraint_type
    dd_mpc_config["n_mpc_step"] = dd_mpc_config["n"] if n_n_mpc_step else 1

    # Randomize the initial state of the system
    x_0 = randomize_initial_system_state(
        system_model=system_model,
        controller_config=dd_mpc_config,
        np_random=np_random,
    )

    # Set system state to the randomized initial state
    system_model.set_state(state=x_0)

    # Generate initial input-output data
    u_d, y_d = generate_initial_input_output_data(
        system_model=system_model,
        controller_config=dd_mpc_config,
        np_random=np_random,
    )

    # Create LTI data-driven MPC controller
    dd_mpc_controller = create_lti_data_driven_mpc_controller(
        controller_config=dd_mpc_config, u_d=u_d, y_d=y_d
    )

    # Verify controller MPC solution on initialization
    assert dd_mpc_controller.get_problem_solve_status() == "optimal"
    assert dd_mpc_controller.get_optimal_cost_value() is not None

    # Simulate data-driven MPC control system
    u_sys, y_sys = simulate_lti_data_driven_mpc_control_loop(
        system_model=system_model,
        data_driven_mpc_controller=dd_mpc_controller,
        n_steps=n_steps,
        np_random=np_random,
        verbose=verbose,
    )

    # Verify input constraint satisfaction if enabled
    if bound_input:
        U = dd_mpc_config["U"]

        assert U is not None
        assert np.all(u_sys >= U[:, 0])
        assert np.all(u_sys <= U[:, 1])

    # Verify system reached stabilization
    # (input and outputs are close to their setpoints)
    u_s = dd_mpc_config["u_s"].flatten()
    y_s = dd_mpc_config["y_s"].flatten()

    np.testing.assert_allclose(u_sys[-1], u_s, rtol=2e-1)
    np.testing.assert_allclose(y_sys[-1], y_s, rtol=1e-1)

    # Test control data plotting
    u_s = dd_mpc_config["u_s"]
    y_s = dd_mpc_config["y_s"]
    U = dd_mpc_config["U"]
    u_bounds_list = U.tolist() if U is not None else None

    plot_input_output(
        u_k=u_sys,
        y_k=y_sys,
        u_s=u_s,
        y_s=y_s,
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
        u_s=u_s,
        y_s=y_s,
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
