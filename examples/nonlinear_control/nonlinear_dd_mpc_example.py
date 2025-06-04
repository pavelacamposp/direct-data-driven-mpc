"""
Nonlinear Data-Driven Model Predictive Control (MPC) Example Script

This script demonstrates the setup, simulation, and visualization of a Direct
Data-Driven MPC controller for Nonlinear systems, applied to a nonlinear
continuous stirred tank reactor (CSTR) based on the research of J. Berberich
et al. [2].

The implementation follows the parameters defined in the example presented in
Section V of [2], including those for the system model, the initial
input-output data generation, and the Data-Driven MPC controller setup.

References:
    [2] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Linear
        Tracking MPC for Nonlinear Systems—Part II: The Data-Driven Case," in
        IEEE Transactions on Automatic Control, vol. 67, no. 9, pp. 4406-4421,
        Sept. 2022, doi: 10.1109/TAC.2022.3166851.
"""

import argparse
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from nonlinear_cstr_model import create_nonlinear_cstr_system

from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    AlphaRegType,
)
from direct_data_driven_mpc.utilities.controller.controller_creation import (
    create_nonlinear_data_driven_mpc_controller,
)
from direct_data_driven_mpc.utilities.controller.controller_params import (
    get_nonlinear_data_driven_mpc_controller_params,
)
from direct_data_driven_mpc.utilities.controller.data_driven_mpc_sim import (
    simulate_nonlinear_data_driven_mpc_control_loop,
)
from direct_data_driven_mpc.utilities.controller.initial_data_generation import (  # noqa: E501
    generate_initial_input_output_data,
)
from direct_data_driven_mpc.utilities.data_visualization import (
    plot_input_output,
    plot_input_output_animation,
    save_animation,
)
from direct_data_driven_mpc.utilities.yaml_config_loading import (
    load_yaml_config_params,
)

# Directory paths
dirname = os.path.dirname
project_dir = dirname(dirname(dirname(__file__)))
examples_dir = os.path.join(project_dir, "examples")
models_config_dir = os.path.join(examples_dir, "config", "models")
controller_config_dir = os.path.join(examples_dir, "config", "controllers")
plot_params_config_dir = os.path.join(examples_dir, "config", "plots")
default_animation_dir = os.path.join(project_dir, "animation_outputs")

# Nonlinear Continuous Stirred Tank Reactor (CSTR) configuration file
cstr_model_config_file = "nonlinear_cstr_system_params.yaml"
cstr_model_config_path = os.path.join(
    models_config_dir, cstr_model_config_file
)
cstr_model_key = "cstr_system"

# Data-Driven MPC controller configuration file
default_controller_config_file = "nonlinear_dd_mpc_example_params.yaml"
default_controller_config_path = os.path.join(
    controller_config_dir, default_controller_config_file
)
default_controller_key = "nonlinear_data_driven_mpc_params"

# Plot parameters configuration file
plot_params_config_file = "plot_params.yaml"
plot_params_config_path = os.path.join(
    plot_params_config_dir, plot_params_config_file
)

# Animation default parameters
default_anim_name = "nonlinear_data-driven_mpc_sim.gif"
default_anim_path = os.path.join(default_animation_dir, default_anim_name)
default_anim_fps = 50.0
default_anim_bitrate = 4500
default_anim_points_per_frame = 20

# Nonlinear Data-Driven MPC controller parameters
alpha_reg_type_mapping = {
    "Approx": AlphaRegType.APPROXIMATED,
    "Previous": AlphaRegType.PREVIOUS,
    "Zero": AlphaRegType.ZERO,
}
default_t_sim = 3000  # Default simulation length in time steps

# Paper reproduction parameters (based on the example from Section V of [2])
x_0 = np.array([0.9492, 0.43])  # Initial state for reproduction
u_ylimits_list = [(0.0, 1.0)]  # Input plot Y-axis limits
y_ylimits_list = [(0.4, 0.7)]  # Output plot Y-axis limits


# Define function to retrieve plot parameters from configuration file
def get_plot_params(config_path: str) -> dict[str, Any]:
    line_params: dict[str, Any] = load_yaml_config_params(
        config_file=config_path, key="line_params"
    )
    legend_params: dict[str, Any] = load_yaml_config_params(
        config_file=config_path, key="legend_params"
    )
    figure_params: dict[str, Any] = load_yaml_config_params(
        config_file=config_path, key="figure_params"
    )

    return {
        "inputs_line_params": line_params["input"],
        "outputs_line_params": line_params["output"],
        "setpoints_line_params": line_params["setpoint"],
        "bounds_line_params": line_params["bounds"],
        "legend_params": legend_params,
        **figure_params,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nonlinear Data-Driven MPC Controller Example"
    )
    # Data-Driven MPC controller configuration file arguments
    parser.add_argument(
        "--controller_config_path",
        type=str,
        default=default_controller_config_path,
        help="The path to the YAML configuration file "
        "containing the Data-Driven MPC controller "
        "parameters.",
    )
    parser.add_argument(
        "--controller_key",
        type=str,
        default=default_controller_key,
        help="The key to access the Data-Driven MPC "
        "controller parameters in the configuration file.",
    )
    # Nonlinear Data-Driven MPC controller arguments
    parser.add_argument(
        "--n_mpc_step",
        type=int,
        default=None,
        help="The number of consecutive applications of the "
        "optimal input for an n-Step Data-Driven MPC Scheme.",
    )
    parser.add_argument(
        "--alpha_reg_type",
        type=str,
        default=None,
        choices=["Approx", "Previous", "Zero"],
        help="The Alpha regularization type for the "
        "Nonlinear Data-Driven MPC.",
    )
    parser.add_argument(
        "--t_sim",
        type=int,
        default=default_t_sim,
        help="The simulation length in time steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for Random Number Generator "
        "initialization to ensure reproducible results. "
        "Defaults to `None`.",
    )
    # Animation video output arguments
    parser.add_argument(
        "--save_anim",
        action="store_true",
        default=False,
        help="If passed, save the generated animation to a "
        "file using ffmpeg. The file format is specified by "
        "the `anim_path` argument value.",
    )
    parser.add_argument(
        "--anim_path",
        type=str,
        default=default_anim_path,
        help="The saving path for the generated animation "
        "file. Includes the file name and its extension "
        "(e.g., 'data-driven_mpc_sim.gif' or "
        "'data-driven_mpc_sim.mp4'). Defaults to "
        "'animation_outputs/data-driven_mpc_sim.gif'",
    )
    parser.add_argument(
        "--anim_fps",
        type=float,
        default=default_anim_fps,
        help="The frames per second value for the saved "
        "video. Defaults to 50.",
    )
    parser.add_argument(
        "--anim_bitrate",
        type=int,
        default=default_anim_bitrate,
        help="The bitrate value for the saved video "
        "(relevant for video formats like .mp4). Defaults to "
        "4500.",
    )
    parser.add_argument(
        "--anim_points_per_frame",
        type=int,
        default=default_anim_points_per_frame,
        help="The number of data points shown per animation "
        "frame. Increasing this value reduces the number of "
        "animation frames required to display all the data. "
        "Defaults to 5 points per frame.",
    )
    # Verbose argument
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="The verbosity level: 0 = no output, 1 = "
        "minimal output, 2 = detailed output.",
    )

    # TODO: Add arguments

    return parser.parse_args()


def main() -> None:
    # --- Parse arguments ---
    args = parse_args()

    # Data-Driven MPC controller parameters
    controller_config_path = args.controller_config_path
    controller_key = args.controller_key

    # Data-Driven MPC controller arguments
    n_mpc_step = args.n_mpc_step
    alpha_reg_type_arg = args.alpha_reg_type

    # Simulation parameters
    t_sim = args.t_sim

    seed = args.seed

    # Animation video output arguments
    save_anim = args.save_anim
    anim_path = args.anim_path
    anim_fps = args.anim_fps
    anim_bitrate = args.anim_bitrate
    anim_points_per_frame = args.anim_points_per_frame

    # Verbose argument
    verbose = args.verbose

    # ==============================================
    # 1. Define Simulation and Controller Parameters
    # ==============================================
    # --- Define system model (simulation) ---
    if verbose:
        print("Loading system parameters from configuration file")

    system_model = create_nonlinear_cstr_system(
        cstr_model_config_path=cstr_model_config_path,
        cstr_model_key=cstr_model_key,
        verbose=verbose,
    )

    if verbose:
        print("Initialized nonlinear Continuous Stirred Tank Reactor system")

    # --- Define Data-Driven MPC Controller Parameters ---
    if verbose:
        print(
            "Loading Data-Driven MPC controller parameters from "
            "configuration file"
        )

    # Load Data-Driven MPC controller parameters from configuration file
    m = system_model.m  # Number of inputs
    p = system_model.p  # Number of outputs
    dd_mpc_config = get_nonlinear_data_driven_mpc_controller_params(
        config_file=controller_config_path,
        controller_key=controller_key,
        m=m,
        p=p,
        verbose=verbose,
    )

    # Override controller parameters with parsed arguments
    if n_mpc_step is not None or alpha_reg_type_arg is not None:
        if verbose:
            print("Overriding Data-Driven MPC controller parameters")

    # Override the number of consecutive applications of the
    # optimal input (n-Step Data-Driven MPC Scheme (multi-step))
    # with parsed argument if passed
    if n_mpc_step is not None:
        dd_mpc_config["n_mpc_step"] = n_mpc_step

        if verbose > 1:
            print(
                "    n-Step Data-Driven MPC parameter (`n_mpc_step`) set "
                f"to: {n_mpc_step}"
            )

    # Override the alpha regularization type type
    # with parsed argument if passed
    if alpha_reg_type_arg is not None:
        dd_mpc_config["alpha_reg_type"] = alpha_reg_type_mapping[
            alpha_reg_type_arg
        ]

        if verbose > 1:
            print(
                "    Data-Driven MPC alpha regularization type set to: "
                f"{dd_mpc_config['alpha_reg_type'].name}"
            )

    # --- Define Control Simulation parameters ---
    n_steps = t_sim + 1  # Number of simulation steps

    # Create a Random Number Generator for reproducibility
    np_random = np.random.default_rng(seed=seed)

    if verbose:
        if seed is None:
            print("Random number generator initialized with a random seed")
        else:
            print(f"Random number generator initialized with seed: {seed}")

    # ============================================
    # 2. Set Initial System State for Reproduction
    # ============================================
    if verbose:
        print(f"Setting initial system state to x0 = {x_0}")

    # Set system state to x_0 for reproduction
    system_model.x = np.array(x_0)

    if verbose > 1:
        print(f"    Initial system state set to: {x_0}")

    # ====================================================
    # 3. Initial Input-Output Data Generation (Simulation)
    # ====================================================
    if verbose:
        print("Generating initial input-output data")

    # Generate initial input-output data using a
    # generated persistently exciting input
    u, y = generate_initial_input_output_data(
        system_model=system_model,
        controller_config=dd_mpc_config,
        np_random=np_random,
    )

    if verbose > 1:
        print(f"    Input data shape: {u.shape}, Output data shape: {y.shape}")

    # ===============================================
    # 4. Data-Driven MPC Controller Instance Creation
    # ===============================================
    if verbose:
        print("Initializing Nonlinear Data-Driven MPC controller")

    # Create a Direct Data-Driven MPC controller
    dd_mpc_controller = create_nonlinear_data_driven_mpc_controller(
        controller_config=dd_mpc_config, u=u, y=y
    )

    # ===============================
    # 5. Data-Driven MPC Control Loop
    # ===============================
    if verbose:
        print("Simulating Nonlinear Data-Driven MPC control system")

    # Simulate the Data-Driven MPC control system following the
    # Nonlinear Data-Driven MPC Scheme described in Algorithm 1 of [2].
    u_sys, y_sys = simulate_nonlinear_data_driven_mpc_control_loop(
        system_model=system_model,
        data_driven_mpc_controller=dd_mpc_controller,
        n_steps=n_steps,
        np_random=np_random,
        verbose=verbose,
    )

    # =====================================================
    # 6. Plot and Animate Control System Inputs and Outputs
    # =====================================================
    N = dd_mpc_config["N"]  # Initial input-output trajectory length
    y_r = dd_mpc_config["y_r"]  # System output setpoint
    U = dd_mpc_config["U"]  # Bounds for the predicted input

    # Construct input bounds tuple list for plotting
    u_bounds_list = U.tolist()

    # --- Plot control system inputs and outputs ---
    plot_title = "Nonlinear Data-Driven MPC"
    y_setpoint_var_symbol = "y^r"
    initial_steps_label = "Online measurements"
    plot_params = get_plot_params(config_path=plot_params_config_path)

    if verbose:
        print("Displaying control system inputs and outputs plot")

    plot_input_output(
        u_k=u_sys,
        y_k=y_sys,
        y_s=y_r,
        u_bounds_list=u_bounds_list,
        y_setpoint_var_symbol=y_setpoint_var_symbol,
        title=plot_title,
        **plot_params,
    )

    # --- Plot data including initial input-output sequences ---
    # Construct data arrays including initial input-output data
    U_data = np.vstack([u, u_sys])
    Y_data = np.vstack([y, y_sys])

    # Plot extended input-output data
    if verbose:
        print(
            "Displaying control system inputs and outputs including "
            "initial input-output measurements"
        )

    plot_input_output(
        u_k=U_data,
        y_k=Y_data,
        y_s=y_r,
        u_bounds_list=u_bounds_list,
        y_setpoint_var_symbol=y_setpoint_var_symbol,
        title=plot_title,
        **plot_params,
    )

    # --- Plot results in a figure replicating Fig. 2 of [2] ---
    plot_title_reprod = "Nonlinear Data-Driven MPC Reproduction"

    # Update figure size to fit figure in `README.md`
    plot_params_reprod = plot_params.copy()
    plot_params_reprod["figsize"] = (6, 8)

    if verbose:
        print(
            "Displaying reproduction plot: Data-Driven MPC for Nonlinear "
            "systems"
        )

    plot_input_output(
        u_k=U_data,
        y_k=Y_data,
        y_s=y_r,
        u_bounds_list=u_bounds_list,
        y_setpoint_var_symbol=y_setpoint_var_symbol,
        u_ylimits_list=u_ylimits_list,
        y_ylimits_list=y_ylimits_list,
        title=plot_title_reprod,
        **plot_params_reprod,
    )

    # --- Animate extended input-output data ---
    if verbose:
        print("Displaying animation from extended input-output data")

    anim = plot_input_output_animation(
        u_k=U_data,
        y_k=Y_data,
        y_s=y_r,
        u_bounds_list=u_bounds_list,
        y_setpoint_var_symbol=y_setpoint_var_symbol,
        initial_steps=N,
        initial_steps_label=initial_steps_label,
        continuous_updates=True,
        display_initial_text=False,
        display_control_text=False,
        interval=1000.0 / anim_fps,
        points_per_frame=anim_points_per_frame,
        title=plot_title,
        **plot_params,
    )
    plt.show()  # Show animation

    if save_anim:
        # Calculate the number of total animation frames
        data_length = N + n_steps
        anim_frames = math.ceil((data_length - 1) / anim_points_per_frame) + 1

        if verbose:
            print("Saving extended input-output animation to file")
            if verbose > 1:
                print(f"    Saving animation to: {anim_path}")
                print(
                    f"    Animation FPS: {anim_fps}, Bitrate: "
                    f"{anim_bitrate} (video only), Data Length: "
                    f"{data_length}, Points per Frame: "
                    f"{anim_points_per_frame}, Total Frames: {anim_frames}"
                )

        # Save input-output animation as an MP4 video
        save_animation(
            animation=anim,
            total_frames=anim_frames,
            fps=anim_fps,
            bitrate=anim_bitrate,
            file_path=anim_path,
        )

        if verbose:
            print("Animation file saved successfully")

    plt.close()  # Close figures


if __name__ == "__main__":
    main()
