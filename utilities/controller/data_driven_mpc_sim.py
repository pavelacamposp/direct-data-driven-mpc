from typing import Tuple

import numpy as np
from numpy.random import Generator

from utilities.models.lti_model import LTIModel
from utilities.models.nonlinear_model import NonlinearSystem

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCController)
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    NonlinearDataDrivenMPCController)

def simulate_lti_data_driven_mpc_control_loop(
    system_model: LTIModel,
    data_driven_mpc_controller: LTIDataDrivenMPCController,
    n_steps: int,
    np_random: Generator,
    verbose: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Data-Driven MPC control loop applied to a Linear Time-Invariant
    (LTI) system and return the resulting input-output data sequences.

    This function simulates the closed-loop operation of a Data-Driven MPC
    controller designed for LTI systems, following the Data-Driven MPC schemes
    described in Algorithm 1 (Nominal) and Algorithm 2 (Robust) of [1].
    
    Args:
        system_model (LTIModel): An `LTIModel` instance representing a Linear
            Time-Invariant (LTI) system.
        data_driven_mpc_controller (LTIDataDrivenMPCController): An
            `LTIDataDrivenMPCController` instance representing a Data-Driven
            MPC controller designed for Linear Time-Invariant (LTI) systems.
        n_steps (int): The number of time steps for the simulation.
        np_random (Generator): A Numpy random number generator for generating
            random noise for the system's output.
        verbose (int): The verbosity level: 0 = no output, 1 = minimal output,
            2 = detailed output.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - An array of shape `(n_steps, m)` representing the optimal control
                inputs applied to the system, where `m` is the number of
                control inputs.
            - An array of shape `(n_steps, p)` representing the output response
                of the system, where `p` is the number of system outputs.

    References:
        [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer,
            "Data-Driven Model Predictive Control With Stability and
            Robustness Guarantees," in IEEE Transactions on Automatic Control,
            vol. 66, no. 4, pp. 1702-1717, April 2021,
            doi: 10.1109/TAC.2020.3000182.
    """
    # Retrieve model parameters
    m = system_model.m  # Number of inputs
    p = system_model.p  # Number of outputs
    eps_max_sim = system_model.eps_max  # Upper bound of the system
    # measurement noise (simulation)

    # Retrieve Data-Driven MPC controller parameters
    # Control input setpoint
    u_s = data_driven_mpc_controller.u_s
    # System output setpoint
    y_s = data_driven_mpc_controller.y_s
    # Number of consecutive applications of the optimal input
    # for an n-Step Data-Driven MPC Scheme (multi-step)
    n_mpc_step = data_driven_mpc_controller.n_mpc_step

    # Initialize control loop input-output data arrays
    u_sys = np.zeros((n_steps, m))
    y_sys = np.zeros((n_steps, p))

    # Generate bounded uniformly distributed additive measurement noise
    w_sys = eps_max_sim * np_random.uniform(-1.0, 1.0, (n_steps, p))

    # --- Simulate Data-Driven MPC control system ---
    # Simulate the Data-Driven MPC control system following Algorithm 1 for a
    # Data-Driven MPC Scheme, and Algorithm 2 for an n-Step Data-Driven MPC
    # Scheme, as described in [1].
    for t in range(0, n_steps, n_mpc_step):
        # --- Algorithm 1 and Algorithm 2 (n-step): ---
        # 1) Solve Data-Driven MPC after taking past `n` input-output
        #    measurements u[t-n, t-1], y[t-n, t-1].

        # Update and solve the Data-Driven MPC problem
        data_driven_mpc_controller.update_and_solve_data_driven_mpc()

        # Simulate closed loop
        for k in range(t, min(t + n_mpc_step, n_steps)):
            # --- Algorithm 1: ---
            # 2) Apply the input ut = ubar*[0](t).
            # --- Algorithm 2 (n-step): ---
            # 2) Apply the input sequence u[t, t+n-1] = ubar*[0, n-1](t)
            #    over the next `n` time steps. 

            # Update control input
            n_step = k - t  # Time step `n`. Results 0 for n_mpc_step = 1
            optimal_u_step_n = (
                data_driven_mpc_controller.get_optimal_control_input_at_step(
                    n_step=n_step))
            u_sys[k, :] = optimal_u_step_n
            
            # --- Simulate system with optimal control input ---
            y_sys[k, :] = system_model.simulate_step(u=u_sys[k, :],
                                                     w=w_sys[k, :])
            
            # --- Algorithm 1 and Algorithm 2 (n-step): ---
            # 1) At time `t`, take the past `n` measurements u[t-n, t-1],
            #    y[t-n, t-1] and solve Data-Driven MPC.
            #
            # Note: The Data-Driven MPC is solved at the start of the next
            # iteration.

            # Update past input-output measurements
            data_driven_mpc_controller.store_input_output_measurement(
                u_current=u_sys[k, :].reshape(-1, 1),
                y_current=y_sys[k, :].reshape(-1, 1)
            )

        # --- Algorithm 1: ---
        # 3) Set t = t + 1 and go back to 1).
        # --- Algorithm 2 (n-step): ---
        # 3) Set t = t + n and go back to 1).

        if verbose > 1:
            # Get current step MPC cost value
            mpc_cost_val = (
                data_driven_mpc_controller.get_optimal_cost_value())
            # Calculate input and output errors
            u_error = u_s.flatten() - u_sys[k, :].flatten()
            y_error = y_s.flatten() - y_sys[k, :].flatten()
            # Format error arrays for printing
            formatted_u_error = ', '.join([f'u_{i + 1}e = {error:>6.3f}'
                                           for i, error
                                           in enumerate(u_error)])
            formatted_y_error = ', '.join([f'y_{i + 1}e = {error:>6.3f}'
                                           for i, error
                                           in enumerate(y_error)])
            # Print time step, MPC cost value, and formatted error
            print(f"    Time step: {t:>4} - MPC cost value: "
                  f"{mpc_cost_val:>8.4f} - Error: {formatted_u_error}, "
                  f"{formatted_y_error}")
    
    return u_sys, y_sys

def simulate_nonlinear_data_driven_mpc_control_loop(
    system_model: NonlinearSystem,
    data_driven_mpc_controller: NonlinearDataDrivenMPCController,
    n_steps: int,
    np_random: Generator,
    verbose: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Data-Driven MPC control loop applied to a Nonlinear system and
    return the resulting input-output data sequences.

    This function simulates the closed-loop operation of a Data-Driven MPC
    controller designed for Nonlinear systems, following the Nonlinear
    Data-Driven MPC scheme described in Algorithm 1 of [2].
    
    Args:
        system_model (NonlinearSystem): A `NonlinearSystem` instance
            representing a Nonlinear system.
        data_driven_mpc_controller (NonlinearDataDrivenMPCController): A
            `NonlinearDataDrivenMPCController` instance representing a
            Data-Driven MPC controller designed for Nonlinear systems.
        n_steps (int): The number of time steps for the simulation.
        np_random (Generator): A Numpy random number generator for generating
            random noise for the system's output.
        verbose (int): The verbosity level: 0 = no output, 1 = minimal output,
            2 = detailed output.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - An array of shape `(n_steps, m)` representing the optimal control
                inputs applied to the system, where `m` is the number of
                control inputs.
            - An array of shape `(n_steps, p)` representing the output response
                of the system, where `p` is the number of system outputs.

    References:
        [2] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Linear
            Tracking MPC for Nonlinear Systems—Part II: The Data-Driven Case,"
            in IEEE Transactions on Automatic Control, vol. 67, no. 9, pp.
            4406-4421, Sept. 2022, doi: 10.1109/TAC.2022.3166851.
    """
    # Retrieve model parameters
    m = system_model.m  # Number of inputs
    p = system_model.p  # Number of outputs
    eps_max_sim = system_model.eps_max  # Upper bound of the system
    # measurement noise (simulation)

    # Retrieve Data-Driven MPC controller parameters
    # System output setpoint
    y_r = data_driven_mpc_controller.y_r
    # Number of consecutive applications of the optimal input
    # for an n-Step Data-Driven MPC Scheme (multi-step)
    n_mpc_step = data_driven_mpc_controller.n_mpc_step

    # Initialize control loop input-output data arrays
    u_sys = np.zeros((n_steps, m))
    y_sys = np.zeros((n_steps, p))

    # Generate bounded uniformly distributed additive measurement noise
    w_sys = eps_max_sim * np_random.uniform(-1.0, 1.0, (n_steps, p))

    # --- Simulate Data-Driven MPC control system ---
    # Simulate the Nonlinear Data-Driven MPC control system
    # as described in Algorithm 1 of [2].
    for t in range(0, n_steps, n_mpc_step):
        # --- Algorithm 1: ---
        # 1) At time `t >= N`, compute `alpha_sr_Lin(Dt)` by solving Equation
        #    (12) or its approximation from Equation (23).
        #
        # Note:
        #   - The `NonlinearDataDrivenMPCController` controller class
        #     implements the approximation of `alpha_sr_Lin(Dt)` by solving
        #     Equation (23) of [2]. This can be enabled by initializing the
        #     controller with the `alpha_reg_type` parameter set to
        #     `AlphaRegType.APPROXIMATED`.
        #   - As described in Section V of [2], the paper's example does not
        #     compute `alpha_sr_Lin(Dt)` by solving either Equation (12) or
        #     (23). Instead, its value is "approximated" using the previous
        #     optimal solution of `alpha` to encourage stationary behavior.
        #     This can be enabled by setting `alpha_reg_type` to
        #     `AlphaRegType.PREVIOUS` during controller initialization.

        # Update and solve the Data-Driven MPC problem
        data_driven_mpc_controller.update_and_solve_data_driven_mpc()
        # Note: The `update_and_solve_data_driven_mpc` method computes the
        # value of `alpha_sr_Lin(Dt)` and solves the Nonlinear Data-Driven MPC
        # problem.

        # Simulate closed loop
        for k in range(t, min(t + n_mpc_step, n_steps)):
            # --- Algorithm 1: ---
            # 2) Solve the Nonlinear Data-Driven MPC problem (22) and apply
            #    the first `n` input components:
            #       u_{t+k} = ubar*_k(t), for k in I[0,n-1].

            # Update control input
            n_step = k - t  # Time step `n`. Results 0 for n_mpc_step = 1
            optimal_u_step_n = (
                data_driven_mpc_controller.get_optimal_control_input_at_step(
                    n_step=n_step))
            u_sys[k, :] = optimal_u_step_n
            
            # --- Simulate system with optimal control input ---
            y_sys[k, :] = system_model.simulate_step(u=u_sys[k, :],
                                                     w=w_sys[k, :])
            
            # --- Algorithm 1: ---
            # Update input-output measurements online after each iteration.
            #
            # Note: This is not explicitly stated in the algorithm, but is
            # described throughout the paper.

            # Update input-output measurements online
            du_current = data_driven_mpc_controller.get_du_value_at_step(
                n_step=n_step)
            data_driven_mpc_controller.store_input_output_measurement(
                u_current=u_sys[k, :],
                y_current=y_sys[k, :],
                du_current=du_current)
            # Note: Input increment updates are required for controllers that
            # use an extended output representation and input increments
            # (as the controller presented in Section V of [2]). This is not
            # necessary for controllers that operate in a standard manner,
            # which use direct control inputs and do not extend the system
            # state.

        if verbose > 1:
            # Get current step MPC cost value
            mpc_cost_val = (
                data_driven_mpc_controller.get_optimal_cost_value())
            # Calculate input and output errors
            y_error = y_r.flatten() - y_sys[k, :].flatten()
            # Format error arrays for printing
            formatted_y_error = ', '.join([f'y_{i + 1}e = {error:>6.3f}'
                                           for i, error
                                           in enumerate(y_error)])
            # Print time step, MPC cost value, and formatted error
            print(f"    Time step: {t:>4} - MPC cost value: "
                  f"{mpc_cost_val:>8.4f} - Error: {formatted_y_error}")
    
    return u_sys, y_sys
