from typing import Tuple, Union

import numpy as np
from numpy.random import Generator

from direct_data_driven_mpc.utilities.models.lti_model import LTIModel
from direct_data_driven_mpc.utilities.models.nonlinear_model import (
    NonlinearSystem)

from direct_data_driven_mpc.utilities.controller.controller_params import (
    LTIDataDrivenMPCParamsDictType)
from direct_data_driven_mpc.utilities.controller.controller_params import (
    NonlinearDataDrivenMPCParamsDictType)

def randomize_initial_system_state(
    system_model: LTIModel,
    controller_config: LTIDataDrivenMPCParamsDictType,
    np_random: Generator
) -> np.ndarray:
    """
    Randomly generate a plausible initial state for a Linear Time-Invariant
    (LTI) system model.

    This function initializes the system state with random values within the
    [-1, 1] range. Afterward, it simulates the system using random input and
    noise sequences to generate an input-output trajectory, which is then used
    to estimate the initial system state.
    
    Note:
        The random input sequence is generated based on the `u_range`
        parameter from the controller configuration (`controller_config`). The
        noise sequence is generated considering the defined noise bounds from
        the system.
    
    Args:
        system_model (LTIModel): An `LTIModel` instance representing a Linear
            Time-Invariant (LTI) system.
        controller_config (LTIDataDrivenMPCParamsDictType): A dictionary
            containing parameters for a Data-Driven MPC controller designed
            for Linear Time-Invariant (LTI) systems, including the range of
            the persistently exciting input (`u_range`).
        np_random (Generator): A Numpy random number generator for generating 
            the random initial system state, persistently exciting input, and
            system output noise.
    
    Returns:
        np.ndarray: A vector of shape `(n, )` representing the estimated initial
            state of the system, where `n` is the system's order.
    """
    # Retrieve model parameters
    ns = system_model.n  # System order (simulation)
    m = system_model.m  # Number of inputs
    p = system_model.p  # Number of outputs
    eps_max_sim = system_model.eps_max  # Upper bound of the system
    # measurement noise (simulation)

    # Retrieve Data-Driven MPC controller parameters
    u_range = controller_config['u_range']  # Range of the persistently
    # exciting input u_d
    
    # Randomize initial system state before excitation
    x_i0 = np_random.uniform(-1.0, 1.0, size=ns)
    system_model.set_state(state=x_i0)

    # Generate a random input array
    u_i = np.hstack([np_random.uniform(*u_range[i], (ns, 1))
                     for i in range(m)])

    # Generate bounded uniformly distributed additive measurement noise
    w_i = eps_max_sim * np_random.uniform(-1.0, 1.0, (ns, p))

    # Simulate the system with the generated random input and noise
    # sequences to obtain output data
    y_i = system_model.simulate(U=u_i, W=w_i, steps=ns)

    # Calculate the initial state of the system
    # from the input-output trajectory
    x_0 = system_model.get_initial_state_from_trajectory(
        U=u_i.flatten(), Y=y_i.flatten())
    
    return x_0

def generate_initial_input_output_data(
    system_model: Union[LTIModel, NonlinearSystem],
    controller_config: Union[LTIDataDrivenMPCParamsDictType,
                             NonlinearDataDrivenMPCParamsDictType],
    np_random: Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate input-output trajectory data from a system using Data-Driven MPC
    controller parameters.

    This function generates a persistently exciting input `u_d` and random
    noise based on the specified controller and system parameters. Then, it
    simulates the system using these input and noise sequences to generate the
    output response `y_d`. The resulting `u_d` and `y_d` arrays represent the
    input-output trajectory data measured from the system, which is necessary
    for system characterization in a Data-Driven MPC formulation.

    Args:
        system_model (Union[LTIModel, NonlinearSystem]): An instance of
            `LTIModel` representing a Linear Time-Invariant (LTI) system or 
            `NonlinearSystem` representing a Nonlinear system.
        controller_config (Union[LTIDataDrivenMPCParamsDictType,
            NonlinearDataDrivenMPCParamsDictType]): A dictionary
            containing parameters for a Data-Driven MPC controller designed
            for Linear Time-Invariant (LTI) or Nonlinear systems. Includes the
            initial input-output trajectory length (`N`) and the range of the
            persistently exciting input (`u_range`).
        np_random (Generator): A Numpy random number generator for generating
            the persistently exciting input and random noise for the system's
            output.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays: a
            persistently exciting input (`u_d`) and the system's output
            response (`y_d`). The input array has shape `(N, m)` and the
            output array has shape `(N, p)`, where `N` is the trajectory
            length, `m` is the number of control inputs, and `p` is the number
            of system outputs.
    """
    # Retrieve model parameters
    m = system_model.m  # Number of inputs
    p = system_model.p  # Number of outputs
    eps_max_sim = system_model.eps_max  # Upper bound of the system
    # measurement noise (simulation)

    # Retrieve Data-Driven MPC controller parameters
    N = controller_config['N']  # Initial input-output trajectory length
    u_range = controller_config['u_range']  # Range of the persistently
    # exciting input u_d

    # Generate a persistently exciting input `u_d` from 0 to (N - 1)
    u_d = np.hstack([np_random.uniform(*u_range[i], (N, 1))
                     for i in range(m)])

    # Generate bounded uniformly distributed additive measurement noise
    w_d = eps_max_sim * np_random.uniform(-1.0, 1.0, (N, p))

    # Simulate the system with the persistently exciting input `u_d` and
    # the generated noise sequence to obtain output data
    y_d = system_model.simulate(U=u_d, W=w_d, steps=N)

    return u_d, y_d

def simulate_n_input_output_measurements(
    system_model: LTIModel,
    controller_config: LTIDataDrivenMPCParamsDictType,
    np_random: Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a control input setpoint applied to a system over `n` (the
    estimated system order) time steps and return the resulting input-output
    data sequences.

    This function retrieves the control input setpoint (`u_s`) and the
    estimated system order (`n`) from a Data-Driven MPC controller
    configuration. Then, it simulates the system using a constant input `u_s`
    and random output noise over `n` time steps. The resulting input-output
    trajectory can be used to update the past `n` input-output measurements
    of a previously initialized Data-Driven MPC controller, allowing it to
    operate on a system with a different state.
    
    Note:
        This function is used for scenarios where a Data-Driven MPC controller
        has been initialized but needs to be adjusted to match different
        system states.

    Args:
        system_model (LTIModel): An `LTIModel` instance representing a Linear
            Time-Invariant (LTI) system.
        controller_config (LTIDataDrivenMPCParamsDictType): A dictionary
            containing parameters for a Data-Driven MPC controller designed
            for Linear Time-Invariant (LTI) systems, including the estimated
            system order (`n`) and the control input setpoint (`u_s`).
        np_random (Generator): A Numpy random number generator for generating
            random noise for the system's output.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - An array of shape `(n, m)` representing the constant input
                setpoint applied to the system over `n` time steps, where `n`
                is the system order and `m` is the number of control inputs.
            - An array of shape `(n, p)` representing the output response of
                the system, where `n` is the system order and `p` is the
                number of system outputs.
    """
    # Retrieve model parameters
    m = system_model.m  # Number of inputs
    p = system_model.p  # Number of outputs
    eps_max_sim = system_model.eps_max  # Upper bound of the system
    # measurement noise

    # Retrieve Data-Driven MPC controller parameters
    n = controller_config['n']  # Estimated system order
    u_s = controller_config['u_s']  # Control input setpoint

    # Construct input array from controller's input setpoint
    U_n = np.tile(u_s, (n, 1)).reshape(n, m)

    # Generate bounded uniformly distributed additive measurement noise
    W_n = eps_max_sim * np_random.uniform(-1.0, 1.0, (n, p))
    
    # Simulate the system with the constant input and generated
    # noise sequences
    Y_n = system_model.simulate(U=U_n, W=W_n, steps=n)
    
    return U_n, Y_n
