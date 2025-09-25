import math
from functools import partial

import numpy as np

from direct_data_driven_mpc.utilities import (
    load_yaml_config_params,
)
from direct_data_driven_mpc.utilities.models import (
    NonlinearSystem,
)


def cstr_dynamics(
    x: np.ndarray,
    u: np.ndarray,
    Ts: float,
    theta: float,
    k: float,
    M: float,
    xf: float,
    xc: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute the dynamics of a Continuous Stirred Tank Reactor (CSTR) system
    based on the implementation example in Section V of [2].

    Args:
        x (np.ndarray): An array containing two state variables (`x1`, `x2`).
        u (np.ndarray): An array containing one input variable (`u`).
        Ts (float): The sampling time parameter, `Ts`, of the CSTR system.
        theta (float): The `theta` parameter of the CSTR system.
        k (float): The `k` parameter of the CSTR system.
        M (float): The `M` parameter of the CSTR system.
        xf (float): The `xf` parameter of the CSTR system.
        xc (float): The `xc` parameter of the CSTR system.
        alpha (float): The `alpha` parameter of the CSTR system.

    Returns:
        np.ndarray: An array containing the updated states (`x1_new`,
        `x2_new`).

    References:
        [2] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Linear
        Tracking MPC for Nonlinear Systems—Part II: The Data-Driven Case," in
        IEEE Transactions on Automatic Control, vol. 67, no. 9, pp. 4406-4421,
        Sept. 2022, doi: 10.1109/TAC.2022.3166851.
    """
    # Get state variables from state array
    x1, x2 = x
    # Get input variable from input array
    u = u[0]

    # Compute CSTR dynamics preventing division by zero
    if x2 != 0:
        x1_new = x1 + Ts * ((1 - x1) / theta - (k * x1 * math.exp(-M / x2)))
        x2_new = x2 + Ts * (
            (xf - x2) / theta
            + (k * x1 * math.exp(-M / x2))
            - alpha * u * (x2 - xc)
        )
    else:
        x1_new = x1 + Ts * (1 - x1) / theta
        x2_new = x2 + Ts * ((xf - x2) / theta - alpha * u * (x2 - xc))

    return np.array([x1_new, x2_new])


def cstr_output(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Define the output function of a Continuous Stirred Tank Reactor (CSTR)
    system based on the implementation example in Section V of [2].

    Args:
        x (np.ndarray): An array containing two state variables (`x1`, `x2`).
        u (np.ndarray): An array containing input variables (ignored in the
            output calculation).

    Returns:
        np.ndarray: An array containing the system output (`x2`).

    References:
        [2] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Linear
        Tracking MPC for Nonlinear Systems—Part II: The Data-Driven Case," in
        IEEE Transactions on Automatic Control, vol. 67, no. 9, pp. 4406-4421,
        Sept. 2022, doi: 10.1109/TAC.2022.3166851.
    """
    return x[[1]]  # Output is the concentration x2


def create_nonlinear_cstr_system(
    cstr_model_config_path: str, cstr_model_key: str, verbose: int
) -> NonlinearSystem:
    """
    Create a `NonlinearSystem` instance representing a nonlinear Continuous
    Stirred Tank Reactor (CSTR) system based on the implementation example in
    Section V of [2].

    This function loads the CSTR system parameters from a YAML configuration
    file, sets up the nonlinear system's dynamics and output functions, and
    initializes a `NonlinearSystem` instance representing this system.

    Args:
        cstr_model_config_path (str): The path to the YAML configuration file
            containing the CSTR system parameters.
        cstr_model_key (str): The key corresponding to the CSTR system
            parameters to be retrieved from the configuration file.
        verbose (int): The verbosity level. If greater than 1, prints parameter
            loading details.

    Returns:
        NonlinearSystem: A `NonlinearSystem` instance representing the
        nonlinear CSTR system.

    References:
        [2] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Linear
        Tracking MPC for Nonlinear Systems—Part II: The Data-Driven Case," in
        IEEE Transactions on Automatic Control, vol. 67, no. 9, pp. 4406-4421,
        Sept. 2022, doi: 10.1109/TAC.2022.3166851.
    """
    # Load model parameters from config file
    params = load_yaml_config_params(
        config_file=cstr_model_config_path, key=cstr_model_key
    )

    if verbose > 1:
        print(
            f"    Model parameters loaded from {cstr_model_config_path} "
            f"with key '{cstr_model_key}'"
        )

    # Retrieve model parameters
    Ts = params["Ts"]
    theta = params["theta"]
    k = params["k"]
    M = params["M"]
    xf = params["xf"]
    xc = params["xc"]
    alpha = params["alpha"]

    eps_max = params["eps_max"]  # Upper bound of the system measurement noise

    # Define CSTR system dynamics function
    cstr_dynamics_pre_bound = partial(
        cstr_dynamics, Ts=Ts, theta=theta, k=k, M=M, xf=xf, xc=xc, alpha=alpha
    )

    # Initialize CSTR nonlinear system
    n = 2  # Number of states
    m = 1  # Number of inputs
    p = 1  # Number of outputs
    cstr_system = NonlinearSystem(
        f=cstr_dynamics_pre_bound,
        h=cstr_output,
        n=n,
        m=m,
        p=p,
        eps_max=eps_max,
    )

    if verbose:
        print(
            "Nonlinear Continuous Stirred Tank Reactor system initialized "
            "with loaded parameters"
        )

    return cstr_system
