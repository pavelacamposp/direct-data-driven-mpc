import numpy as np

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCController,
)
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    NonlinearDataDrivenMPCController,
)
from direct_data_driven_mpc.utilities.controller.controller_params import (
    LTIDataDrivenMPCParamsDictType,
    NonlinearDataDrivenMPCParamsDictType,
)


def create_lti_data_driven_mpc_controller(
    controller_config: LTIDataDrivenMPCParamsDictType,
    u_d: np.ndarray,
    y_d: np.ndarray,
    use_terminal_constraints: bool = True,
) -> LTIDataDrivenMPCController:
    """
    Create an `LTIDataDrivenMPCController` instance using a specified
    Data-Driven MPC controller configuration and initial input-output
    trajectory data measured from a system.

    Args:
        controller_config (LTIDataDrivenMPCParamsDictType): A dictionary
            containing configuration parameters for a Data-Driven MPC
            controller designed for Linear Time-Invariant (LTI) systems.
        u_d (np.ndarray): An array of shape `(N, m)` representing a
            persistently exciting input sequence used to generate output data
            from the system. `N` is the trajectory length and `m` is the
            number of control inputs.
        y_d (np.ndarray): An array of shape `(N, p)` representing the system's
            output response to `u_d`. `N` is the trajectory length and `p` is
            the number of system outputs.
        use_terminal_constraints (bool): If `True`, include terminal equality
            constraints in the Data-Driven MPC formulation. If `False`, the
            controller will not enforce these constraints. Defaults to `True`.

    Returns:
        LTIDataDrivenMPCController: An `LTIDataDrivenMPCController` instance,
            which represents a Data-Driven MPC controller designed for Linear
            Time-Invariant (LTI) systems, based on the specified configuration.
    """
    # Get model parameters from input-output trajectory data
    m = u_d.shape[1]  # Number of inputs
    p = y_d.shape[1]  # Number of outputs

    # Retrieve Data-Driven MPC controller parameters
    n = controller_config["n"]  # Estimated system order
    L = controller_config["L"]  # Prediction horizon
    Q = controller_config["Q"]  # Output weighting matrix
    R = controller_config["R"]  # Input weighting matrix

    u_s = controller_config["u_s"]  # Control input setpoint
    y_s = controller_config["y_s"]  # System output setpoint

    # Estimated upper bound of the system measurement noise
    eps_max = controller_config["eps_max"]
    # Ridge regularization base weight for `alpha` (scaled by `eps_max`)
    lamb_alpha = controller_config["lamb_alpha"]
    # Ridge regularization weight for sigma
    lamb_sigma = controller_config["lamb_sigma"]

    U = controller_config["U"]  # Bounds for the predicted input

    # Convex slack variable constraint constant
    c = controller_config["c"]

    # Slack variable constraint type
    slack_var_constraint_type = controller_config["slack_var_constraint_type"]

    # Data-Driven MPC controller type
    controller_type = controller_config["controller_type"]

    # n-Step Data-Driven MPC Scheme parameters
    # Number of consecutive applications of the optimal input
    n_mpc_step = controller_config["n_mpc_step"]

    # Create Data-Driven MPC controller
    lti_data_driven_mpc_controller = LTIDataDrivenMPCController(
        n=n,
        m=m,
        p=p,
        u_d=u_d,
        y_d=y_d,
        L=L,
        Q=Q,
        R=R,
        u_s=u_s,
        y_s=y_s,
        eps_max=eps_max,
        lamb_alpha=lamb_alpha,
        lamb_sigma=lamb_sigma,
        U=U,
        c=c,
        slack_var_constraint_type=slack_var_constraint_type,
        controller_type=controller_type,
        n_mpc_step=n_mpc_step,
        use_terminal_constraints=use_terminal_constraints,
    )

    return lti_data_driven_mpc_controller


def create_nonlinear_data_driven_mpc_controller(
    controller_config: NonlinearDataDrivenMPCParamsDictType,
    u: np.ndarray,
    y: np.ndarray,
) -> NonlinearDataDrivenMPCController:
    """
    Create a `NonlinearDataDrivenMPCController` instance using a specified
    Data-Driven MPC controller configuration and initial input-output
    trajectory data measured from a system.

    Args:
        controller_config (NonlinearDataDrivenMPCParamsDictType): A dictionary
            containing configuration parameters for a Data-Driven MPC
            controller designed for Nonlinear systems.
        u (np.ndarray): An array of shape `(N, m)` representing a
            persistently exciting input sequence used to generate output data
            from the system. `N` is the trajectory length and `m` is the
            number of control inputs.
        y (np.ndarray): An array of shape `(N, p)` representing the system's
            output response to `u`. `N` is the trajectory length and `p` is
            the number of system outputs.

    Returns:
        NonlinearDataDrivenMPCController: A `NonlinearDataDrivenMPCController`
            instance, which represents a Data-Driven MPC controller designed
            for Nonlinear systems, based on the specified configuration.
    """
    # Get model parameters from input-output trajectory data
    m = u.shape[1]  # Number of inputs
    p = y.shape[1]  # Number of outputs

    # Retrieve Data-Driven MPC controller parameters
    n = controller_config["n"]  # Estimated system order
    L = controller_config["L"]  # Prediction horizon
    Q = controller_config["Q"]  # Output weighting matrix
    R = controller_config["R"]  # Input weighting matrix
    S = controller_config["S"]  # Output setpoint weighting matrix

    # Ridge regularization weight for alpha
    lamb_alpha = controller_config["lamb_alpha"]
    # Ridge regularization weight for sigma
    lamb_sigma = controller_config["lamb_sigma"]

    # Bounds for the predicted input
    U = controller_config["U"]
    # Bounds for the predicted input setpoint
    Us = controller_config["Us"]

    # Alpha regularization type
    alpha_reg_type = controller_config["alpha_reg_type"]

    # Nonlinear MPC parameters for alpha_reg_type = 0 (Approximated)
    # Ridge regularization weight for alpha_s
    lamb_alpha_s = controller_config["lamb_alpha_s"]
    # Ridge regularization weight for sigma_s
    lamb_sigma_s = controller_config["lamb_sigma_s"]

    # System Output setpoint
    y_r = controller_config["y_r"]

    # Extended Output Representation and Incremental Input
    ext_out_incr_in = controller_config["ext_out_incr_in"]

    # Tracking cost value threshold
    update_cost_threshold = controller_config["update_cost_threshold"]

    # n-Step Data-Driven MPC Scheme parameters
    # Number of consecutive applications of the optimal input
    n_mpc_step = controller_config["n_mpc_step"]

    # Create Data-Driven MPC controller
    nonlinear_data_driven_mpc_controller = NonlinearDataDrivenMPCController(
        n=n,
        m=m,
        p=p,
        u=u,
        y=y,
        L=L,
        Q=Q,
        R=R,
        S=S,
        y_r=y_r,
        lamb_alpha=lamb_alpha,
        lamb_sigma=lamb_sigma,
        U=U,
        Us=Us,
        lamb_alpha_s=lamb_alpha_s,
        lamb_sigma_s=lamb_sigma_s,
        alpha_reg_type=alpha_reg_type,
        ext_out_incr_in=ext_out_incr_in,
        update_cost_threshold=update_cost_threshold,
        n_mpc_step=n_mpc_step,
    )

    return nonlinear_data_driven_mpc_controller
