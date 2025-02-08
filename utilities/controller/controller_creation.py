import numpy as np

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCController)
from utilities.controller.controller_params import (
    LTIDataDrivenMPCParamsDictType)

def create_data_driven_mpc_controller(
    controller_config: LTIDataDrivenMPCParamsDictType,
    u_d: np.ndarray,
    y_d: np.ndarray,
    use_terminal_constraint: bool = True
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
        use_terminal_constraint (bool): If True, include terminal equality
            constraints in the Data-Driven MPC formulation. If False, the
            controller will not enforce this constraint. Defaults to True.
    
    Returns:
        LTIDataDrivenMPCController: An `LTIDataDrivenMPCController` instance,
            which represents a Data-Driven MPC controller designed for Linear
            Time-Invariant (LTI) systems, based on the specified configuration.
    """
    # Get model parameters from input-output trajectory data
    m = u_d.shape[1]  # Number of inputs
    p = y_d.shape[1]  # Number of outputs

    # Retrieve Data-Driven MPC controller parameters
    n = controller_config['n']  # Estimated system order
    L = controller_config['L']  # Prediction horizon
    Q = controller_config['Q']  # Output weighting matrix
    R = controller_config['R']  # Input weighting matrix

    u_s = controller_config['u_s']  # Control input setpoint
    y_s = controller_config['y_s']  # System output setpoint

    # Estimated upper bound of the system measurement noise
    eps_max = controller_config['eps_max']
    # Ridge regularization base weight for `alpha` (scaled by `eps_max`)
    lamb_alpha = controller_config['lamb_alpha']
    # Ridge regularization weight for sigma
    lamb_sigma = controller_config['lamb_sigma']
    # Convex slack variable constraint constant
    c = controller_config['c']

    # Slack variable constraint type
    slack_var_constraint_type = controller_config['slack_var_constraint_type']

    # Data-Driven MPC controller type
    controller_type = controller_config['controller_type']

    # n-Step Data-Driven MPC Scheme parameters
    # Number of consecutive applications of the optimal input
    n_mpc_step = controller_config['n_mpc_step']

    # Create Data-Driven MPC controller
    direct_data_driven_mpc_controller = LTIDataDrivenMPCController(
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
        c=c,
        slack_var_constraint_type=slack_var_constraint_type,
        controller_type=controller_type,
        n_mpc_step=n_mpc_step,
        use_terminal_constraint=use_terminal_constraint)
    
    return direct_data_driven_mpc_controller
