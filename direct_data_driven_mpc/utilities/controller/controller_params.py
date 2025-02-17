from typing import TypedDict, Tuple, Optional

import numpy as np

from direct_data_driven_mpc.utilities.yaml_config_loading import (
    load_yaml_config_params)

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCType, SlackVarConstraintType)
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    AlphaRegType)

# Define mapping dictionaries for controller parameter retrieval
# from YAML config files

# LTI Data-Driven MPC: Controller type
LTIDataDrivenMPCTypesMap = {
    0: LTIDataDrivenMPCType.NOMINAL,
    1: LTIDataDrivenMPCType.ROBUST
}

# LTI Data-Driven MPC: Slack variable constraint type
SlackVarConstraintTypesMap = {
    0: SlackVarConstraintType.NONE,
    1: SlackVarConstraintType.CONVEX,
    2: SlackVarConstraintType.NON_CONVEX
}

# Nonlinear Data-Driven MPC: Alpha regularization type
AlphaRegTypesMap = {
    0: AlphaRegType.APPROXIMATED,
    1: AlphaRegType.PREVIOUS,
    2: AlphaRegType.ZERO
}

# Define dictionary type hints for Data-Driven MPC controller parameters
class LTIDataDrivenMPCParamsDictType(TypedDict, total=False):
    n: int  # Estimated system order
    
    N: int  # Initial input-output trajectory length
    L: int  # Prediction horizon
    Q: np.ndarray  # Output weighting matrix Q
    R: np.ndarray  # Input weighting matrix R

    eps_max: float  # Estimated upper bound of system measurement noise
    lamb_alpha: float  # Regularization parameter for alpha
    lamb_sigma: float  # Regularization parameter for sigma
    c: float  # Convex slack variable constraint constant

    u_range: np.ndarray  # Range of the persistently exciting input u

    # Slack variable constraint type
    slack_var_constraint_type: SlackVarConstraintType

    controller_type: LTIDataDrivenMPCType  # Data-Driven MPC controller type
    n_mpc_step: int  # Number of consecutive applications of the optimal input
    
    u_s: np.ndarray  # Control input setpoint
    y_s: np.ndarray  # System output setpoint

class NonlinearDataDrivenMPCParamsDictType(TypedDict, total=False):
    n: int  # Estimated system order
    
    N: int  # Initial input-output trajectory length
    L: int  # Prediction horizon
    Q: np.ndarray  # Output weighting matrix Q
    R: np.ndarray  # Input weighting matrix R
    S: np.ndarray  # Output setpoint weighting matrix S

    lamb_alpha: float  # Regularization parameter for alpha
    lamb_sigma: float  # Regularization parameter for sigma
    c: float  # Convex slack variable constraint constant

    U: np.ndarray  # Bounds for the predicted input
    Us: np.ndarray  # Bounds for the predicted input setpoint
    u_range: np.ndarray  # Range of the persistently exciting input u

    alpha_reg_type: AlphaRegType  # Alpha regularization type

    lamb_alpha_s: Optional[float]  #  Regularization parameter for alpha_s
    lamb_sigma_s: Optional[float]  #  Regularization parameter for sigma_s

    y_r: np.ndarray  # System output setpoint

    ext_out_incr_in: bool  # Specifies whether the controller uses an extended
    # output representation and input increments, or operates as a standard
    # controller with direct control inputs without system state extensions

    update_cost_threshold: Optional[float]  # Tracking cost value threshold

    n_mpc_step: int  # Number of consecutive applications of the optimal input

# Define lists of required Data-Driven controller parameters
# from configuration files
LTI_DD_MPC_FILE_PARAMS = [
    'n', 'N', 'L', 'Q_scalar', 'R_scalar', 'epsilon_bar', 'lambda_sigma',
    'lambda_alpha_epsilon_bar', 'u_d_range', 'slack_var_constraint_type',
    'controller_type', 'u_s', 'y_s', 'n_n_mpc_step']

NONLINEAR_DD_MPC_FILE_PARAMS = [
    'n', 'N', 'L', 'Q_scalar', 'R_scalar', 'S_scalar', 'lambda_alpha',
    'lambda_sigma', 'U', 'Us', 'u_range', 'alpha_reg_type', 'lamb_alpha_s',
    'lamb_sigma_s', 'y_r', 'ext_out_incr_in', 'update_cost_threshold',
    'n_n_mpc_step']

def get_lti_data_driven_mpc_controller_params(
    config_file: str,
    controller_key_value: str,
    m: int,
    p: int,
    verbose: int = 0
) -> LTIDataDrivenMPCParamsDictType:
    """
    Load and initialize parameters for a Data-Driven MPC controller designed
    for Linear Time-Invariant (LTI) systems from a YAML configuration file.
    
    The controller parameters are defined based on the Nominal and Robust
    Data-Driven MPC controller formulations of [1]. The number of control
    inputs (`m`) and system outputs (`p`) are used to construct the output
    (`Q`) and input (`R`) weighting matrices.

    Args:
        config_file (str): The path to the YAML configuration file.
        controller_key_value (str): The key to access the specific controller
            parameters in the config file.
        m (int): The number of control inputs.
        p (int): The number of system outputs.
        verbose (int): The verbosity level: 0 = no output, 1 = minimal
                output, 2 = detailed output.
    
    Returns:
        LTIDataDrivenMPCParamsDictType: A dictionary of configuration
            parameters for a Data-Driven MPC controller designed for Linear
            Time-Invariant (LTI) systems.
    
    Raises:
        FileNotFoundError: If the YAML configuration file is not found.
        ValueError: If `controller_key_value` or if required Data-Driven
            controller parameters are missing in the configuration file.

    References:
        [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer,
            "Data-Driven Model Predictive Control With Stability and
            Robustness Guarantees," in IEEE Transactions on Automatic Control,
            vol. 66, no. 4, pp. 1702-1717, April 2021,
            doi: 10.1109/TAC.2020.3000182.
    """
    # Load controller parameters from config file
    params = load_yaml_config_params(config_file=config_file,
                                     key=controller_key_value)

    if verbose > 1:
        print(f"    Data-Driven MPC controller parameters loaded from "
              f"{config_file} with key '{controller_key_value}'")
    
    # Validate that required parameter keys are present
    for key in LTI_DD_MPC_FILE_PARAMS:
        if key not in params:
            raise ValueError(f"Missing required parameter key '{key}' in the "
                             "configuration file.")
    
    # Initialize Data-Driven MPC controller parameter dict
    dd_mpc_params = {}

    # --- Define initial Input-Output data generation parameters ---
    # Persistently exciting input range
    dd_mpc_params['u_range'] = np.array(params['u_d_range'], dtype=float)
    # Initial input-output trajectory length
    dd_mpc_params['N'] = params['N']

    # --- Define Data-Driven MPC parameters ---
    # Estimated system order
    n = params['n']
    dd_mpc_params['n'] = n
    # Estimated upper bound of the system measurement noise
    eps_max = params['epsilon_bar']
    dd_mpc_params['eps_max'] = eps_max
    # Prediction horizon
    L = params['L']
    dd_mpc_params['L'] = L
    # Output weighting matrix
    dd_mpc_params['Q'] = params['Q_scalar'] * np.eye(p * L)
    # Input weighting matrix
    dd_mpc_params['R'] = params['R_scalar'] * np.eye(m * L)

    # Define ridge regularization base weight for alpha, preventing
    # division by zero in noise-free conditions
    lambda_alpha_epsilon_bar = params['lambda_alpha_epsilon_bar']
    if eps_max != 0:
        dd_mpc_params['lamb_alpha'] = lambda_alpha_epsilon_bar / eps_max
    else:
        # Set a high value if eps_max is zero
        dd_mpc_params['lamb_alpha'] = 1000.0
    
    # Ridge regularization weight for sigma
    dd_mpc_params['lamb_sigma'] = params['lambda_sigma']

    # Convex slack variable constraint constant (see Remark 3 of [1])
    dd_mpc_params['c'] = 1.0

    # Slack variable constraint type
    slack_var_constraint_type_config = params['slack_var_constraint_type']
    dd_mpc_params['slack_var_constraint_type'] = (
        SlackVarConstraintTypesMap.get(slack_var_constraint_type_config,
                                       SlackVarConstraintType.NONE))
    
    # Controller type
    controller_type_config = params['controller_type']
    dd_mpc_params['controller_type'] = (
        LTIDataDrivenMPCTypesMap.get(controller_type_config,
                                     LTIDataDrivenMPCType.ROBUST))

    # Number of consecutive applications of the optimal input
    # for an n-Step Data-Driven MPC Scheme (multi-step)
    if params['n_n_mpc_step']:
        dd_mpc_params['n_mpc_step'] = n
        # Defaults to the estimated system order, as defined
        # in Algorithm 2 of [1]
    else:
        dd_mpc_params['n_mpc_step'] = 1

    # Define Input-Output equilibrium setpoint pair
    u_s = params['u_s']
    y_s = params['y_s']
    # Control input setpoint
    dd_mpc_params['u_s'] = np.array(u_s, dtype=float).reshape(-1, 1)
    # System output setpoint
    dd_mpc_params['y_s'] = np.array(y_s, dtype=float).reshape(-1, 1)

    # Print Data-Driven MPC controller initialization details
    # based on verbosity level
    if verbose == 1:
        print("Data-Driven MPC controller initialized with loaded parameters")
    if verbose > 1:
        print("Data-Driven MPC controller initialized with:")
        for key, value in dd_mpc_params.items():
            if key in ['Q', 'R']:
                # Print scalar and shape for large matrices
                print(f"    {key}: scalar {value[0, 0]} {value.shape}")
            elif key in ['controller_type', 'slack_var_constraint_type']:
                # Print name for enum types
                print(f"    {key}: {value.name}")
            elif key in ['u_range']:
                # Format input bounds and ranges
                formatted_array = ', '.join([f"[{', '.join(map(str, row))}]"
                                             for row in value])
                print(f"    {key}: [{formatted_array}]")
            elif key in ['u_s', 'y_s']:
                # Format setpoint arrays in a single line
                formatted_array = ', '.join([f"[{row[0]}]" for row in value])
                print(f"    {key}: [{formatted_array}]")
            else:
                print(f"    {key}: {value}")

    return dd_mpc_params

def get_nonlinear_data_driven_mpc_controller_params(
    config_file: str,
    controller_key_value: str,
    m: int,
    p: int,
    verbose: int = 0
) -> NonlinearDataDrivenMPCParamsDictType:
    """
    Load and initialize parameters for a Data-Driven MPC controller designed
    for Nonlinear systems from a YAML configuration file.
    
    The controller parameters are defined based on the Nonlinear Data-Driven
    MPC controller formulation of [2]. The number of control inputs (`m`)
    and system outputs (`p`) are used to construct the output (`Q`), input
    (`R`), and output setpoint (`S`) weighting matrices.

    Args:
        config_file (str): The path to the YAML configuration file.
        controller_key_value (str): The key to access the specific controller
            parameters in the config file.
        m (int): The number of control inputs.
        p (int): The number of system outputs.
        verbose (int): The verbosity level: 0 = no output, 1 = minimal
                output, 2 = detailed output.
    
    Returns:
        NonlinearDataDrivenMPCParamsDictType: A dictionary of configuration
            parameters for a Data-Driven MPC controller designed for Nonlinear
            systems.
    
    Raises:
        FileNotFoundError: If the YAML configuration file is not found.
        ValueError: If `controller_key_value` or if required Data-Driven
            controller parameters are missing in the configuration file.

    References:
        [2] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Linear
            Tracking MPC for Nonlinear Systems—Part II: The Data-Driven Case,"
            in IEEE Transactions on Automatic Control, vol. 67, no. 9, pp.
            4406-4421, Sept. 2022, doi: 10.1109/TAC.2022.3166851.
    """
    # Load controller parameters from config file
    params = load_yaml_config_params(config_file=config_file,
                                     key=controller_key_value)

    if verbose > 1:
        print(f"    Data-Driven MPC controller parameters loaded from "
              f"{config_file} with key '{controller_key_value}'")
    
    # Validate that required parameter keys are present
    for key in NONLINEAR_DD_MPC_FILE_PARAMS:
        if key not in params:
            raise ValueError(f"Missing required parameter key '{key}' in the "
                             "configuration file.")
    
    # Initialize Data-Driven MPC controller parameter dict
    dd_mpc_params = {}

    # --- Define initial Input-Output data generation parameters ---
    # Persistently exciting input range
    dd_mpc_params['u_range'] = np.array(params['u_range'], dtype=float)
    # Initial input-output trajectory length
    dd_mpc_params['N'] = params['N']

    # --- Define Data-Driven MPC parameters ---
    # Estimated system order
    n = params['n']
    dd_mpc_params['n'] = n

    # Extended Output Representation and Incremental Input
    # If `True`: The controller uses an extended output representation
    #            (y_ext[k] = [y[k], u[k]]) and updates the control input
    #            incrementally (u[k] = u[k-1] + du[k-1]). This ensures
    #            control-affine system dynamics (Section V of [2]).
    # If `False`: The controller directly applies control inputs without
    #             extending its state representation.
    ext_out_incr_in = params['ext_out_incr_in']
    dd_mpc_params['ext_out_incr_in'] = ext_out_incr_in

    # Tracking cost value threshold
    # Online input-output data updates are disabled when the tracking cost
    # value is less than this value. This ensures prediction data is
    # persistently exciting (Section V of [2]).
    dd_mpc_params['update_cost_threshold'] = params['update_cost_threshold']
    
    # Prediction horizon
    L = params['L']
    dd_mpc_params['L'] = L
    
    # Output and Input weighting matrix based on controller structure
    if ext_out_incr_in:
        # Output weighting matrix
        # Construct this matrix considering the extended output
        # representation: y_ext[k] = [y[k], u[k]]
        Q_size = (m + p) * (L + n + 1)
        diag_vals = np.tile(
            ([params['Q_scalar']] * p +  #  Q_scalar value for outputs
             [params['R_scalar']] * m),  #  R_scalar value for inputs
            Q_size // (m + p))
        dd_mpc_params['Q'] = np.diag(diag_vals)
        
        # Input weighting matrix
        # This matrix weights input increments (du[k]) and not absolute inputs
        # (u[k]) in this controller structure. It is currently set to an
        # identity matrix, but this may vary depending on the application.
        dd_mpc_params['R'] = np.eye(m * (L + n + 1))
    else:
        # Output weighting matrix
        dd_mpc_params['Q'] = params['Q_scalar'] * np.eye(p * (L + n + 1))
        # Input weighting matrix
        dd_mpc_params['R'] = params['R_scalar'] * np.eye(m * (L + n + 1))

    # Output setpoint weighting matrix
    dd_mpc_params['S'] = params['S_scalar'] * np.eye(p)
    
    # Ridge regularization weight for alpha
    dd_mpc_params['lamb_alpha'] = params['lambda_alpha']
    # Ridge regularization weight for sigma
    dd_mpc_params['lamb_sigma'] = params['lambda_sigma']

    # Bounds for the predicted input
    dd_mpc_params['U'] = np.array(params['U'], dtype=float)
    # Bounds for the predicted input setpoint
    dd_mpc_params['Us'] = np.array(params['Us'], dtype=float)

    # Alpha regularization type
    alpha_reg_type_value = params['alpha_reg_type']
    dd_mpc_params['alpha_reg_type'] = (
        AlphaRegTypesMap.get(alpha_reg_type_value, AlphaRegType.APPROXIMATED))
    
    # Nonlinear MPC parameters for alpha_reg_type = 0 (Approximated)
    # Ridge regularization weight for alpha_s
    dd_mpc_params['lamb_alpha_s'] = params['lamb_alpha_s']
    # Ridge regularization weight for sigma_s
    dd_mpc_params['lamb_sigma_s'] = params['lamb_sigma_s']

    # System Output setpoint
    y_r = params['y_r']
    dd_mpc_params['y_r'] = np.array(y_r, dtype=float).reshape(-1, 1)

    # Number of consecutive applications of the optimal input
    # for an n-Step Data-Driven MPC Scheme (multi-step)
    if params['n_n_mpc_step']:
        dd_mpc_params['n_mpc_step'] = n
    else:
        dd_mpc_params['n_mpc_step'] = 1

    # Print Data-Driven MPC controller initialization details
    # based on verbosity level
    if verbose == 1:
        print("Nonlinear Data-Driven MPC controller initialized with loaded "
              "parameters")
    if verbose > 1:
        print("Nonlinear Data-Driven MPC controller initialized with:")
        for key, value in dd_mpc_params.items():
            if key in ['Q', 'R', 'S']:
                # Print scalar and shape for large matrices
                print(f"    {key}: scalar {value[0, 0]} {value.shape}")
            elif key in ['alpha_reg_type']:
                # Print name for enum types
                print(f"    {key}: {value.name}")
            elif key in ['u_range', 'U', 'Us']:
                # Format input bounds and ranges
                formatted_array = ', '.join([f"[{', '.join(map(str, row))}]"
                                             for row in value])
                print(f"    {key}: [{formatted_array}]")
            elif key in ['y_r']:
                # Format setpoint arrays in a single line
                formatted_array = ', '.join([f"[{row[0]}]" for row in value])
                print(f"    {key}: [{formatted_array}]")
            else:
                print(f"    {key}: {value}")

    return dd_mpc_params
