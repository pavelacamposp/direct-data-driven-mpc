from typing import Any

import numpy as np
import pytest

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCType,
    SlackVarConstraintType,
)
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    AlphaRegType,
)


@pytest.fixture
def test_dd_mpc_controller_yaml_config() -> dict[str, Any]:
    return {
        "n": 2,
        "N": 50,
        "L": 4,
        "Q_weights": 1.0,
        "R_weights": 0.1,
        "epsilon_bar": 0.01,
        "lambda_alpha": 0.1,
        "lambda_sigma": 0.1,
        "lambda_alpha_epsilon_bar": 0.001,
        "U": [[-1, 1]],
        "Us": [[-1, 1]],
        "u_d_range": [[-1, 1]],
        "u_range": [[-1, 1]],
        "slack_var_constraint_type": 1,
        "controller_type": 0,
        "u_s": [0.0],
        "y_s": [0.0],
        "y_r": [0.0],
        "n_n_mpc_step": True,
        "alpha_reg_type": 0,
        "lambda_alpha_s": 0.01,
        "lambda_sigma_s": 0.01,
        "ext_out_incr_in": False,
        "update_cost_threshold": 0.0,
    }


@pytest.fixture
def test_dd_mpc_controller_config() -> dict[str, Any]:
    return {
        "n": 2,
        "L": 4,
        "Q": np.eye(1),
        "R": np.eye(1),
        "S": np.eye(1),
        "u_s": np.zeros((1,)),
        "y_s": np.zeros((1,)),
        "y_r": np.zeros((1,)),
        "eps_max": 0.1,
        "lamb_alpha": 0.01,
        "lamb_sigma": 0.01,
        "U": np.zeros((2, 2)),
        "Us": np.zeros((2, 2)),
        "c": 1.0,
        "slack_var_constraint_type": SlackVarConstraintType.CONVEX,
        "controller_type": LTIDataDrivenMPCType.ROBUST,
        "n_mpc_step": 1,
        "alpha_reg_type": AlphaRegType.APPROXIMATED,
        "lamb_alpha_s": 0.01,
        "lamb_sigma_s": 0.01,
        "ext_out_incr_in": False,
        "update_cost_threshold": 0.0,
    }
