from unittest.mock import Mock, patch

import numpy as np
import pytest

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCController,
)
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    NonlinearDataDrivenMPCController,
)
from direct_data_driven_mpc.utilities.controller.controller_creation import (
    LTIDataDrivenMPCParams,
    NonlinearDataDrivenMPCParams,
)


@pytest.fixture
@patch.object(LTIDataDrivenMPCController, "initialize_data_driven_mpc")
def dummy_lti_controller(
    _: Mock,
    dummy_lti_controller_data: tuple[
        LTIDataDrivenMPCParams, np.ndarray, np.ndarray
    ],
) -> LTIDataDrivenMPCController:
    controller_params, u_d, y_d = dummy_lti_controller_data
    m = u_d.shape[1]
    p = y_d.shape[1]

    dummy_lti_controller = LTIDataDrivenMPCController(
        n=controller_params["n"],
        m=m,
        p=p,
        u_d=u_d,
        y_d=y_d,
        L=controller_params["L"],
        Q=controller_params["Q"],
        R=controller_params["R"],
        u_s=controller_params["u_s"],
        y_s=controller_params["y_s"],
        eps_max=controller_params["eps_max"],
        lamb_alpha=controller_params["lamb_alpha"],
        lamb_sigma=controller_params["lamb_sigma"],
        U=controller_params["U"],
        c=controller_params["c"],
        slack_var_constraint_type=(
            controller_params["slack_var_constraint_type"]
        ),
        controller_type=controller_params["controller_type"],
        n_mpc_step=controller_params["n_mpc_step"],
        use_terminal_constraints=True,
    )

    return dummy_lti_controller


@pytest.fixture
@patch.object(LTIDataDrivenMPCController, "initialize_data_driven_mpc")
def dummy_nonlinear_controller(
    _: Mock,
    dummy_nonlinear_controller_data: tuple[
        NonlinearDataDrivenMPCParams, np.ndarray, np.ndarray
    ],
) -> NonlinearDataDrivenMPCController:
    controller_params, u, y = dummy_nonlinear_controller_data
    m = u.shape[1]
    p = y.shape[1]

    dummy_nonlinear_controller = NonlinearDataDrivenMPCController(
        n=controller_params["n"],
        m=m,
        p=p,
        u=u,
        y=y,
        L=controller_params["L"],
        Q=controller_params["Q"],
        R=controller_params["R"],
        S=controller_params["S"],
        y_r=controller_params["y_r"],
        lamb_alpha=controller_params["lamb_alpha"],
        lamb_sigma=controller_params["lamb_sigma"],
        U=controller_params["U"],
        Us=controller_params["Us"],
        alpha_reg_type=controller_params["alpha_reg_type"],
        lamb_alpha_s=controller_params["lamb_alpha_s"],
        lamb_sigma_s=controller_params["lamb_sigma_s"],
        ext_out_incr_in=controller_params["ext_out_incr_in"],
        update_cost_threshold=controller_params["update_cost_threshold"],
        n_mpc_step=controller_params["n_mpc_step"],
    )

    return dummy_nonlinear_controller
