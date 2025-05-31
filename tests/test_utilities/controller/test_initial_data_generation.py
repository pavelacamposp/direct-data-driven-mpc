from typing import Any

import numpy as np

from direct_data_driven_mpc.utilities.controller.initial_data_generation import (  # noqa: E501
    generate_initial_input_output_data,
    randomize_initial_system_state,
    simulate_n_input_output_measurements,
)
from direct_data_driven_mpc.utilities.models.lti_model import (
    LTIModel,
)


def test_randomize_initial_system_state(mock_lti_model: LTIModel) -> None:
    # Define test parameters
    n = mock_lti_model.n  # System order
    m = mock_lti_model.m  # Number of inputs
    controller_config: Any = {"u_range": np.array([[-1, 1]] * m)}
    np_random = np.random.default_rng(0)

    x0 = randomize_initial_system_state(
        mock_lti_model, controller_config, np_random
    )

    # Verify system state has correct dimensions
    assert x0.shape == (n,)


def test_generate_initial_input_output_data(mock_lti_model: LTIModel) -> None:
    # Define test parameters
    m = mock_lti_model.m  # Number of inputs
    p = mock_lti_model.p  # Number of outputs
    N = 5  # Initial data trajectory length
    controller_config: Any = {"N": N, "u_range": np.array([[-1, 1]] * m)}
    np_random = np.random.default_rng(0)

    u_d, y_d = generate_initial_input_output_data(
        mock_lti_model, controller_config, np_random
    )

    # Verify correct input-output data dimensions
    assert u_d.shape == (N, m)
    assert y_d.shape == (N, p)

    # Verify input is correctly bounded
    assert np.all(u_d >= -1.0) and np.all(u_d <= 1.0)


def test_simulate_n_input_output_measurements(
    mock_lti_model: LTIModel,
) -> None:
    # Define test parameters
    m = mock_lti_model.m  # Number of inputs
    p = mock_lti_model.p  # Number of outputs
    n = 4  # Estimated system order
    controller_config: Any = {"n": n, "u_s": np.zeros((m, 1))}
    np_random = np.random.default_rng(0)

    U_n, Y_n = simulate_n_input_output_measurements(
        mock_lti_model, controller_config, np_random
    )

    # Verify correct input-output data dimensions
    assert U_n.shape == (n, m)
    assert Y_n.shape == (n, p)
