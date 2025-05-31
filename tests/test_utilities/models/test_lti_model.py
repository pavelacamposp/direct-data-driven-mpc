from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from direct_data_driven_mpc.utilities.models.lti_model import (
    LTIModel,
    LTISystemModel,
)

LOAD_YAML_PATCH_PATH = (
    "direct_data_driven_mpc.utilities.models.lti_model.load_yaml_config_params"
)


def test_lti_model_simulation(test_linear_system: Any) -> None:
    # Define test parameters
    A, B, C, D, _, _, _ = test_linear_system

    # Instantiate LTI model
    model = LTIModel(A, B, C, D, eps_max=0.0)

    m = model.m  # Number of inputs
    p = model.p  # Number of outputs

    # Test single step simulation
    u = np.ones((m,))
    w = np.zeros((p,))
    expected_x = np.array([0, 1, 2])
    expected_y = np.array([0, 1])

    y = model.simulate_step(u, w)

    np.testing.assert_allclose(model.x, expected_x)
    np.testing.assert_allclose(y, expected_y)

    # Test multiple steps simulation
    n_steps = 5
    U = np.ones((n_steps, m))
    W = np.zeros((n_steps, p))
    expected_x = np.array([1.13164, 1.12084, 1.72342])
    expected_Y = np.array(
        [
            [8, 18],
            [8.4, 20.8],
            [8.3, 20.94],
            [8.514, 21.352],
            [8.5514, 21.4716],
        ]
    )

    Y = model.simulate(U, W, n_steps)

    np.testing.assert_allclose(model.x, expected_x)
    np.testing.assert_allclose(Y, expected_Y)


@pytest.mark.parametrize("valid_state", [True, False])
def test_lti_model_setters(valid_state: bool, test_linear_system: Any) -> None:
    # Define test parameters
    A, B, C, D, _, _, _ = test_linear_system

    # Instantiate LTI model
    model = LTIModel(A, B, C, D, eps_max=0.0)

    # Test model internal state setter
    if valid_state:
        x_set = np.array([0, 0, 0])

        model.set_state(x_set)

        np.testing.assert_allclose(model.x, x_set)

    else:
        x_set = np.array([0])

        # Verify `set_state` raises a `ValueError` when
        # given a state with incorrect dimensions
        with pytest.raises(ValueError):
            model.set_state(x_set)

    # Test system measurement noise setter
    eps_max = 0.01
    model.set_eps_max(eps_max)

    assert model.eps_max == eps_max


def test_lti_model_utility_methods(test_linear_system: Any) -> None:
    # Define test parameters
    A, B, C, D, _, _, _ = test_linear_system

    # Instantiate LTI model
    model = LTIModel(A, B, C, D, eps_max=0.0)

    n = model.n  # System order
    m = model.m  # Number of inputs

    # Test initial state estimation
    expected_estimated_x = np.zeros((n, 1))
    U_n = np.ones((n, m)).reshape(-1, 1)
    Y = np.array([[0, 1], [8, 18], [8.4, 20.8]]).reshape(-1, 1)

    estimated_x = model.get_initial_state_from_trajectory(U_n, Y)

    np.testing.assert_allclose(estimated_x, expected_estimated_x)

    # Test calculation of equilibrium output from input
    test_u_eq = np.ones((m,))
    expected_y_eq = np.array([8.54945055, 21.48351648])

    y_eq = model.get_equilibrium_output_from_input(test_u_eq)

    np.testing.assert_allclose(y_eq, expected_y_eq)

    # Test calculation of equilibrium input from output
    # (Round-trip test using calculated `y_eq`)
    u_eq = model.get_equilibrium_input_from_output(y_eq)

    np.testing.assert_allclose(u_eq, test_u_eq)


@pytest.mark.parametrize("valid_inputs", [True, False])
@patch(LOAD_YAML_PATCH_PATH)
def test_lti_system_model(
    mock_load_yaml: Mock, valid_inputs: bool, test_linear_system: Any
) -> None:
    # Define test parameters
    A, B, C, D, t, _, _ = test_linear_system

    loaded_model_params = {"A": A, "B": B, "C": C, "D": D, "eps_max": 0.01}

    if not valid_inputs:
        loaded_model_params["A"] = np.eye(1)
        loaded_model_params["B"] = np.eye(5)

    # Mock return value of `load_yaml_config_params`
    mock_load_yaml.return_value = loaded_model_params

    # Check for exception or validate correct controller instantiation
    if valid_inputs:
        # Instantiate LTI model
        system_model = LTISystemModel("dummy_path.yaml", "model_key")

        assert system_model.A.shape == A.shape
        assert system_model.eps_max == 0.01
        assert isinstance(system_model, LTIModel)
    else:
        with pytest.raises(ValueError):
            system_model = LTISystemModel("dummy_path.yaml", "model_key")
