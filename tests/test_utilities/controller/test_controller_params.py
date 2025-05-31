import copy
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from direct_data_driven_mpc.lti_data_driven_mpc_controller import (
    LTIDataDrivenMPCType,
    SlackVarConstraintType,
)
from direct_data_driven_mpc.nonlinear_data_driven_mpc_controller import (
    AlphaRegType,
)
from direct_data_driven_mpc.utilities.controller.controller_params import (
    LTIDataDrivenMPCParams,
    NonlinearDataDrivenMPCParams,
    construct_weighting_matrix,
    get_lti_data_driven_mpc_controller_params,
    get_nonlinear_data_driven_mpc_controller_params,
    get_weights_list_from_param,
    print_parameter_loading_details,
)

LOAD_YAML_PATCH_PATH = (
    "direct_data_driven_mpc.utilities.controller.controller_params."
    "load_yaml_config_params"
)


@pytest.mark.parametrize("scalar_weights", [True, False])
@patch(LOAD_YAML_PATCH_PATH)
def test_get_lti_data_driven_mpc_controller_params(
    mock_load_yaml: Mock,
    scalar_weights: bool,
    test_dd_mpc_controller_yaml_config: dict[str, Any],
) -> None:
    # Define test parameters
    m = 2  # Number of inputs
    p = 1  # Number of outputs
    loaded_params = copy.deepcopy(test_dd_mpc_controller_yaml_config)

    if scalar_weights:
        loaded_params["Q_weights"] = 1.0
        loaded_params["R_weights"] = 0.1
    else:
        loaded_params["Q_weights"] = [1.0] * p
        loaded_params["R_weights"] = [0.1] * m

    loaded_params["U"] = [[0, 1]] * m
    loaded_params["u_s"] = [0.1] * m
    loaded_params["y_s"] = [1.0] * p

    # Mock return value of `load_yaml_config_params`
    mock_load_yaml.return_value = loaded_params

    # Load controller parameters
    params = get_lti_data_driven_mpc_controller_params(
        config_file="dummy.yaml",
        controller_key_value="controller_key",
        m=m,
        p=p,
    )

    # Validate correct parameter initialization
    L = loaded_params["L"]
    assert isinstance(params["Q"], np.ndarray)
    assert isinstance(params["R"], np.ndarray)
    assert params["Q"].shape[0] == p * L
    assert params["R"].shape[0] == m * L
    assert params["controller_type"] == LTIDataDrivenMPCType.NOMINAL
    assert params["slack_var_constraint_type"] == SlackVarConstraintType.CONVEX


@pytest.mark.parametrize("ext_out_incr_in", [True, False])
@pytest.mark.parametrize("scalar_weights", [True, False])
@patch(LOAD_YAML_PATCH_PATH)
def test_get_nonlinear_data_driven_mpc_controller_params(
    mock_load_yaml: Mock,
    scalar_weights: bool,
    ext_out_incr_in: bool,
    test_dd_mpc_controller_yaml_config: dict[str, Any],
) -> None:
    # Define test parameters
    m = 2  # Number of inputs
    p = 1  # Number of outputs
    loaded_params = copy.deepcopy(test_dd_mpc_controller_yaml_config)

    if scalar_weights:
        loaded_params["Q_weights"] = 1.0
        loaded_params["R_weights"] = 0.1
        loaded_params["S_weights"] = 0.01
    else:
        loaded_params["Q_weights"] = [1.0] * p
        loaded_params["R_weights"] = [0.1] * m
        loaded_params["S_weights"] = [0.01] * p

    loaded_params["U"] = [[0, 1]] * m
    loaded_params["y_r"] = [1.0] * p
    loaded_params["ext_out_incr_in"] = ext_out_incr_in

    # Mock return value of `load_yaml_config_params`
    mock_load_yaml.return_value = loaded_params

    # Load controller parameters
    params = get_nonlinear_data_driven_mpc_controller_params(
        config_file="dummy.yaml",
        controller_key_value="controller_key",
        m=m,
        p=p,
    )

    # Validate correct parameter initialization
    n = loaded_params["n"]
    L = loaded_params["L"]
    assert isinstance(params["Q"], np.ndarray)
    assert isinstance(params["R"], np.ndarray)
    assert isinstance(params["S"], np.ndarray)

    if ext_out_incr_in:
        assert params["Q"].shape[0] == (m + p) * (L + n + 1)
    else:
        assert params["Q"].shape[0] == p * (L + n + 1)

    assert params["R"].shape[0] == m * (L + n + 1)
    assert params["S"].shape[0] == p

    assert params["alpha_reg_type"] == AlphaRegType.APPROXIMATED


@patch(LOAD_YAML_PATCH_PATH)
def test_lti_param_missing_key_raises(mock_load_yaml: Mock) -> None:
    # Missing required keys in config should raise `ValueError`
    mock_load_yaml.return_value = {"N": 100}

    with pytest.raises(ValueError, match="Missing required parameter key"):
        get_lti_data_driven_mpc_controller_params(
            "dummy.yaml", "controller_key", m=1, p=1
        )


@patch(LOAD_YAML_PATCH_PATH)
def test_nonlinear_param_missing_key_raises(mock_load_yaml: Mock) -> None:
    # Missing required keys in config should raise `ValueError`
    mock_load_yaml.return_value = {"N": 100}

    with pytest.raises(ValueError, match="Missing required parameter key"):
        get_nonlinear_data_driven_mpc_controller_params(
            "dummy.yaml", "controller_key", m=1, p=1
        )


@pytest.mark.parametrize(
    "weights_param, n_vars, horizon, expected_matrix",
    [
        (2.0, 3, 2, np.kron(np.eye(2), np.diag([2.0, 2.0, 2.0]))),
        ([1.0, 2.0], 2, 3, np.kron(np.eye(3), np.diag([1.0, 2.0]))),
    ],
)
def test_construct_weighting_matrix_valid(
    weights_param: float | list[float],
    n_vars: int,
    horizon: int,
    expected_matrix: np.ndarray,
) -> None:
    matrix = construct_weighting_matrix(weights_param, n_vars, horizon)

    np.testing.assert_array_equal(matrix, expected_matrix)


@pytest.mark.parametrize(
    "weights_param, n_vars, expected_message",
    [
        ([1.0, 2.0], 3, "Expected a list of length 3"),
        ("invalid_weight", 2, "Expected a scalar or a list of length 2"),
    ],
)
def test_construct_weighting_matrix_invalid(
    weights_param: Any, n_vars: int, expected_message: str
) -> None:
    # Verify a `ValueError` is raised with invalid parameters
    with pytest.raises(ValueError, match=expected_message):
        construct_weighting_matrix(weights_param, n_vars, 2)


@pytest.mark.parametrize(
    "weights_param, size, expected",
    [
        (1.0, 3, [1.0, 1.0, 1.0]),
        ([0.5, 1.5], 2, [0.5, 1.5]),
    ],
)
def test_get_weights_list_from_param_valid(
    weights_param: float | list[float], size: int, expected: list[float]
) -> None:
    result = get_weights_list_from_param(weights_param, size)

    assert result == expected


@pytest.mark.parametrize(
    "weights_param, size, expected_message",
    [
        ([1.0], 2, "Expected a scalar or a list of length 2"),
        ("invalid_weight", 2, "Expected a scalar or a list of length 2"),
    ],
)
def test_get_weights_list_from_param_invalid(
    weights_param: Any, size: int, expected_message: str
) -> None:
    # Verify a `ValueError` is raised with invalid parameters
    with pytest.raises(ValueError, match=expected_message):
        get_weights_list_from_param(weights_param, size)


@pytest.mark.parametrize("verbosity_level", [0, 1, 2])
@pytest.mark.parametrize("dd_mpc_type", ["lti", "nonlinear"])
def test_print_parameter_loading_details(
    dd_mpc_type: str,
    verbosity_level: int,
    dummy_lti_controller_data: tuple[
        LTIDataDrivenMPCParams, np.ndarray, np.ndarray
    ],
    dummy_nonlinear_controller_data: tuple[
        NonlinearDataDrivenMPCParams, np.ndarray, np.ndarray
    ],
    capsys: pytest.CaptureFixture,
) -> None:
    # Define test parameters
    controller_params: LTIDataDrivenMPCParams | NonlinearDataDrivenMPCParams
    if dd_mpc_type == "lti":
        controller_params, _, _ = dummy_lti_controller_data
        cost_horizon = controller_params["L"]
        controller_type = controller_params["controller_type"].name
        controller_label = "LTI"
    else:
        controller_params, _, _ = dummy_nonlinear_controller_data
        cost_horizon = controller_params["L"] + controller_params["n"] + 1
        controller_label = "Nonlinear"
        alpha_reg_type = controller_params["alpha_reg_type"].name

    print_parameter_loading_details(
        controller_params, cost_horizon, verbosity_level, controller_label
    )

    # Capture printed output
    out, _ = capsys.readouterr()

    # Verify print output based on the verbosity level
    if verbosity_level == 0:
        assert out == ""
    elif verbosity_level == 1:
        assert (
            f"Loaded {controller_label} Data-Driven MPC controller "
            "parameters\n" in out
        )
    else:
        assert (
            f"Loaded {controller_label} Data-Driven MPC controller parameters:"
            in out
        )
        assert "Q weights:" in out

        if dd_mpc_type == "lti":
            assert f"controller_type: {controller_type}" in out
        else:
            assert f"alpha_reg_type: {alpha_reg_type}" in out
