from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from direct_data_driven_mpc.utilities.visualization import (
    plot_input_output_comparison,
)
from direct_data_driven_mpc.utilities.visualization.comparison_plot import (
    validate_comparison_plot_parameters,
)

matplotlib.use("Agg")  # Prevent GUI backend


@pytest.mark.parametrize("num_sim", [1, 2])
@pytest.mark.parametrize("include_u_s", [True, False])
def test_plot_input_output_comparison(
    include_u_s: bool,
    num_sim: int,
    dummy_plot_data: tuple[np.ndarray, ...],
) -> None:
    # Define test parameters
    u_k, y_k, u_s, y_s = dummy_plot_data
    _, m = u_k.shape

    u_data = [u_k for _ in range(num_sim)]
    y_data = [y_k for _ in range(num_sim)]

    inputs_line_param_list = [{"linestyle": "-"} for _ in range(num_sim)]
    outputs_line_param_list = [{"linestyle": "--"} for _ in range(num_sim)]
    var_suffix_list = [f"sim{i}" for i in range(num_sim)]
    setpoints_line_params = {"color": "k", "linestyle": ":"}
    legend_params = {"fontsize": 8, "loc": "upper right"}
    title = "Test Comparison Plot"

    # Test control data comparison plot
    try:
        plot_input_output_comparison(
            u_data=u_data,
            y_data=y_data,
            y_s=y_s,
            u_s=u_s if include_u_s else None,
            inputs_line_param_list=inputs_line_param_list,
            outputs_line_param_list=outputs_line_param_list,
            setpoints_line_params=setpoints_line_params,
            var_suffix_list=var_suffix_list,
            legend_params=legend_params,
            title=title,
        )
    except Exception as e:
        pytest.fail(f"`plot_input_output_comparison` raised an exception: {e}")

    # Get current figure and axes
    fig = plt.gcf()
    axs = fig.axes

    # Verify number of lines in input plots
    for ax in axs[:m]:
        for line in ax.get_lines():
            print(line)

        num_lines = len(ax.get_lines())
        expected_lines = num_sim + int(include_u_s)  # Sim data + setpoint

        assert num_lines == expected_lines

    # Verify number of lines in output plots
    for ax in axs[m:]:
        for line in ax.get_lines():
            print(line)

        num_lines = len(ax.get_lines())
        expected_lines = num_sim + 1  # Sim data + setpoint

        assert num_lines == expected_lines

    plt.close(fig)


@pytest.mark.parametrize("num_sim", [1, 2])
def test_plot_input_output_comparison_custom_params_labels(
    num_sim: int,
    dummy_plot_data: tuple[np.ndarray, ...],
) -> None:
    # Define test parameters
    u_k, y_k, u_s, y_s = dummy_plot_data
    _, m = u_k.shape

    u_data = [u_k for _ in range(num_sim)]
    y_data = [y_k for _ in range(num_sim)]

    inputs_line_param_list = [{"linestyle": "-"} for _ in range(num_sim)]
    outputs_line_param_list = [{"linestyle": "--"} for _ in range(num_sim)]
    input_labels = [f"u{i}" for i in range(num_sim)]
    output_labels = [f"y{i}" for i in range(num_sim)]

    plot_input_output_comparison(
        u_data=u_data,
        y_data=y_data,
        y_s=y_s,
        u_s=u_s,
        inputs_line_param_list=inputs_line_param_list,
        outputs_line_param_list=outputs_line_param_list,
        input_labels=input_labels,
        output_labels=output_labels,
    )

    # Get current figure and axes
    fig = plt.gcf()
    axs = fig.axes

    # Verify that custom variable labels are included in the plot
    # Check for inputs
    for ax in axs[:m]:
        legend_texts = [
            text.get_text() for text in ax.get_legend().get_texts()
        ]
        assert all(label in legend_texts for label in input_labels)

    # Check for outputs
    for ax in axs[m:]:
        legend_texts = [
            text.get_text() for text in ax.get_legend().get_texts()
        ]
        assert all(label in legend_texts for label in output_labels)

    plt.close(fig)


@pytest.mark.parametrize(
    "expected_error_match, u_data, y_data",
    [
        # Case 1: Empty input/output data lists
        ("must contain at least", [], []),
        # Case 2: Mismatched number of simulations
        (
            "number of trajectories",
            [np.zeros((5, 2)), np.zeros((5, 2))],
            [np.zeros((5, 2))],
        ),
        # Case 3: Mismatched input array shapes
        (
            "All `u_data` arrays",
            [np.zeros((5, 3)), np.zeros((6, 3))],
            [np.zeros((5, 3)), np.zeros((5, 3))],
        ),
        # Case 3: Mismatched output array shapes
        (
            "All `y_data` arrays",
            [np.zeros((5, 3)), np.zeros((5, 3))],
            [np.zeros((6, 3)), np.zeros((5, 3))],
        ),
    ],
)
def test_validate_comparison_plot_parameters_input_output(
    expected_error_match: str,
    u_data: list[np.ndarray],
    y_data: list[np.ndarray],
) -> None:
    # Verify `ValueError` is raised on mismatched dimensions for
    # input-output data lists
    with pytest.raises(ValueError, match=expected_error_match):
        validate_comparison_plot_parameters(u_data, y_data)


@pytest.mark.parametrize(
    "invalid_parameter",
    [
        "inputs_line_param_list",
        "outputs_line_param_list",
        "var_suffix_list",
        "input_labels",
        "output_labels",
    ],
)
@pytest.mark.parametrize("num_sim", [1, 2])
def test_validate_comparison_plot_parameters_lists(
    num_sim: int,
    invalid_parameter: str,
) -> None:
    # Define test parameters
    u_data = [np.zeros((1, 2)) for _ in range(num_sim)]
    y_data = [np.zeros((1, 2)) for _ in range(num_sim)]

    kwargs: dict[str, Any] = {invalid_parameter: [["test"]] * (num_sim + 1)}

    expected_error_match = rf"{invalid_parameter}.*does not match"

    # Verify `ValueError` is raised on invalid parameter list lengths
    with pytest.raises(ValueError, match=expected_error_match):
        validate_comparison_plot_parameters(u_data, y_data, **kwargs)
