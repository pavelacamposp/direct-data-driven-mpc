import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Rectangle

from direct_data_driven_mpc.utilities.visualization.control_plot import (
    plot_data,
    plot_input_output,
)

matplotlib.use("Agg")  # Prevent GUI backend


@pytest.mark.parametrize("include_u_s", [True, False])
def test_plot_input_output(
    include_u_s: bool, dummy_plot_data: tuple[np.ndarray, ...]
) -> None:
    # Define test parameters
    u_k, y_k, u_s, y_s = dummy_plot_data
    T, m = u_k.shape
    _, p = y_k.shape
    u_bounds_list = [(0.0, 1.0)] * m
    y_bounds_list = [(0.0, 1.0)] * p

    # Test input-output plot
    plot_input_output(
        u_k=u_k,
        y_k=y_k,
        u_s=u_s if include_u_s else None,
        y_s=y_s,
        u_bounds_list=u_bounds_list,
        y_bounds_list=y_bounds_list,
    )

    # Get current figure and axes
    fig = plt.gcf()
    axs = fig.axes

    # Verify correct number of input-output subplots
    assert len(axs) == m + p

    # Verify that input data is correctly plotted
    for j, ax in enumerate(axs[:m]):
        # Check input line data
        lines = ax.get_lines()
        np.testing.assert_equal(np.asarray(lines[0].get_ydata()), u_k[:, j])

        # Check input setpoint line data if included
        if include_u_s:
            expected_u_s = np.full_like(u_k[:, j], u_s[j, 0])
            np.testing.assert_equal(
                np.asarray(lines[1].get_ydata()), expected_u_s
            )

        # Check input constraint lines
        constraint_lines = lines[2:] if include_u_s else lines[1:]
        assert len(constraint_lines) == 2
        for line, bound in zip(
            constraint_lines, u_bounds_list[j], strict=True
        ):
            y_data = line.get_ydata()

            # Check if constraint is plotted as a horizontal line
            assert np.allclose(y_data, [bound, bound])

    # Verify that output data is correctly plotted
    for j, ax in enumerate(axs[m:]):
        # Check input line data
        lines = ax.get_lines()
        np.testing.assert_equal(np.asarray(lines[0].get_ydata()), y_k[:, j])

        # Check output setpoint line data
        expected_y_s = np.full_like(y_k[:, j], y_s[j, 0])
        np.testing.assert_equal(np.asarray(lines[1].get_ydata()), expected_y_s)

        # Check output constraint lines
        constraint_lines = lines[2:]
        assert len(constraint_lines) == 2
        for line, bound in zip(
            constraint_lines, u_bounds_list[j], strict=True
        ):
            y_data = line.get_ydata()

            # Check if constraint is plotted as a horizontal line
            assert np.allclose(y_data, [bound, bound])

    plt.close(fig)


def test_plot_input_output_custom_labels(
    dummy_plot_data: tuple[np.ndarray, ...],
) -> None:
    # Define test parameters
    u_k, y_k, u_s, y_s = dummy_plot_data
    _, m = u_k.shape
    _, p = y_k.shape

    # Define custom labels
    input_label = "Custom Input"
    output_label = "Custom Output"

    u_setpoint_labels = [f"Setpoint U{i}" for i in range(m)]
    y_setpoint_labels = [f"Setpoint Y{i}" for i in range(p)]
    x_axis_labels = [f"Time X-Axis {i}" for i in range(m)]
    input_y_axis_labels = [f"Input Y-Axis {i}" for i in range(m)]
    output_y_axis_labels = [f"Output Y-Axis {i}" for i in range(p)]

    plot_input_output(
        u_k=u_k,
        y_k=y_k,
        u_s=u_s,
        y_s=y_s,
        input_label=input_label,
        output_label=output_label,
        u_setpoint_labels=u_setpoint_labels,
        y_setpoint_labels=y_setpoint_labels,
        input_y_axis_labels=input_y_axis_labels,
        output_y_axis_labels=output_y_axis_labels,
        x_axis_labels=x_axis_labels,
    )

    # Get current figure and axes
    fig = plt.gcf()
    axs = fig.axes

    # Verify that custom variable and axis labels are included in the plot
    # Check for inputs
    for i, ax in enumerate(axs[:m]):
        assert ax.get_xlabel() == x_axis_labels[i]
        assert ax.get_ylabel() == input_y_axis_labels[i]

        legend_texts = [
            text.get_text() for text in ax.get_legend().get_texts()
        ]
        assert input_label in legend_texts
        assert u_setpoint_labels[i] in legend_texts

    # Check for outputs
    for i, ax in enumerate(axs[m:]):
        assert ax.get_xlabel() == x_axis_labels[i]
        assert ax.get_ylabel() == output_y_axis_labels[i]

        legend_texts = [
            text.get_text() for text in ax.get_legend().get_texts()
        ]
        assert output_label in legend_texts
        assert y_setpoint_labels[i] in legend_texts

    plt.close(fig)


@pytest.mark.parametrize("highlight_initial_steps", [True, False])
@pytest.mark.parametrize("plot_bounds", [True, False])
@pytest.mark.parametrize("include_setpoint", [True, False])
def test_plot_data(
    include_setpoint: bool, plot_bounds: bool, highlight_initial_steps: bool
) -> None:
    # Define test parameters
    fig, ax = plt.subplots()
    T = 50
    data = np.linspace(0, 1, T)
    setpoint = np.full(T, [0.7]) if include_setpoint else None
    var_symbol = "u"
    setpoint_var_symbol = "u^s"
    var_suffix = "_test"
    bounds = (0.2, 0.8) if plot_bounds else None
    initial_steps = 10 if highlight_initial_steps else None

    plot_data(
        axis=ax,
        data=data,
        setpoint=setpoint,
        index=0,
        data_line_params={"color": "blue"},
        setpoint_line_params={"color": "green", "linestyle": "--"},
        bounds_line_params={"color": "red", "linestyle": ":"},
        var_symbol=var_symbol,
        setpoint_var_symbol=setpoint_var_symbol,
        var_label="Input",
        var_suffix=var_suffix,
        initial_text="Init",
        control_text="Control",
        display_initial_text=True,
        display_control_text=True,
        fontsize=10,
        legend_params={"fontsize": 8, "loc": "upper right"},
        fig=fig,
        bounds=bounds,
        initial_steps=initial_steps,
        plot_ylimits=(0.0, 1.0),
    )

    # Verify plot lines are correctly plotted
    lines = ax.get_lines()

    # Expected lines: data, setpoint, 2 x bounds, initial steps line
    expected_lines = (
        1
        + int(include_setpoint)
        + 2 * int(plot_bounds)
        + int(highlight_initial_steps)
    )

    assert len(lines) == expected_lines

    # Verify data is correctly plotted
    line_data_label_tuples: list[tuple[np.ndarray, str]]
    if include_setpoint:
        line_data_label_tuples = [
            (data, f"${var_symbol}_1${var_suffix}"),
            (np.full(T, setpoint), f"${setpoint_var_symbol}_1$"),
        ]
    else:
        line_data_label_tuples = [(data, f"${var_symbol}_1${var_suffix}")]

    for line_data, line_label in line_data_label_tuples:
        data_line = next(
            (line for line in lines if line.get_label() == line_label), None
        )

        assert data_line is not None

        x_data, y_data = data_line.get_data()
        np.testing.assert_equal(x_data, np.arange(T))
        np.testing.assert_equal(y_data, line_data)

    # Check legend
    legend = ax.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert f"${var_symbol}_1${var_suffix}" in labels

    if include_setpoint:
        assert f"${setpoint_var_symbol}_1$" in labels

    if plot_bounds:
        assert "Constraints" in labels

    # Check initial steps highlighting elements if enabled
    if highlight_initial_steps:
        # Check rectangle (axvspan) for initial steps
        rects = ax.patches
        assert len(rects) > 0

        assert isinstance(rects[0], Rectangle)  # Ensure mypy infers rect type
        rect = rects[0]
        assert rect.get_x() == 0
        assert rect.get_width() == initial_steps

        # Check vertical dashed line (axvline)
        for line in lines:
            pattern = getattr(line, "_dash_pattern", None)
            if pattern is not None:
                if pattern == (0.0, [5.0, 5.0]):
                    assert np.asarray(line.get_xdata())[0] == initial_steps

    plt.close(fig)
