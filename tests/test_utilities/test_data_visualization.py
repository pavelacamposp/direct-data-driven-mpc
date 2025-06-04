import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from direct_data_driven_mpc.utilities.data_visualization import (
    HandlerInitMeasurementRect,
    create_input_output_figure,
    initialize_data_animation,
    plot_data,
    plot_input_output,
    plot_input_output_animation,
    save_animation,
    update_data_animation,
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

    # Test input-output plot
    try:
        if include_u_s:
            plot_input_output(u_k=u_k, y_k=y_k, u_s=u_s, y_s=y_s)
        else:
            plot_input_output(u_k=u_k, y_k=y_k, y_s=y_s)

    except Exception as e:
        pytest.fail(
            f"`plot_input_output` (`include_u_s` = {include_u_s}) raised an "
            f"exception: {e}"
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

    # Verify that output data is correctly plotted
    for j, ax in enumerate(axs[m:]):
        # Check input line data
        lines = ax.get_lines()
        np.testing.assert_equal(np.asarray(lines[0].get_ydata()), y_k[:, j])

        # Check input setpoint line data if included
        expected_y_s = np.full_like(y_k[:, j], y_s[j, 0])
        np.testing.assert_equal(np.asarray(lines[1].get_ydata()), expected_y_s)

    plt.close("all")


def test_plot_input_output_mismatched_dim_len(
    dummy_plot_data: tuple[np.ndarray, ...],
) -> None:
    u_k, y_k, u_s, y_s = dummy_plot_data
    m = u_k.shape[1]
    p = y_k.shape[1]

    # Verify `ValueError` is raised on dimension
    # mismatch between input-output arrays
    with pytest.raises(ValueError, match="Dimension mismatch"):
        plot_input_output(u_k=u_k[:-1], y_k=y_k, y_s=y_s)

    # Verify `ValueError` is raised on dimension
    # mismatch between setpoint arrays
    with pytest.raises(ValueError, match="Dimension mismatch"):
        plot_input_output(u_k=u_k, y_k=y_k, y_s=y_s[:1])

    with pytest.raises(ValueError, match="Dimension mismatch"):
        plot_input_output(u_k=u_k, y_k=y_k, u_s=u_s[:1], y_s=y_s)

    # Verify `ValueError` is raised when bounds or
    # y-limits list lengths mismatch
    mismatch_cases = [
        ("u_bounds_list", {"u_bounds_list": [(0.0, 1.0)] * (m + 1)}),
        ("y_bounds_list", {"y_bounds_list": [(0.0, 1.0)] * (p + 1)}),
        ("u_ylimits_list", {"u_ylimits_list": [(0.0, 1.0)] * (m + 1)}),
        ("y_ylimits_list", {"y_ylimits_list": [(0.0, 1.0)] * (p + 1)}),
    ]

    for param_name, kwargs in mismatch_cases:
        with pytest.raises(ValueError, match=f"{param_name}.*does not match"):
            # Prevent mypy [arg-type] error
            safe_kwargs: dict[str, Any] = kwargs

            plot_input_output(
                u_k=u_k,
                y_k=y_k,
                u_s=u_s,
                y_s=y_s,
                **safe_kwargs,
            )

    plt.close("all")


@pytest.mark.parametrize("highlight_initial_steps", [True, False])
@pytest.mark.parametrize("plot_bounds", [True, False])
def test_plot_data(plot_bounds: bool, highlight_initial_steps: bool) -> None:
    # Define test parameters
    fig, ax = plt.subplots()
    T = 50
    data = np.linspace(0, 1, T)
    setpoint = np.full(T, [0.7])
    var_symbol = "u"
    setpoint_var_symbol = "u^s"
    data_label = "_test"
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
        data_label=data_label,
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
    expected_lines = 2 + 2 * int(plot_bounds) + int(highlight_initial_steps)
    assert len(lines) == expected_lines

    # Verify data is correctly plotted
    line_data_label_tuples: list[tuple[np.ndarray, str]] = [
        (data, f"${var_symbol}_1${data_label}"),
        (np.full(T, setpoint), f"${setpoint_var_symbol}_1$"),
    ]

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
    assert f"${var_symbol}_1${data_label}" in labels
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


@pytest.mark.parametrize("continuous_updates", [True, False])
@pytest.mark.parametrize("highlight_initial_steps", [True, False])
def test_plot_input_output_animation_return(
    highlight_initial_steps: bool,
    continuous_updates: bool,
    dummy_plot_data: tuple[np.ndarray, ...],
) -> None:
    # Define test parameters
    u_k, y_k, u_s, y_s = dummy_plot_data
    initial_steps = 10 if highlight_initial_steps else None

    anim = plot_input_output_animation(
        u_k=u_k,
        y_k=y_k,
        u_s=u_s,
        y_s=y_s,
        initial_steps=initial_steps,
        continuous_updates=continuous_updates,
    )

    assert isinstance(anim, FuncAnimation)

    plt.close("all")


@pytest.mark.parametrize("continuous_updates", [True, False])
@pytest.mark.parametrize("highlight_initial_steps", [True, False])
@pytest.mark.parametrize("plot_bounds", [True, False])
def test_initialize_data_animation(
    plot_bounds: bool,
    highlight_initial_steps: bool,
    continuous_updates: bool,
) -> None:
    # Define test parameters
    fig, ax = plt.subplots()
    T = 50
    data = np.sin(np.linspace(0, 2 * np.pi, T))
    setpoint = np.cos(np.linspace(0, 2 * np.pi, T))
    bounds = (0.2, 0.8) if plot_bounds else None
    initial_steps = 10 if highlight_initial_steps else None

    # Initialize plot element storage lists
    data_lines: list[Line2D] = []
    setpoint_lines: list[Line2D] = []
    rects: list[Rectangle] = []
    right_lines: list[Line2D] = []
    left_lines: list[Line2D] = []
    init_texts: list[Text] = []
    control_texts: list[Text] = []
    y_centers: list[float] = []

    # Test animation plot element initialization
    initialize_data_animation(
        axis=ax,
        data=data,
        setpoint=setpoint,
        index=0,
        data_line_params={"color": "blue"},
        setpoint_line_params={"color": "green"},
        bounds_line_params={"color": "red"},
        var_symbol="u",
        setpoint_var_symbol="u^s",
        var_label="Input",
        initial_text="Init",
        control_text="Control",
        fontsize=10,
        legend_params={},
        data_lines=data_lines,
        setpoint_lines=setpoint_lines,
        rects=rects,
        right_rect_lines=right_lines,
        left_rect_lines=left_lines,
        init_texts=init_texts,
        control_texts=control_texts,
        y_axis_centers=y_centers,
        bounds=bounds,
        initial_steps=initial_steps,
        initial_steps_label="Init period",
        continuous_updates=continuous_updates,
    )

    # Verify plot objects are correctly created and stored in lists
    assert len(data_lines) == 1
    assert len(setpoint_lines) == 1
    assert len(y_centers) == 1

    if initial_steps:
        assert len(rects) == 1
        assert len(right_lines) == 1
        assert len(init_texts) == 1
        assert len(control_texts) == 1

        if continuous_updates:
            assert len(left_lines) == 1
        else:
            assert len(left_lines) == 0
    else:
        assert len(rects) == 0
        assert len(right_lines) == 0
        assert len(init_texts) == 0
        assert len(control_texts) == 0
        assert len(left_lines) == 0

    plt.close(fig)


@pytest.mark.parametrize("continuous_updates", [True, False])
@pytest.mark.parametrize("highlight_initial_steps", [True, False])
def test_update_data_animation(
    highlight_initial_steps: bool,
    continuous_updates: bool,
) -> None:
    # Define test parameters
    fig, ax = plt.subplots()
    T = 25
    index = 20
    data = np.sin(np.linspace(0, 1, T))
    setpoint = np.cos(np.linspace(0, 1, T))
    initial_steps = 10 if highlight_initial_steps else None

    # Create dummy plot elements
    data_line = Line2D([], [])
    setpoint_line = Line2D([], [])
    rect = Rectangle((0, 0), 0, 1)
    right_line = Line2D([0], [0])
    left_line = Line2D([0], [0])
    init_text = ax.text(0, 0, "Init")
    control_text = ax.text(0, 0, "Control")

    ax.add_line(data_line)
    ax.add_line(setpoint_line)
    ax.add_patch(rect)
    ax.add_line(right_line)
    ax.add_line(left_line)

    # Test data animation update
    update_data_animation(
        index=index,
        data=data,
        setpoint=setpoint,
        data_length=T,
        points_per_frame=1,
        initial_steps=initial_steps,
        continuous_updates=continuous_updates,
        data_line=data_line,
        setpoint_line=setpoint_line,
        rect=rect,
        y_axis_center=0.5,
        right_rect_line=right_line,
        left_rect_line=left_line,
        init_text_obj=init_text,
        control_text_obj=control_text,
        display_initial_text=True,
        display_control_text=True,
        init_text_width=5,
        control_text_width=5,
    )

    # Verify updated line data
    x_line, y_line = data_line.get_data()
    assert len(np.asarray(x_line)) == index + 1
    np.testing.assert_equal(np.asarray(y_line), data[: index + 1])

    if initial_steps:
        # Check rectangle updated position and width
        if continuous_updates:
            assert rect.get_width() == initial_steps
            assert rect.get_x() == index - initial_steps
        else:
            assert rect.get_width() == 0
            assert rect.get_x() == 0

        # Check if boundary lines are at their expected position
        if continuous_updates:
            assert np.asarray(right_line.get_xdata())[0] == index
            assert (
                np.asarray(left_line.get_xdata())[0] == index - initial_steps
            )
        else:
            assert np.asarray(left_line.get_xdata())[0] == 0
            assert np.asarray(right_line.get_xdata())[0] == 0

        # Check label visibility
        assert init_text.get_visible()
        assert control_text.get_visible()

    plt.close(fig)


def test_save_animation(tmp_path: Path) -> None:
    # Define test parameters
    fig, _ = plt.subplots()

    def dummy_update(frame: int) -> list:
        return []

    anim = FuncAnimation(fig, dummy_update, frames=10)
    file_path = os.path.join(tmp_path, "anim.gif")

    # Test save animation
    save_animation(
        anim, total_frames=10, fps=30, bitrate=2000, file_path=file_path
    )

    # Assert file exists (animation file was created)
    assert os.path.isfile(file_path)

    plt.close(fig)


def test_create_input_output_figure() -> None:
    # Define test parameters
    m = 2
    p = 3

    fig, axs_u, axs_y = create_input_output_figure(
        m, p, figsize=(10, 6), dpi=100, fontsize=12, title="Test Fig"
    )

    # Ensure figure is created with correct number of subplots
    assert isinstance(fig, Figure)
    assert len(axs_u) == m
    assert len(axs_y) == p
    for ax in list(axs_u) + list(axs_y):
        assert hasattr(ax, "plot")

    plt.close(fig)


def test_custom_legend_handler() -> None:
    # Define test parameters
    handler = HandlerInitMeasurementRect()
    fig, ax = plt.subplots()
    dummy_legend = Legend(ax, handles=[], labels=[])
    dummy_rect = Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.2)

    legend_artists = handler.create_artists(
        legend=dummy_legend,
        orig_handle=dummy_rect,
        xdescent=0,
        ydescent=0,
        width=1,
        height=1,
        fontsize=10,
        trans=plt.gca().transData,
    )

    assert any(isinstance(artist, Rectangle) for artist in legend_artists)

    plt.close(fig)
