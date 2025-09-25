import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from direct_data_driven_mpc.utilities.visualization import (
    plot_input_output_animation,
    save_animation,
)
from direct_data_driven_mpc.utilities.visualization.control_plot_anim import (
    initialize_data_animation,
    update_data_animation,
)

matplotlib.use("Agg")  # Prevent GUI backend


@pytest.mark.parametrize("continuous_updates", [True, False])
@pytest.mark.parametrize("highlight_initial_steps", [True, False])
@pytest.mark.parametrize("dynamic_setpoint_lines", [True, False])
def test_plot_input_output_animation_return(
    dynamic_setpoint_lines: bool,
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
        dynamic_setpoint_lines=dynamic_setpoint_lines,
        initial_steps=initial_steps,
        continuous_updates=continuous_updates,
    )

    assert isinstance(anim, FuncAnimation)

    plt.close("all")


@pytest.mark.parametrize("continuous_updates", [True, False])
@pytest.mark.parametrize("highlight_initial_steps", [True, False])
@pytest.mark.parametrize("dynamic_setpoint_lines", [True, False])
@pytest.mark.parametrize("plot_bounds", [True, False])
@pytest.mark.parametrize("include_setpoint", [True, False])
def test_initialize_data_animation(
    include_setpoint: bool,
    plot_bounds: bool,
    dynamic_setpoint_lines: bool,
    highlight_initial_steps: bool,
    continuous_updates: bool,
) -> None:
    # Define test parameters
    fig, ax = plt.subplots()
    T = 50
    data = np.sin(np.linspace(0, 2 * np.pi, T))
    setpoint = (
        np.cos(np.linspace(0, 2 * np.pi, T)) if include_setpoint else None
    )
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
        dynamic_setpoint_lines=dynamic_setpoint_lines,
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

    if include_setpoint:
        assert len(setpoint_lines) == (1 if dynamic_setpoint_lines else 0)
    else:
        assert len(setpoint_lines) == 0

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
@pytest.mark.parametrize("include_setpoint", [True, False])
def test_update_data_animation(
    include_setpoint: bool,
    highlight_initial_steps: bool,
    continuous_updates: bool,
) -> None:
    # Define test parameters
    fig, ax = plt.subplots()
    T = 25
    index = 20
    data = np.sin(np.linspace(0, 1, T))
    setpoint = np.cos(np.linspace(0, 1, T)) if include_setpoint else None
    initial_steps = 10 if highlight_initial_steps else None

    # Create dummy plot elements
    data_line = Line2D([], [])
    setpoint_line = Line2D([], []) if include_setpoint else None
    rect = Rectangle((0, 0), 0, 1)
    right_line = Line2D([0], [0])
    left_line = Line2D([0], [0])
    init_text = ax.text(0, 0, "Init")
    control_text = ax.text(0, 0, "Control")

    ax.add_line(data_line)

    if setpoint_line is not None:
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
