import math
import os
from typing import Any

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from tqdm import tqdm

from .plot_utilities import (
    HandlerInitMeasurementRect,
    create_input_output_figure,
    get_padded_limits,
    get_text_width_in_data,
    init_dict_if_none,
    validate_data_dimensions,
)


def plot_input_output_animation(
    u_k: np.ndarray,
    y_k: np.ndarray,
    y_s: np.ndarray,
    u_s: np.ndarray | None = None,
    u_bounds_list: list[tuple[float, float]] | None = None,
    y_bounds_list: list[tuple[float, float]] | None = None,
    inputs_line_params: dict[str, Any] | None = None,
    outputs_line_params: dict[str, Any] | None = None,
    setpoints_line_params: dict[str, Any] | None = None,
    bounds_line_params: dict[str, Any] | None = None,
    dynamic_setpoint_lines: bool = False,
    u_setpoint_var_symbol: str = "u^s",
    y_setpoint_var_symbol: str = "y^s",
    initial_steps: int | None = None,
    initial_steps_label: str | None = None,
    continuous_updates: bool = False,
    initial_excitation_text: str = "Init. Excitation",
    initial_measurement_text: str = "Init. Measurement",
    control_text: str = "Data-Driven MPC",
    display_initial_text: bool = True,
    display_control_text: bool = True,
    figsize: tuple[float, float] = (12.0, 8.0),
    dpi: int = 300,
    interval: float = 20.0,
    points_per_frame: int = 1,
    fontsize: int = 12,
    legend_params: dict[str, Any] | None = None,
    title: str | None = None,
) -> FuncAnimation:
    """
    Create a Matplotlib animation showing the progression of input-output data
    over time.

    This function generates a figure with two rows of subplots: the top
    subplots display control inputs and the bottom subplots display system
    outputs. Each subplot shows the data series for each sequence alongside
    its setpoint as a constant line. The appearance of plot lines and legends
    can be customized by passing dictionaries of Matplotlib line and legend
    properties.

    The number of data points shown in each animation frame and the animation
    speed can be configured via the `points_per_frame` and `interval`
    parameters, respectively. These parameters allow control over the speed
    at which data is shown in the animation, as well as the total number of
    animation frames required to display all the data.

    If provided, the first 'initial_steps' time steps can be highlighted to
    emphasize the initial input-output data measurement period representing
    the data-driven system characterization phase in a Data-Driven MPC
    algorithm. Additionally, custom labels can be displayed to indicate the
    initial measurement and the subsequent MPC control periods, but only if
    there is enough space to prevent them from overlapping with other plot
    elements.

    Args:
        u_k (np.ndarray): An array containing control input data of shape (T,
            m), where `m` is the number of inputs and `T` is the number of
            time steps.
        y_k (np.ndarray): An array containing system output data of shape (T,
            p), where `p` is the number of outputs and `T` is the number of
            time steps.
        y_s (np.ndarray): An array containing output setpoint values of shape
            (T, p), where `p` is the number of outputs and `T` is the number of
            time steps.
        u_s (np.ndarray | None): An array containing input setpoint values of
            shape (T, m), where `m` is the number of inputs and `T` is the
            number of time steps. If `None`, input setpoint lines will not be
            plotted. Defaults to `None`.
        u_bounds_list (list[tuple[float, float]] | None): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each input data
            sequence. If provided, horizontal lines representing these bounds
            will be plotted in each subplot. If `None`, no horizontal lines
            will be plotted. The number of tuples must match the number of
            input data sequences. Defaults to `None`.
        y_bounds_list (list[tuple[float, float]] | None): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each output data
            sequence. If provided, horizontal lines representing these bounds
            will be plotted in each subplot. If `None`, no horizontal lines
            will be plotted. The number of tuples must match the number of
            output data sequences. Defaults to `None`.
        inputs_line_params (dict[str, Any] | None): A dictionary of Matplotlib
            properties for customizing the lines used to plot the input data
            series (e.g., color, linestyle, linewidth). If not provided,
            Matplotlib's default line properties will be used.
        outputs_line_params (dict[str, Any] | None): A dictionary of Matplotlib
            properties for customizing the lines used to plot the output data
            series (e.g., color, linestyle, linewidth). If not provided,
            Matplotlib's default line properties will be used.
        setpoints_line_params (dict[str, Any] | None): A dictionary of
            Matplotlib properties for customizing the lines used to plot the
            setpoint values (e.g., color, linestyle, linewidth). If not
            provided, Matplotlib's default line properties will be used.
        bounds_line_params (dict[str, Any] | None): A dictionary of Matplotlib
            properties for customizing the lines used to plot the bounds of
            input-output data series (e.g., color, linestyle, linewidth). If
            not provided, Matplotlib's default line properties will be used.
        dynamic_setpoint_lines (bool): Whether to update setpoint lines
            dynamically. If `False`, they will be fully plotted at the start
            using all setpoint data.
        u_setpoint_var_symbol (str): The variable symbol used to label the
            input setpoint data series (e.g., "u^s").
        y_setpoint_var_symbol (str): The variable symbol used to label the
            output setpoint data series (e.g., "y^s").
        initial_steps (int | None): The number of initial time steps during
            which input-output measurements were taken for the data-driven
            characterization of the system. This highlights the initial
            measurement period in the plot. If `None`, no special highlighting
            will be applied. Defaults to `None`.
        initial_steps_label (str | None): Label text to use for the legend
            entry representing the initial input-output measurement highlight
            in the plot. If `None`, this element will not appear in the
            legend. Defaults to `None`.
        continuous_updates (bool): Whether the initial measurement period
            highlight should move with the latest data to represent continuous
            input-output measurement updates. Defaults to `False`.
        initial_excitation_text (str): Label text to display over the initial
            excitation period of the input plots. Default is
            "Init. Excitation".
        initial_measurement_text (str): Label text to display over the initial
            measurement period of the output plots. Default is
            "Init. Measurement".
        control_text (str): Label text to display over the post-initial
            control period. Default is "Data-Driven MPC".
        display_initial_text (bool): Whether to display the `initial_text`
            label on the plot. Default is True.
        display_control_text (bool): Whether to display the `control_text`
            label on the plot. Default is True.
        figsize (tuple[float, float]): The (width, height) dimensions of the
            created Matplotlib figure.
        dpi (int): The DPI resolution of the figure.
        interval (float): The time between frames in milliseconds. Defaults
            to 20 ms.
        points_per_frame (int): The number of data points shown per animation
            frame. Increasing this value reduces the number of frames required
            to display all the data, resulting in faster data transitions.
            Defaults to 1.
        fontsize (int): The fontsize for labels and axes ticks.
        legend_params (dict[str, Any] | None): A dictionary of Matplotlib
            properties for customizing the plot legend (e.g., fontsize, loc,
            handlelength). If not provided, Matplotlib's default legend
            properties will be used.
        title (str | None): The title for the created plot figure. If `None`,
            no title will be displayed. Defaults to `None`.

    Returns:
        FuncAnimation: A Matplotlib `FuncAnimation` object that animates the
            progression of input-output data over time.

    Raises:
        ValueError: If any array dimensions mismatch expected shapes, or if
            the lengths of `u_bounds_list` or `y_bounds_list` do not match the
            number of subplots.
    """
    # Validate data dimensions
    validate_data_dimensions(
        u_k=u_k,
        y_k=y_k,
        u_s=u_s,
        y_s=y_s,
        u_bounds_list=u_bounds_list,
        y_bounds_list=y_bounds_list,
    )

    # Retrieve number of input and output data sequences and their length
    m = u_k.shape[1]  # Number of inputs
    p = y_k.shape[1]  # Number of outputs
    T = u_k.shape[0]  # Length of data

    # Initialize Matplotlib params if not provided
    inputs_line_params = init_dict_if_none(inputs_line_params)
    outputs_line_params = init_dict_if_none(outputs_line_params)
    setpoints_line_params = init_dict_if_none(setpoints_line_params)
    bounds_line_params = init_dict_if_none(bounds_line_params)
    legend_params = init_dict_if_none(legend_params)

    # Create figure and subplots
    fig, axs_u, axs_y = create_input_output_figure(
        m=m, p=p, figsize=figsize, dpi=dpi, fontsize=fontsize, title=title
    )

    # Define input-output line lists
    u_lines: list[Line2D] = []
    y_lines: list[Line2D] = []
    u_s_lines: list[Line2D] = []
    y_s_lines: list[Line2D] = []

    # Define initial measurement rectangles and texts lists
    u_rects: list[Rectangle] = []
    u_right_rect_lines: list[Line2D] = []
    u_left_rect_lines: list[Line2D] = []
    u_init_texts: list[Text] = []
    u_control_texts: list[Text] = []
    y_rects: list[Rectangle] = []
    y_right_rect_lines: list[Line2D] = []
    y_left_rect_lines: list[Line2D] = []
    y_init_texts: list[Text] = []
    y_control_texts: list[Text] = []

    # Define y-axis center
    u_y_axis_centers: list[float] = []
    y_y_axis_centers: list[float] = []

    # Initialize input plot elements
    for i in range(m):
        # Get input setpoint if provided
        u_setpoint = u_s[:, i] if u_s is not None else None

        # Define plot index based on the number of input plots
        plot_index = -1 if m == 1 else i

        # Get input bounds if provided
        u_bounds = u_bounds_list[i] if u_bounds_list else None

        initialize_data_animation(
            axis=axs_u[i],
            data=u_k[:, i],
            setpoint=u_setpoint,
            index=plot_index,
            data_line_params=inputs_line_params,
            bounds_line_params=bounds_line_params,
            setpoint_line_params=setpoints_line_params,
            dynamic_setpoint_lines=dynamic_setpoint_lines,
            var_symbol="u",
            setpoint_var_symbol=u_setpoint_var_symbol,
            var_label="Input",
            initial_text=initial_excitation_text,
            control_text=control_text,
            fontsize=fontsize,
            legend_params=legend_params,
            data_lines=u_lines,
            setpoint_lines=u_s_lines,
            rects=u_rects,
            right_rect_lines=u_right_rect_lines,
            left_rect_lines=u_left_rect_lines,
            init_texts=u_init_texts,
            control_texts=u_control_texts,
            y_axis_centers=u_y_axis_centers,
            bounds=u_bounds,
            initial_steps=initial_steps,
            initial_steps_label=initial_steps_label,
            continuous_updates=continuous_updates,
            legend_loc="upper right",
        )

    # Initialize output plot elements
    for j in range(p):
        # Define plot index based on the number of output plots
        plot_index = -1 if p == 1 else j

        # Get output bounds if provided
        y_bounds = y_bounds_list[i] if y_bounds_list else None

        initialize_data_animation(
            axis=axs_y[j],
            data=y_k[:, j],
            setpoint=y_s[:, j],
            index=plot_index,
            data_line_params=outputs_line_params,
            bounds_line_params=bounds_line_params,
            setpoint_line_params=setpoints_line_params,
            dynamic_setpoint_lines=dynamic_setpoint_lines,
            var_symbol="y",
            setpoint_var_symbol=y_setpoint_var_symbol,
            var_label="Output",
            initial_text=initial_measurement_text,
            control_text=control_text,
            fontsize=fontsize,
            legend_params=legend_params,
            data_lines=y_lines,
            setpoint_lines=y_s_lines,
            rects=y_rects,
            right_rect_lines=y_right_rect_lines,
            left_rect_lines=y_left_rect_lines,
            init_texts=y_init_texts,
            control_texts=y_control_texts,
            y_axis_centers=y_y_axis_centers,
            bounds=y_bounds,
            initial_steps=initial_steps,
            initial_steps_label=initial_steps_label,
            continuous_updates=continuous_updates,
            legend_loc="lower right",
        )

    # Calculate text bounding box width for initial and
    # control data if initial steps highlighting is enabled
    init_text_width = 0.0
    control_text_width = 0.0
    if initial_steps:
        # Get initial text bounding box width
        init_text_width_input = get_text_width_in_data(
            text_object=u_init_texts[0], axis=axs_u[0], fig=fig
        )
        init_text_width_output = get_text_width_in_data(
            text_object=y_init_texts[0], axis=axs_y[0], fig=fig
        )

        # Calculate maximum text width between input and
        # output labels to show them at the same time
        init_text_width = max(init_text_width_input, init_text_width_output)

        # Get control text bounding box width
        control_text_width = get_text_width_in_data(
            text_object=u_control_texts[0], axis=axs_u[0], fig=fig
        )

    # Animation update function
    def update(frame: int) -> list[Any]:
        # Calculate the current index based on the number of points per frame,
        # ensuring it does not exceed the last valid data index
        current_index = min(frame * points_per_frame, T - 1)

        # Update input plot data
        for i in range(m):
            # Get input setpoint if provided
            u_s_data = u_s[: current_index + 1, i] if u_s is not None else None
            u_s_line = (
                u_s_lines[i]
                if u_s is not None and dynamic_setpoint_lines
                else None
            )

            # Get initial step plot elements if
            # initial steps highlighting is enabled
            if initial_steps:
                u_rect = u_rects[i]
                u_right_rect_line = u_right_rect_lines[i]
                u_init_text = u_init_texts[i]
                u_control_text = u_control_texts[i]

                # Get lower boundary line of the initial measurement
                # region if continuous updates are enabled
                u_left_rect_line = (
                    u_left_rect_lines[i] if continuous_updates else None
                )
            else:
                u_rect = None
                u_right_rect_line = None
                u_init_text = None
                u_control_text = None
                u_left_rect_line = None

            update_data_animation(
                index=current_index,
                data=u_k[: current_index + 1, i],
                setpoint=u_s_data,
                data_length=T,
                points_per_frame=points_per_frame,
                initial_steps=initial_steps,
                continuous_updates=continuous_updates,
                data_line=u_lines[i],
                setpoint_line=u_s_line,
                rect=u_rect,
                y_axis_center=u_y_axis_centers[i],
                right_rect_line=u_right_rect_line,
                left_rect_line=u_left_rect_line,
                init_text_obj=u_init_text,
                control_text_obj=u_control_text,
                display_initial_text=display_initial_text,
                display_control_text=display_control_text,
                init_text_width=init_text_width,
                control_text_width=control_text_width,
            )

        # Update output plot data
        for j in range(p):
            # Get output setpoint line
            y_s_line = y_s_lines[j] if dynamic_setpoint_lines else None

            # Get initial step plot elements if
            # initial steps highlighting is enabled
            if initial_steps:
                y_rect = y_rects[j]
                y_right_rect_line = y_right_rect_lines[j]
                y_init_text = y_init_texts[j]
                y_control_text = y_control_texts[j]

                # Get lower boundary line of the initial measurement
                # region if continuous updates are enabled
                y_left_rect_line = (
                    y_left_rect_lines[j] if continuous_updates else None
                )
            else:
                y_rect = None
                y_right_rect_line = None
                y_init_text = None
                y_control_text = None
                y_left_rect_line = None

            update_data_animation(
                index=current_index,
                data=y_k[: current_index + 1, j],
                setpoint=y_s[: current_index + 1, j],
                data_length=T,
                points_per_frame=points_per_frame,
                initial_steps=initial_steps,
                continuous_updates=continuous_updates,
                data_line=y_lines[j],
                setpoint_line=y_s_line,
                rect=y_rect,
                y_axis_center=y_y_axis_centers[j],
                right_rect_line=y_right_rect_line,
                left_rect_line=y_left_rect_line,
                init_text_obj=y_init_text,
                control_text_obj=y_control_text,
                display_initial_text=display_initial_text,
                display_control_text=display_control_text,
                init_text_width=init_text_width,
                control_text_width=control_text_width,
            )

        return (
            u_lines
            + y_lines
            + u_s_lines
            + y_s_lines
            + u_rects
            + u_right_rect_lines
            + u_left_rect_lines
            + u_init_texts
            + u_control_texts
            + y_rects
            + y_init_texts
            + y_control_texts
            + y_right_rect_lines
            + y_left_rect_lines
        )

    # Calculate the number of animation frames
    n_frames = math.ceil((T - 1) / points_per_frame) + 1

    # Create animation
    animation = FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True
    )

    return animation


def initialize_data_animation(
    axis: Axes,
    data: np.ndarray,
    setpoint: np.ndarray | None,
    index: int,
    data_line_params: dict[str, Any],
    setpoint_line_params: dict[str, Any],
    bounds_line_params: dict[str, Any],
    dynamic_setpoint_lines: bool,
    var_symbol: str,
    setpoint_var_symbol: str,
    var_label: str,
    initial_text: str,
    control_text: str,
    fontsize: int,
    legend_params: dict[str, Any],
    data_lines: list[Line2D],
    setpoint_lines: list[Line2D],
    rects: list[Rectangle],
    right_rect_lines: list[Line2D],
    left_rect_lines: list[Line2D],
    init_texts: list[Text],
    control_texts: list[Text],
    y_axis_centers: list[float],
    bounds: tuple[float, float] | None = None,
    initial_steps: int | None = None,
    initial_steps_label: str | None = None,
    continuous_updates: bool = False,
    legend_loc: str = "best",
) -> None:
    """
    Initialize plot elements for a data series animation with setpoints.

    This function initializes and appends several elements to the plot, such
    as plot lines representing data, rectangles and lines representing an
    initial input-output data measurement period, and text labels for both the
    initial measurement and control periods. It also adjusts the axis limits
    and stores the y-axis center values. The appearance of plot lines and
    legends can be customized by passing dictionaries of Matplotlib line and
    legend properties.

    Args:
        axis (Axes): The Matplotlib axis object to plot on.
        data (np.ndarray): An array containing data to be plotted.
        setpoint (float | None): An array containing setpoint values to be
            plotted. If `None`, the setpoint line will not be plotted.
        index (int): The index of the data used for labeling purposes (e.g.,
            "u_1", "u_2"). If set to -1, subscripts will not be added to
            labels.
        data_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the line used to plot the data series
            (e.g., color, linestyle, linewidth).
        setpoint_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the line used to plot the setpoint
            value (e.g., color, linestyle, linewidth).
        bounds_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the lines used to plot the bounds of
            the data series (e.g., color, linestyle, linewidth).
        dynamic_setpoint_lines (bool): Whether to update setpoint lines
            dynamically. If `False`, they will be fully plotted at the start
            using all setpoint data.
        var_symbol (str): The variable symbol used to label the data series
            (e.g., "u" for inputs, "y" for outputs).
        setpoint_var_symbol (str): The variable symbol used to label the
            setpoint data series (e.g., "u^s" for inputs, "y^s" for outputs).
        var_label (str): The variable label representing the control signal
            (e.g., "Input", "Output").
        initial_text (str): Label text to display over the initial measurement
            period of the plot.
        control_text (str): Label text to display over the post-initial
            control period.
        fontsize (int): The fontsize for labels and axes ticks.
        legend_params (dict[str, Any]): A dictionary of Matplotlib properties
            for customizing the plot legend (e.g., fontsize, loc,
            handlelength). If the 'loc' key is present in the dictionary, it
            overrides the `legend_loc` value.
        data_lines (list[Line2D]): The list where the initialized data plot
            lines will be stored.
        setpoint_lines (list[Line2D]): The list where the initialized setpoint
            plot lines will be stored.
        rects (list[Rectangle]): The list where the initialized rectangles
            representing the initial measurement region will be stored.
        right_rect_lines (list[Line2D]): The list where the initialized
            vertical lines representing the upper boundary of the initial
            measurement region will be stored.
        left_rect_lines (list[Line2D]): The list where the initialized
            vertical lines representing the lower boundary of the initial
            measurement region will be stored.
        init_texts (list[Text]): The list where the initialized initial
            measurement label texts will be stored.
        control_texts (list[Text]): The list where the initialized control
            label texts will be stored.
        y_axis_centers (list[float]): The list where the y-axis center from
            the adjusted axis will be stored.
        bounds (tuple[float, float] | None): A tuple (lower_bound, upper_bound)
            specifying the bounds of the data to be plotted. If provided,
            horizontal lines representing these bounds will be plotted.
            Defaults to `None`.
        initial_steps (int | None): The number of initial time steps during
            which input-output measurements were taken for the data-driven
            characterization of the system. This highlights the initial
            measurement period in the plot. If `None`, no special highlighting
            will be applied. Defaults to `None`.
        initial_steps_label (str | None): Label text to use for the legend
            entry representing the initial input-output measurement highlight
            in the plot. If `None`, this element will not appear in the
            legend. Defaults to `None`.
        continuous_updates (bool): Whether the initial measurement period
            highlight should move with the latest data to represent continuous
            input-output measurement updates. Defaults to `False`.
        legend_loc (str): The location of the legend on the plot. Corresponds
            to Matplotlib's `loc` parameter for legends. Defaults to 'best'.

    Note:
        This function updates the `lines`, `rects`, `right_rect_lines`,
        `left_rect_lines` `init_texts`, and `control_texts` with the
        initialized plot elements. It also adjusts the y-axis limits to a
        fixed range and stores the center values in `y_axis_centers`.
    """
    T = data.shape[0]  # Data length

    # Construct index label string based on index value
    index_str = f"_{index + 1}" if index != -1 else ""

    # Initialize data plot lines
    data_lines.append(
        axis.plot(
            [], [], **data_line_params, label=f"${var_symbol}{index_str}$"
        )[0]
    )

    # Plot bounds if provided
    if bounds is not None:
        lower_bound, upper_bound = bounds
        bounds_label = "Constraints"
        # Plot lower bound line
        axis.axhline(y=lower_bound, **bounds_line_params, label=bounds_label)
        # Plot upper bound line
        axis.axhline(y=upper_bound, **bounds_line_params)

    # Initialize setpoint plot lines if provided
    setpoint_label = f"${setpoint_var_symbol}{index_str}$"
    if setpoint is not None:
        if dynamic_setpoint_lines:
            setpoint_lines.append(
                axis.plot(
                    [], [], **setpoint_line_params, label=setpoint_label
                )[0]
            )
        else:
            # Plot setpoint line with entire setpoint data
            axis.plot(
                range(0, T),
                setpoint,
                **setpoint_line_params,
                label=setpoint_label,
            )

    # Define axis limits
    # Get minimum and maximum Y-axis values from data and setpoint
    if setpoint is not None:
        u_lim_min, u_lim_max = get_padded_limits(data, setpoint)
    else:
        u_lim_min, u_lim_max = get_padded_limits(data)

    # Compare minimum and maximum values with bounds, if provided
    if bounds is not None:
        u_lim_min, u_lim_max = get_padded_limits(
            np.array([u_lim_min, u_lim_max]), np.array(bounds)
        )

    # Set axis limits
    axis.set_xlim((0, T - 1))
    axis.set_ylim(u_lim_min, u_lim_max)
    y_axis_centers.append((u_lim_min + u_lim_max) / 2)

    # Highlight initial input-output data measurement period if provided
    if initial_steps:
        # Initialize initial measurement rectangle
        rect = axis.axvspan(0, 0, color="gray", alpha=0.1)
        rects.append(rect)

        # Initialize initial measurement rectangle boundary lines
        right_rect_lines.append(
            axis.axvline(
                x=0, color="black", linestyle=(0, (5, 5)), linewidth=1
            )
        )

        if continuous_updates:
            # Add left boundary line to show continuous updates, if enabled
            left_rect_lines.append(
                axis.axvline(
                    x=0, color="black", linestyle=(0, (5, 5)), linewidth=1
                )
            )

        # Get y axis center
        y_axis_center = (
            y_axis_centers[index] if index != -1 else y_axis_centers[0]
        )

        # Initialize initial measurement text
        init_texts.append(
            axis.text(
                initial_steps / 2,
                y_axis_center,
                initial_text,
                fontsize=fontsize - 1,
                ha="center",
                va="center",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "black"},
            )
        )

        # Initialize control text
        control_texts.append(
            axis.text(
                (T + initial_steps) / 2,
                y_axis_center,
                control_text,
                fontsize=fontsize - 1,
                ha="center",
                va="center",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "black"},
            )
        )

    # Format labels and ticks
    axis.set_xlabel("Time step $k$", fontsize=fontsize)
    axis.set_ylabel(
        f"{var_label} ${var_symbol}{index_str}$", fontsize=fontsize
    )
    axis.tick_params(axis="both", labelsize=fontsize)

    # Format legend:
    # Collect all legend handles and labels
    handles, labels = axis.get_legend_handles_labels()
    custom_handlers = {}

    # Add rectangle to legend with custom handler if used
    if initial_steps and initial_steps_label:
        handles.append(rect)
        labels.append(initial_steps_label)

        # Add legend with custom handlers
        custom_handlers = {Rectangle: HandlerInitMeasurementRect()}

    # Create a mapping of labels to handles for labels repositioning
    labels_map = dict(zip(labels, handles, strict=False))

    # Reposition labels to move bound and setpoint labels to the last
    end_labels_list = [setpoint_label]
    if bounds is not None:
        end_labels_list.append(bounds_label)
    if initial_steps_label:
        end_labels_list.append(initial_steps_label)

    for last_label in end_labels_list:
        if last_label in labels_map:
            # Move the last label to the end
            last_handle = labels_map.pop(last_label)
            labels_map[last_label] = last_handle

    # Format legend
    axis.legend(
        handles=labels_map.values(),
        labels=labels_map.keys(),
        handler_map=custom_handlers,
        **legend_params,
        loc=legend_loc,
    )


def update_data_animation(
    index: int,
    data: np.ndarray,
    setpoint: np.ndarray | None,
    data_length: int,
    points_per_frame: int,
    initial_steps: int | None,
    continuous_updates: bool,
    data_line: Line2D,
    setpoint_line: Line2D | None,
    rect: Rectangle | None,
    y_axis_center: float,
    right_rect_line: Line2D | None,
    left_rect_line: Line2D | None,
    init_text_obj: Text | None,
    control_text_obj: Text | None,
    display_initial_text: bool,
    display_control_text: bool,
    init_text_width: float,
    control_text_width: float,
) -> None:
    """
    Update the plot elements in a data series animation with setpoints.

    This function updates data plot elements based on the current data index.
    If 'initial_steps' is provided, it also updates the rectangle and line
    representing the initial input-output measurement period, as well as the
    text labels indicating the initial measurement and control periods. These
    labels will be displayed if there is enough space to prevent them from
    overlapping with other plot elements.

    Args:
        index (int): The current data index.
        data (np.ndarray): An array containing data to be plotted.
        setpoint (float | None): An array containing setpoint values to be
            plotted. If `None`, the setpoint line will not be plotted.
        data_length (int): The length of the `data` array.
        points_per_frame (int): The number of data points shown per animation
            frame.
        initial_steps (int| None): The number of initial time steps during
            which input-output measurements were taken for the data-driven
            characterization of the system. This highlights the initial
            measurement period in the plot.
        continuous_updates (bool): Whether the initial measurement period
            highlight should move with the latest data to represent continuous
            input-output measurement updates.
        data_line (Line2D): The plot line corresponding to the data series
            plot.
        setpoint_line (Line2D | None): The plot line corresponding to the
            setpoint series plot. If `None`, the setpoint line will not be
            plotted.
        rect (Rectangle | None): The rectangle representing the initial
            measurement region.
        y_axis_center (float): The y-axis center of the plot axis.
        right_rect_line (Line2D | None): The line object representing the upper
            boundary of the initial measurement region.
        left_rect_line (Line2D | None]): The line object representing the lower
            boundary of the initial measurement region.
        init_text_obj (Text | None): The text object containing the initial
            measurement period label.
        control_text_obj (Text | None): The text object containing the control
            period label.
        display_initial_text (bool): Whether to display the `initial_text`
            label on the plot.
        display_control_text (bool): Whether to display the `control_text`
            label on the plot.
        init_text_width (float): The width of the `init_text_obj` object in
            data coordinates.
        control_text_width (float): The width of the `control_text_obj` object
            in data coordinates.
    """
    # Update data plot line
    data_line.set_data(range(0, index + 1), data[: index + 1])

    # Update setpoint plot line
    if setpoint is not None and setpoint_line is not None:
        setpoint_line.set_data(range(0, index + 1), setpoint[: index + 1])

    # Determine if an update is needed. Always update for continuous updates
    if initial_steps:
        needs_update = (
            index <= initial_steps + points_per_frame or continuous_updates
        )
    else:
        needs_update = False

    # Update initial measurement rectangle and texts
    if initial_steps and needs_update:
        # Calculate measurement period limit index
        lim_index = (
            min(index, initial_steps) if not continuous_updates else index
        )

        # Ensure type safety for static checking
        assert rect is not None
        assert right_rect_line is not None
        assert init_text_obj is not None
        assert control_text_obj is not None

        # Update rectangle width
        rect_width = min(lim_index, initial_steps)
        rect.set_width(rect_width)

        # Update rectangle position
        left_index = max(lim_index - initial_steps, 0)
        rect.set_xy((left_index, 0))

        # Update rectangle boundary line positions
        right_rect_line.set_xdata([lim_index])

        if continuous_updates and left_rect_line:
            # Update left boundary line if continuous updates are enabled
            left_rect_line.set_xdata([left_index])

            # Toggle visibility based on its index
            left_rect_line.set_visible(left_index != 0)

        # Hide initial measurement and control texts
        init_text_obj.set_visible(False)
        control_text_obj.set_visible(False)

        # Show initial measurement text
        if display_initial_text and index >= init_text_width:
            init_text_obj.set_position((lim_index / 2, y_axis_center))
            init_text_obj.set_visible(True)

        # Show control text if possible
        if display_control_text and index >= initial_steps:
            if (data_length - initial_steps) >= control_text_width:
                control_text_obj.set_visible(True)


def save_animation(
    animation: FuncAnimation,
    total_frames: int,
    fps: int,
    bitrate: int,
    file_path: str,
) -> None:
    """
    Save a Matplotlib animation using an ffmpeg writer with progress bar
    tracking.

    This function saves the given Matplotlib animation to the specified file
    path and displays a progress bar the console to track the saving progress.
    If the file path contains directories that do not exist, they will be
    created.

    Args:
        animation (FuncAnimation): The animation object to save.
        total_frames (int): The total number of frames in the animation.
        fps (int): The frames per second of saved video.
        bitrate (int): The bitrate of saved video.
        file_path (str): The path (including directory and file name) where
            the animation will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Set up the ffmpeg writer
    writer = FFMpegWriter(fps=fps, metadata={"artist": "Me"}, bitrate=bitrate)

    # Save animation while displaying a progress bar
    with tqdm(total=total_frames, desc="Saving animation") as pbar:
        animation.save(
            file_path,
            writer=writer,
            progress_callback=lambda i, n: pbar.update(1),
        )
