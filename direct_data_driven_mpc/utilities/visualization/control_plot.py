from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from .plot_utilities import (
    create_input_output_figure,
    filter_and_reorder_legend,
    get_text_width_in_data,
    init_dict_if_none,
    validate_data_dimensions,
)


def plot_input_output(
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
    u_setpoint_var_symbol: str = "u^s",
    y_setpoint_var_symbol: str = "y^s",
    initial_steps: int | None = None,
    initial_excitation_text: str = "Init. Excitation",
    initial_measurement_text: str = "Init. Measurement",
    control_text: str = "Data-Driven MPC",
    display_initial_text: bool = True,
    display_control_text: bool = True,
    figsize: tuple[float, float] = (12.0, 8.0),
    dpi: int = 300,
    u_ylimits_list: list[tuple[float, float]] | None = None,
    y_ylimits_list: list[tuple[float, float]] | None = None,
    fontsize: int = 12,
    legend_params: dict[str, Any] | None = None,
    var_suffix: str = "",
    axs_u: list[Axes] | None = None,
    axs_y: list[Axes] | None = None,
    title: str | None = None,
    input_label: str | None = None,
    output_label: str | None = None,
    u_setpoint_labels: list[str] | None = None,
    y_setpoint_labels: list[str] | None = None,
    x_axis_labels: list[str] | None = None,
    input_y_axis_labels: list[str] | None = None,
    output_y_axis_labels: list[str] | None = None,
) -> None:
    """
    Plot input-output data with setpoints in a Matplotlib figure.

    This function creates a figure with two rows of subplots, with the first
    row containing control inputs, and the second row, system outputs. Each
    subplot shows the data series for each data sequence alongside its setpoint
    as a constant line. The appearance of plot lines and legends can be
    customized by passing dictionaries of Matplotlib line and legend
    properties.

    If provided, the first 'initial_steps' time steps are highlighted to
    emphasize the initial input-output data measurement period representing
    the data-driven system characterization phase in a Data-Driven MPC
    algorithm. Additionally, custom labels can be displayed to indicate the
    initial measurement and the subsequent MPC control periods, but only if
    there is enough space to prevent them from overlapping with other plot
    elements.

    Note:
        If `axs_u` and `axs_y` are provided, the data will be plotted on these
        external axes and no new figure will be created. This allows for
        multiple data sequences to be plotted on the same external figure.
        Each data sequence can be differentiated  using the `data_label`
        argument.

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
        u_setpoint_var_symbol (str): The variable symbol used to label the
            input setpoint data series (e.g., "u^s").
        y_setpoint_var_symbol (str): The variable symbol used to label the
            output setpoint data series (e.g., "y^s").
        initial_steps (int | None): The number of initial time steps during
            which input-output measurements were taken for the data-driven
            characterization of the system. This highlights the initial
            measurement period in the plot. If `None`, no special highlighting
            will be applied. Defaults to `None`.
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
        u_ylimits_list (list[tuple[float, float]] | None): A list of tuples
            (lower_limit, upper_limit) specifying the Y-axis limits for each
            input subplot. If `None`, the Y-axis limits will be determined
            automatically. Defaults to `None`.
        y_ylimits_list (list[tuple[float, float]] | None): A list of tuples
            (lower_limit, upper_limit) specifying the Y-axis limits for each
            output subplot. If `None`, the Y-axis limits will be determined
            automatically. Defaults to `None`.
        fontsize (int): The fontsize for labels and axes ticks.
        legend_params (dict[str, Any] | None): A dictionary of Matplotlib
            properties for customizing the plot legends (e.g., fontsize,
            loc, handlelength). If not provided, Matplotlib's default legend
            properties will be used.
        var_suffix (str): A string appended to each data series label in the
            plot legend.
        axs_u (list[Axes] | None): A list of external axes for input plots.
            Defaults to `None`.
        axs_y (list[Axes] | None): A list of external axes for output plots.
            Defaults to `None`.
        title (str | None): The title for the created plot figure. Set only if
            the figure is created internally (i.e., `axs_u` and `axs_y` are not
            provided). If `None`, no title will be displayed. Defaults to
            `None`.
        input_label (str | None): A custom legend label for the input data
            series. If provided, this label will override the default label
            constructed using `var_suffix`.
        output_label (str | None): A custom legend label for the output data
            series. If provided, this label will override the default label
            constructed using `var_suffix`.
        u_setpoint_labels (list[str] | None): A list of strings specifying
            custom legend labels for input setpoint series. If provided, the
            label at each index will override the default label constructed
            using `u_setpoint_var_symbol`.
        y_setpoint_labels (list[str] | None): A list of strings specifying
            custom legend labels for output setpoint series. If provided, the
            label at each index will override the default label constructed
            using `y_setpoint_var_symbol`.
        x_axis_labels (list[str] | None): A list of strings specifying custom
            X-axis labels for each subplot. If provided, the label at each
            index will override the default "Time step $k$".
        input_y_axis_labels (list[str] | None): A list of strings specifying
            custom Y-axis labels for each input subplot. If provided, the label
            at each index will override the default constructed labels.
        output_y_axis_labels (list[str] | None): A list of strings specifying
            custom Y-axis labels for each output subplot. If provided, the
            label at each index will override the default constructed labels.

    Raises:
        ValueError: If any array dimensions mismatch expected shapes, or if
            the lengths of `u_bounds_list`, `y_bounds_list`, `u_ylimits_list`,
            or `y_ylimits_list` do not match the number of subplots.
    """
    # Validate data dimensions
    validate_data_dimensions(
        u_k=u_k,
        y_k=y_k,
        u_s=u_s,
        y_s=y_s,
        u_bounds_list=u_bounds_list,
        y_bounds_list=y_bounds_list,
        u_ylimits_list=u_ylimits_list,
        y_ylimits_list=y_ylimits_list,
        u_setpoint_labels=u_setpoint_labels,
        y_setpoint_labels=y_setpoint_labels,
        x_axis_labels=x_axis_labels,
        input_y_axis_labels=input_y_axis_labels,
        output_y_axis_labels=output_y_axis_labels,
    )

    # Initialize Matplotlib params if not provided
    inputs_line_params = init_dict_if_none(inputs_line_params)
    outputs_line_params = init_dict_if_none(outputs_line_params)
    setpoints_line_params = init_dict_if_none(setpoints_line_params)
    bounds_line_params = init_dict_if_none(bounds_line_params)
    legend_params = init_dict_if_none(legend_params)

    # Retrieve number of input and output data sequences
    m = u_k.shape[1]  # Number of inputs
    p = y_k.shape[1]  # Number of outputs

    # Create figure if lists of Axes are not provided
    is_ext_fig = axs_u is not None and axs_y is not None  # External figure

    fig: Figure | SubFigure

    if not is_ext_fig:
        # Create figure and subplots
        fig, axs_u, axs_y = create_input_output_figure(
            m=m, p=p, figsize=figsize, dpi=dpi, fontsize=fontsize, title=title
        )
    else:
        assert axs_u is not None  # Prevent mypy [index] error

        # Use figure from the provided axes
        fig = axs_u[0].figure

    # Plot input data
    m = u_k.shape[1]  # Number of inputs
    p = y_k.shape[1]  # Number of outputs

    for i in range(m):
        # Get input setpoint if provided
        u_setpoint = u_s[:, i] if u_s is not None else None

        # Define plot index based on the number of input plots
        plot_index = -1 if m == 1 else i

        # Get input bounds if provided
        u_bounds = u_bounds_list[i] if u_bounds_list else None

        # Get plot Y-axis limit if provided
        u_plot_ylimit = u_ylimits_list[i] if u_ylimits_list else None

        # Prevent mypy [index] error
        assert axs_u is not None

        # Plot data
        plot_data(
            axis=axs_u[i],
            data=u_k[:, i],
            setpoint=u_setpoint,
            index=plot_index,
            data_line_params=inputs_line_params,
            bounds_line_params=bounds_line_params,
            setpoint_line_params=setpoints_line_params,
            var_symbol="u",
            setpoint_var_symbol=u_setpoint_var_symbol,
            var_label="Input",
            var_suffix=var_suffix,
            initial_text=initial_excitation_text,
            control_text=control_text,
            display_initial_text=display_initial_text,
            display_control_text=display_control_text,
            fontsize=fontsize,
            legend_params=legend_params,
            fig=fig,
            bounds=u_bounds,
            initial_steps=initial_steps,
            plot_ylimits=u_plot_ylimit,
            data_label=input_label,
            setpoint_labels=u_setpoint_labels,
            x_axis_labels=x_axis_labels,
            y_axis_labels=input_y_axis_labels,
        )

    # Plot output data
    for j in range(p):
        # Define plot index based on the number of output plots
        plot_index = -1 if p == 1 else j

        # Get output bounds if provided
        y_bounds = y_bounds_list[j] if y_bounds_list else None

        # Get plot Y-axis limit if provided
        y_plot_ylimits = y_ylimits_list[j] if y_ylimits_list else None

        # Prevent mypy [index] error
        assert axs_y is not None

        # Plot data
        plot_data(
            axis=axs_y[j],
            data=y_k[:, j],
            setpoint=y_s[:, j],
            index=plot_index,
            data_line_params=outputs_line_params,
            bounds_line_params=bounds_line_params,
            setpoint_line_params=setpoints_line_params,
            var_symbol="y",
            setpoint_var_symbol=y_setpoint_var_symbol,
            var_label="Output",
            var_suffix=var_suffix,
            initial_text=initial_measurement_text,
            control_text=control_text,
            display_initial_text=display_initial_text,
            display_control_text=display_control_text,
            fontsize=fontsize,
            legend_params=legend_params,
            fig=fig,
            bounds=y_bounds,
            initial_steps=initial_steps,
            plot_ylimits=y_plot_ylimits,
            data_label=output_label,
            setpoint_labels=y_setpoint_labels,
            x_axis_labels=x_axis_labels,
            y_axis_labels=output_y_axis_labels,
        )

    # Show the plot if the figure was created internally
    if not is_ext_fig:
        plt.show()


def plot_data(
    axis: Axes,
    data: np.ndarray,
    setpoint: np.ndarray | None,
    index: int,
    data_line_params: dict[str, Any],
    setpoint_line_params: dict[str, Any],
    bounds_line_params: dict[str, Any],
    var_symbol: str,
    setpoint_var_symbol: str,
    var_label: str,
    var_suffix: str,
    initial_text: str,
    control_text: str,
    display_initial_text: bool,
    display_control_text: bool,
    fontsize: int,
    legend_params: dict[str, Any],
    fig: Figure | SubFigure,
    bounds: tuple[float, float] | None = None,
    initial_steps: int | None = None,
    plot_ylimits: tuple[float, float] | None = None,
    data_label: str | None = None,
    setpoint_labels: list[str] | None = None,
    x_axis_labels: list[str] | None = None,
    y_axis_labels: list[str] | None = None,
) -> None:
    """
    Plot a data series with setpoints in a specified axis. Optionally,
    highlight the initial measurement and control phases using shaded regions
    and text labels. The labels will be displayed if there is enough space to
    prevent them from overlapping with other plot elements.

    Note:
        The appearance of plot lines and legend can be customized by passing
        dictionaries of Matplotlib line and legend properties.

    Args:
        axis (Axes): The Matplotlib axis object to plot on.
        data (np.ndarray): An array containing data to be plotted.
        setpoint (np.ndarray | None): An array containing setpoint values to be
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
        var_symbol (str): The variable symbol used to label the data series
            (e.g., "u" for inputs, "y" for outputs).
        setpoint_var_symbol (str): The variable symbol used to label the
            setpoint data series (e.g., "u^s" for inputs, "y^s" for outputs).
        var_label (str): The variable label representing the control signal
            (e.g., "Input", "Output").
        var_suffix (str): A string appended to each data series label in the
            plot legend.
        initial_text (str): Label text to display over the initial measurement
            period of the plot.
        control_text (str): Label text to display over the post-initial
            control period.
        display_initial_text (bool): Whether to display the `initial_text`
            label on the plot.
        display_control_text (bool): Whether to display the `control_text`
            label on the plot.
        fontsize (int): The fontsize for labels and axes ticks.
        legend_params (dict[str, Any]): A dictionary of Matplotlib properties
            for customizing the plot legend (e.g., fontsize, loc,
            handlelength).
        fig (Figure | SubFigure): The Matplotlib figure or subfigure that
            contains the axis.
        bounds (tuple[float, float] | None): A tuple (lower_bound,
            upper_bound) specifying the bounds of the data to be plotted. If
            provided, horizontal lines representing these bounds will be
            plotted. Defaults to `None`.
        initial_steps (int | None): The number of initial time steps during
            which input-output measurements were taken for the data-driven
            characterization of the system. This highlights the initial
            measurement period in the plot. Defaults to `None`.
        plot_ylimits (tuple[float, float] | None): A tuple (lower_limit,
            upper_limit) specifying the Y-axis limits for the plot. If `None`,
            the Y-axis limits will be determined automatically. Defaults to
            `None`.
        data_label (str | None): A custom legend label for the data series. If
            provided, this label will override the default constructed label
            using `var_symbol` and `var_suffix`.
        setpoint_labels (list[str] | None): A list of strings specifying custom
            legend labels for the setpoint series. If provided, the label at
            `index` will be used instead of the default label constructed using
            `setpoint_var_symbol`.
        x_axis_labels (list[str] | None): A list of strings specifying custom
            X-axis labels for each subplot or data index. If provided, the
            label at `index` will override the default "Time step $k$".
        y_axis_labels (list[str] | None): A list of strings specifying custom
            Y-axis labels for each subplot or data index. If provided, the
            label at `index` will override the default label constructed from
            `var_label` and `var_symbol`.
    """
    T = data.shape[0]  # Data length

    # Construct index label string based on index value
    index_str = f"_{index + 1}" if index != -1 else ""

    # Plot data series
    data_label_str = (
        data_label if data_label else f"${var_symbol}{index_str}${var_suffix}"
    )

    axis.plot(
        range(0, T),
        data,
        **data_line_params,
        label=data_label_str,
    )

    # Plot setpoint if provided
    setpoint_label_str = (
        setpoint_labels[index]
        if setpoint_labels
        else f"${setpoint_var_symbol}{index_str}$"
    )

    if setpoint is not None:
        axis.plot(
            range(0, T),
            setpoint,
            **setpoint_line_params,
            label=setpoint_label_str,
        )

    # Plot bounds if provided
    if bounds is not None:
        lower_bound, upper_bound = bounds
        bounds_label = "Constraints"
        # Plot lower bound line
        axis.axhline(y=lower_bound, **bounds_line_params, label=bounds_label)
        # Plot upper bound line
        axis.axhline(y=upper_bound, **bounds_line_params)

    # Highlight initial input-output data measurement period if provided
    if initial_steps:
        # Highlight period with a grayed rectangle
        axis.axvspan(0, initial_steps, color="gray", alpha=0.1)
        # Add a vertical line at the right side of the rectangle
        axis.axvline(
            x=initial_steps, color="black", linestyle=(0, (5, 5)), linewidth=1
        )

        # Display initial measurement text if enabled
        if display_initial_text:
            # Get y-axis limits
            y_min, y_max = axis.get_ylim()
            # Place label at the center of the highlighted area
            u_init_text = axis.text(
                initial_steps / 2,
                (y_min + y_max) / 2,
                initial_text,
                fontsize=fontsize - 1,
                ha="center",
                va="center",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "black"},
            )
            # Get initial text bounding box width
            init_text_width = get_text_width_in_data(
                text_object=u_init_text, axis=axis, fig=fig
            )
            # Hide text box if it overflows the plot area
            if initial_steps < init_text_width:
                u_init_text.set_visible(False)

        # Display Data-Driven MPC control text if enabled
        if display_control_text:
            # Get y-axis limits
            y_min, y_max = axis.get_ylim()
            # Place label at the center of the remaining area
            u_control_text = axis.text(
                (T + initial_steps) / 2,
                (y_min + y_max) / 2,
                control_text,
                fontsize=fontsize - 1,
                ha="center",
                va="center",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "black"},
            )
            # Get control text bounding box width
            control_text_width = get_text_width_in_data(
                text_object=u_control_text, axis=axis, fig=fig
            )
            # Hide text box if it overflows the plot area
            if (T - initial_steps) < control_text_width:
                u_control_text.set_visible(False)

    # Format labels, legend and ticks
    x_axis_label_str = (
        x_axis_labels[index] if x_axis_labels else "Time step $k$"
    )
    y_axis_label_str = (
        y_axis_labels[index]
        if y_axis_labels
        else f"{var_label} ${var_symbol}{index_str}$"
    )

    axis.set_xlabel(x_axis_label_str, fontsize=fontsize)
    axis.set_ylabel(y_axis_label_str, fontsize=fontsize)
    axis.legend(**legend_params)
    axis.tick_params(axis="both", labelsize=fontsize)

    # Remove duplicate labels from legend (required for external figures
    # that plot multiple data sequences on the same plot to avoid label
    # repetition) and reposition labels
    end_labels_list = [setpoint_label_str]
    if bounds is not None:
        end_labels_list.append(bounds_label)

    filter_and_reorder_legend(
        axis=axis, legend_params=legend_params, end_labels_list=end_labels_list
    )

    # Set x-limits
    axis.set_xlim((0, T - 1))

    # Set y-limits if provided
    if plot_ylimits:
        axis.set_ylim(plot_ylimits)
