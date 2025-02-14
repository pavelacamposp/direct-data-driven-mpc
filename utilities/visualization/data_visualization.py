from typing import Tuple, Optional, List, Union, Any
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerPatch
from matplotlib.legend import Legend
from matplotlib.transforms import Transform

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
import os
import math

# Define a custom legend handler for the rectangle representing
# the initial input-output measurement period in plots
class HandlerInitMeasurementRect(HandlerPatch):
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Rectangle,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform
    ) -> List[Union[Rectangle, Line2D]]:
        # Define the main rectangle
        rect = Rectangle([xdescent, ydescent],
                         width,
                         height,
                         transform=trans,
                         color=orig_handle.get_facecolor(),
                         alpha=orig_handle.get_alpha())
        
        # Create dashed vertical lines at the sides of the rectangle
        line1 = Line2D([xdescent, xdescent],
                       [ydescent, ydescent + height],
                       color='black',
                       linestyle=(0, (2, 2)),
                       linewidth=1)
        line2 = Line2D([xdescent + width, xdescent + width],
                       [ydescent, ydescent + height],
                       color='black',
                       linestyle=(0, (2, 2)),
                       linewidth=1)
        
        # Add transform to the vertical lines
        line1.set_transform(trans)
        line2.set_transform(trans)
        
        return [rect, line1, line2]

def plot_input_output(
    u_k: np.ndarray,
    y_k: np.ndarray,
    u_s: np.ndarray,
    y_s: np.ndarray,
    u_bounds_list: Optional[List[Tuple[float, float]]] = None,
    y_bounds_list: Optional[List[Tuple[float, float]]] = None,
    inputs_line_params: dict[str, Any] = {},
    outputs_line_params: dict[str, Any] = {},
    setpoints_line_params: dict[str, Any] = {},
    bounds_line_params: dict[str, Any] = {},
    initial_steps: Optional[int] = None,
    initial_excitation_text: str = "Init. Excitation",
    initial_measurement_text: str = "Init. Measurement",
    control_text: str = "Data-Driven MPC",
    display_initial_text: bool = True,
    display_control_text: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    u_ylimits_list: Optional[List[Tuple[float, float]]] = None,
    y_ylimits_list: Optional[List[Tuple[float, float]]] = None,
    fontsize: int = 12,
    legend_params: dict[str, Any] = {},
    data_label: str = "",
    axs_u: Optional[List[Axes]] = None,
    axs_y: Optional[List[Axes]] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot input-output data with setpoints in a Matplotlib figure.

    This function creates 2 rows of subplots, with the first row containing
    control inputs, and the second row, system outputs. Each subplot shows the
    data series for each data sequence alongside its setpoint as a constant
    line. The appearance of plot lines and legends can be customized by
    passing dictionaries of Matplotlib line and legend properties.

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
        u_s (np.ndarray): An array of shape (m, 1) containing `m` input
            setpoint values.
        y_s (np.ndarray): An array of shape (p, 1) containing `p` output
            setpoint values.
        u_bounds_list (Optional[List[Tuple[float, float]]]): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each input data
            sequence. If provided, horizontal lines representing these bounds
            will be plotted in each subplot. If `None`, no horizontal lines
            will be plotted. The number of tuples must match the number of
            input data sequences. Defaults to `None`.
        y_bounds_list (Optional[List[Tuple[float, float]]]): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each output data
            sequence. If provided, horizontal lines representing these bounds
            will be plotted in each subplot. If `None`, no horizontal lines
            will be plotted. The number of tuples must match the number of
            output data sequences. Defaults to `None`.
        inputs_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the lines used to plot the input data
            series (e.g., color, linestyle, linewidth). If not provided,
            Matplotlib's default line properties will be used.
        outputs_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the lines used to plot the output data
            series (e.g., color, linestyle, linewidth). If not provided,
            Matplotlib's default line properties will be used.
        setpoints_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the lines used to plot the setpoint
            values (e.g., color, linestyle, linewidth). If not provided,
            Matplotlib's default line properties will be used.
        bounds_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the lines used to plot the bounds of
            input-output data series (e.g., color, linestyle, linewidth). If
            not provided, Matplotlib's default line properties will be used.
        initial_steps (Optional[int]): The number of initial time steps during
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
        figsize (Tuple[int, int]): The (width, height) dimensions of the
            created Matplotlib figure.
        dpi (int): The DPI resolution of the figure.
        u_ylimits_list (Optional[List[Tuple[float, float]]]): A list of tuples
            (lower_limit, upper_limit) specifying the Y-axis limits for each
            input subplot. If `None`, the Y-axis limits will be determined
            automatically. Defaults to `None`.
        y_ylimits_list (Optional[List[Tuple[float, float]]]): A list of tuples
            (lower_limit, upper_limit) specifying the Y-axis limits for each
            output subplot. If `None`, the Y-axis limits will be determined
            automatically. Defaults to `None`.
        fontsize (int): The fontsize for labels and axes ticks.
        legend_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the plot legends (e.g., fontsize,
            loc, handlelength).
        data_label (str): The label for the current data sequences.
        axs_u (Optional[List[Axes]]): List of external axes for input plots.
            Defaults to `None`.
        axs_y (Optional[List[Axes]]): List of external axes for output plots.
            Defaults to `None`.
        title (Optional[str]): The title for the created plot figure. Set
            only if the figure is created internally (i.e., `axs_u` and
            `axs_y` are not provided). If `None`, no title will be displayed.
            Defaults to `None`.
    
    Raises:
        ValueError: If any array dimensions mismatch expected shapes, or if
            the lengths of `u_bounds_list`, `y_bounds_list`, `u_ylimits_list`,
            or `y_ylimits_list` do not match the number of subplots.
    """
    # Check input-output data dimensions
    if not (u_k.shape[0] == y_k.shape[0]):
        raise ValueError("Dimension mismatch. The number of time steps for "
                         "u_k and y_k do not match.")
    if not (u_k.shape[1] == u_s.shape[0] and y_k.shape[1] == y_s.shape[0]):
        raise ValueError("Dimension mismatch. The number of inputs from u_k "
                         "and u_s, and the number of outputs from y_k and "
                         "y_s should match.")
    
    # Retrieve number of input and output data sequences
    m = u_k.shape[1]  # Number of inputs
    p = y_k.shape[1]  # Number of outputs

    # Error handling for bounds list lengths
    if u_bounds_list and len(u_bounds_list) != m:
        raise ValueError(f"The length of `u_bounds_list` ("
                         f"{len(u_bounds_list)}) does not match the number "
                         f"of input subplots ({m}).")
    if y_bounds_list and len(y_bounds_list) != p:
        raise ValueError(f"The length of `y_bounds_list` ("
                         f"{len(y_bounds_list)}) does not match the number "
                         f"of output subplots ({p}).")

    # Error handling for y-limit lengths
    if u_ylimits_list and len(u_ylimits_list) != m:
        raise ValueError(f"The length of `u_ylimits_list` ("
                         f"{len(u_ylimits_list)}) does not match the number "
                         f"of input subplots ({m}).")
    if y_ylimits_list and len(y_ylimits_list) != p:
        raise ValueError(f"The length of `y_ylimits_list` ("
                         f"{len(y_ylimits_list)}) does not match the number "
                         f"of output subplots ({p}).")
    
    # Create figure if lists of Axes are not provided
    is_ext_fig = axs_u is not None and axs_y is not None  # External figure
    if not is_ext_fig:
        # Create figure and subplots
        fig, axs_u, axs_y = create_input_output_figure(m=m,
                                                       p=p,
                                                       figsize=figsize,
                                                       dpi=dpi,
                                                       fontsize=fontsize,
                                                       title=title)
    else:
        # Use figure from the provided axes
        fig = axs_u[0].figure

    # Plot input data
    for i in range(m):
        # Get input bounds if provided
        u_bounds = u_bounds_list[i] if u_bounds_list else None
        # Get plot Y-axis limit if provided
        u_plot_ylimit = u_ylimits_list[i] if u_ylimits_list else None
        # Plot data
        plot_data(axis=axs_u[i],
                  data=u_k[:, i],
                  setpoint=u_s[i, :],
                  index=i,
                  data_line_params=inputs_line_params,
                  bounds_line_params=bounds_line_params,
                  setpoint_line_params=setpoints_line_params,
                  var_symbol="u",
                  var_label="Input",
                  data_label=data_label,
                  initial_text=initial_excitation_text,
                  control_text=control_text,
                  display_initial_text=display_initial_text,
                  display_control_text=display_control_text,
                  fontsize=fontsize,
                  legend_params=legend_params,
                  fig=fig,
                  bounds=u_bounds,
                  initial_steps=initial_steps,
                  plot_ylimits=u_plot_ylimit)
        
        # Remove duplicate labels from legend
        # if figure was created externally
        if is_ext_fig:
            remove_legend_duplicates(axis=axs_u[i],
                                     legend_params=legend_params,
                                     last_label=f'$u_{i + 1}^s$')

    # Plot output data
    for j in range(p):
        # Get output bounds if provided
        y_bounds = y_bounds_list[i] if y_bounds_list else None
        # Get plot Y-axis limit if provided
        y_plot_ylimits = y_ylimits_list[j] if y_ylimits_list else None
        # Plot data
        plot_data(axis=axs_y[j],
                  data=y_k[:, j],
                  setpoint=y_s[j, :],
                  index=j,
                  data_line_params=outputs_line_params,
                  bounds_line_params=bounds_line_params,
                  setpoint_line_params=setpoints_line_params,
                  var_symbol="y",
                  var_label="Output",
                  data_label=data_label,
                  initial_text=initial_measurement_text,
                  control_text=control_text,
                  display_initial_text=display_initial_text,
                  display_control_text=display_control_text,
                  fontsize=fontsize,
                  legend_params=legend_params,
                  fig=fig,
                  bounds=y_bounds,
                  initial_steps=initial_steps,
                  plot_ylimits=y_plot_ylimits)
        
        # Remove duplicate labels from legend
        # if figure was created externally
        if is_ext_fig:
            remove_legend_duplicates(axis=axs_y[j],
                                     legend_params=legend_params,
                                     last_label=f'$y_{j + 1}^s$')
            
    # Show the plot if the figure was created internally
    if not is_ext_fig:
        plt.show()

def plot_data(
    axis: Axes,
    data: np.ndarray,
    setpoint: float,
    index: int,
    data_line_params: dict[str, Any],
    setpoint_line_params: dict[str, Any],
    bounds_line_params: dict[str, Any],
    var_symbol: str,
    var_label: str,
    data_label: str,
    initial_text: str,
    control_text: str,
    display_initial_text: bool,
    display_control_text: bool,
    fontsize: int,
    legend_params: dict[str, Any],
    fig: Figure,
    bounds: Optional[Tuple[float, float]] = None,
    initial_steps: Optional[int] = None,
    plot_ylimits: Optional[Tuple[float, float]] = None
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
        setpoint (float): The setpoint value for the data.
        index (int): The index of the data used for labeling purposes (e.g.,
            "u_1", "u_2").
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
        var_label (str): The variable label representing the control signal
            (e.g., "Input", "Output").
        data_label (str): The label for the current data sequence.
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
        fig (Figure): The Matplotlib figure object containing the axis.
        bounds (Optional[Tuple[float, float]]): A tuple (lower_bound,
            upper_bound) specifying the bounds of the data to be plotted. If
            provided, horizontal lines representing these bounds will be
            plotted. Defaults to `None`.
        initial_steps (Optional[int]): The number of initial time steps during
            which input-output measurements were taken for the data-driven
            characterization of the system. This highlights the initial
            measurement period in the plot. Defaults to `None`.
        plot_ylimits (Optional[Tuple[float, float]]): A tuple (lower_limit,
            upper_limit) specifying the Y-axis limits for the plot. If `None`,
            the Y-axis limits will be determined automatically. Defaults to
            `None`.
    """
    T = data.shape[0]  # Data length

    # Plot data series
    axis.plot(range(0, T),
              data,
              **data_line_params,
              label=f'${var_symbol}_{index + 1}${data_label}')
    # Plot setpoint
    axis.plot(range(0, T),
              np.full(T, setpoint),
              **setpoint_line_params,
              label=f'${var_symbol}_{index + 1}^s$')
    
    # Plot bounds if provided
    if bounds:
        lower_bound, upper_bound = bounds
        # Plot lower bound line
        axis.axhline(y=lower_bound,
                     **bounds_line_params,
                     label='Constraints')
        # Plot upper bound line
        axis.axhline(y=upper_bound,
                     **bounds_line_params)
    
    # Highlight initial input-output data measurement period if provided
    if initial_steps:
        # Highlight period with a grayed rectangle
        axis.axvspan(0, initial_steps, color='gray', alpha=0.1)
        # Add a vertical line at the right side of the rectangle
        axis.axvline(x=initial_steps, color='black',
                            linestyle=(0, (5, 5)), linewidth=1)
        
        # Display initial measurement text if enabled
        if display_initial_text:
            # Get y-axis limits
            y_min, y_max = axis.get_ylim()
            # Place label at the center of the highlighted area
            u_init_text = axis.text(
                initial_steps / 2, (y_min + y_max) / 2,
                initial_text, fontsize=fontsize - 1,
                ha='center', va='center', color='black',
                bbox=dict(facecolor='white', edgecolor='black'))
            # Get initial text bounding box width
            init_text_width = get_text_width_in_data(
                text_object=u_init_text, axis=axis, fig=fig)
            # Hide text box if it overflows the plot area
            if initial_steps < init_text_width:
                u_init_text.set_visible(False)
        
        # Display Data-Driven MPC control text if enabled
        if display_control_text:
            # Get y-axis limits
            y_min, y_max = axis.get_ylim()
            # Place label at the center of the remaining area
            u_control_text = axis.text(
                (T + initial_steps) / 2, (y_min + y_max) / 2,
                control_text, fontsize=fontsize - 1,
                ha='center', va='center', color='black',
                bbox=dict(facecolor='white', edgecolor='black'))
            # Get control text bounding box width
            control_text_width = get_text_width_in_data(
                text_object=u_control_text, axis=axis, fig=fig)
            # Hide text box if it overflows the plot area
            if (T - initial_steps) < control_text_width:
                u_control_text.set_visible(False)
    
    # Format labels, legend and ticks
    axis.set_xlabel('Time step $k$', fontsize=fontsize)
    axis.set_ylabel(f'{var_label} ${var_symbol}_{index + 1}$',
                    fontsize=fontsize)
    axis.legend(**legend_params)
    axis.tick_params(axis='both', labelsize=fontsize)
    
    # Set x-limits
    axis.set_xlim([0, T - 1])

    # Set y-limits if provided
    if plot_ylimits:
        axis.set_ylim(plot_ylimits)

def plot_input_output_animation(
    u_k: np.ndarray,
    y_k: np.ndarray,
    u_s: np.ndarray,
    y_s: np.ndarray,
    u_bounds_list: Optional[List[Tuple[float, float]]] = None,
    y_bounds_list: Optional[List[Tuple[float, float]]] = None,
    inputs_line_params: dict[str, Any] = {},
    outputs_line_params: dict[str, Any] = {},
    setpoints_line_params: dict[str, Any] = {},
    bounds_line_params: dict[str, Any] = {},
    initial_steps: Optional[int] = None,
    initial_steps_label: Optional[str] = None,
    continuous_updates: bool = False,
    initial_excitation_text: str = "Init. Excitation",
    initial_measurement_text: str = "Init. Measurement",
    control_text: str = "Data-Driven MPC",
    display_initial_text: bool = True,
    display_control_text: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    interval: float = 20.0,
    points_per_frame: int = 1,
    fontsize: int = 12,
    legend_params: dict[str, Any] = {},
    title: Optional[str] = None
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
        u_s (np.ndarray): An array of shape (m, 1) containing `m` input
            setpoint values.
        y_s (np.ndarray): An array of shape (p, 1) containing `p` output
            setpoint values.
        u_bounds_list (Optional[List[Tuple[float, float]]]): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each input data
            sequence. If provided, horizontal lines representing these bounds
            will be plotted in each subplot. If `None`, no horizontal lines
            will be plotted. The number of tuples must match the number of
            input data sequences. Defaults to `None`.
        y_bounds_list (Optional[List[Tuple[float, float]]]): A list of tuples
            (lower_bound, upper_bound) specifying bounds for each output data
            sequence. If provided, horizontal lines representing these bounds
            will be plotted in each subplot. If `None`, no horizontal lines
            will be plotted. The number of tuples must match the number of
            output data sequences. Defaults to `None`.
        inputs_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the lines used to plot the input data
            series (e.g., color, linestyle, linewidth).
        outputs_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the lines used to plot the output data
            series (e.g., color, linestyle, linewidth).
        setpoints_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the lines used to plot the setpoint
            values (e.g., color, linestyle, linewidth).
        bounds_line_params (dict[str, Any]): A dictionary of Matplotlib
            properties for customizing the lines used to plot the bounds of
            input-output data series (e.g., color, linestyle, linewidth). If
            not provided, Matplotlib's default line properties will be used.
        initial_steps (Optional[int]): The number of initial time steps during
            which input-output measurements were taken for the data-driven
            characterization of the system. This highlights the initial
            measurement period in the plot. If `None`, no special highlighting
            will be applied. Defaults to `None`.
        initial_steps_label (Optional[str]): Label text to use for the legend
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
        figsize (Tuple[int, int]): The (width, height) dimensions of the
            created Matplotlib figure.
        dpi (int): The DPI resolution of the figure.
        interval (float): The time between frames in milliseconds. Defaults
            to 20 ms.
        points_per_frame (int): The number of data points shown per animation
            frame. Increasing this value reduces the number of frames required
            to display all the data, resulting in faster data transitions.
            Defaults to 1.
        fontsize (int): The fontsize for labels and axes ticks.
        legend_params (dict[str, Any]): A dictionary of Matplotlib properties
            for customizing the plot legend (e.g., fontsize, loc,
            handlelength).
        title (Optional[str]): The title for the created plot figure. If
            `None`, no title will be displayed. Defaults to `None`.
    
    Returns:
        FuncAnimation: A Matplotlib `FuncAnimation` object that animates the
            progression of input-output data over time.

    Raises:
        ValueError: If any array dimensions mismatch expected shapes, or if
            the lengths of `u_bounds_list` or `y_bounds_list` do not match the
            number of subplots.
    """
    # Check input-output data dimensions
    if not (u_k.shape[0] == y_k.shape[0]):
        raise ValueError("Dimension mismatch. The number of time steps for "
                         "u_k and y_k do not match.")
    if not (u_k.shape[1] == u_s.shape[0] and y_k.shape[1] == y_s.shape[0]):
        raise ValueError("Dimension mismatch. The number of inputs from u_k "
                         "and u_s, and the number of outputs from y_k and "
                         "y_s should match.")
    
    # Retrieve number of input and output data sequences and their length
    m = u_k.shape[1]  # Number of inputs
    p = y_k.shape[1]  # Number of outputs
    T = u_k.shape[0]  # Length of data

    # Error handling for bounds list lengths
    if u_bounds_list and len(u_bounds_list) != m:
        raise ValueError(f"The length of `u_bounds_list` ("
                         f"{len(u_bounds_list)}) does not match the number "
                         f"of input subplots ({m}).")
    if y_bounds_list and len(y_bounds_list) != p:
        raise ValueError(f"The length of `y_bounds_list` ("
                         f"{len(y_bounds_list)}) does not match the number "
                         f"of output subplots ({p}).")

    # Create figure and subplots
    fig, axs_u, axs_y = create_input_output_figure(
        m=m, p=p, figsize=figsize, dpi=dpi, fontsize=fontsize, title=title)

    # Define input-output line lists
    u_lines: List[Line2D] = []
    y_lines: List[Line2D] = []

    # Define initial measurement rectangles and texts lists
    u_rects: List[Rectangle] = []
    u_right_rect_lines: List[Line2D] = []
    u_left_rect_lines: List[Line2D] = []
    u_init_texts: List[Text] = []
    u_control_texts: List[Text] = []
    y_rects: List[Rectangle] = []
    y_right_rect_lines: List[Line2D] = []
    y_left_rect_lines: List[Line2D] = []
    y_init_texts: List[Text] = []
    y_control_texts: List[Text] = []

    # Define y-axis center
    u_y_axis_centers: List[float] = []
    y_y_axis_centers: List[float] = []
        
    # Initialize input plot elements
    for i in range(m):
        # Get input bounds if provided
        u_bounds = u_bounds_list[i] if u_bounds_list else None
        initialize_data_animation(axis=axs_u[i],
                                  data=u_k[:, i],
                                  setpoint=u_s[i, :],
                                  index=i,
                                  data_line_params=inputs_line_params,
                                  bounds_line_params=bounds_line_params,
                                  setpoint_line_params=setpoints_line_params,
                                  var_symbol="u",
                                  var_label="Input",
                                  initial_text=initial_excitation_text,
                                  control_text=control_text,
                                  fontsize=fontsize,
                                  legend_params=legend_params,
                                  lines=u_lines,
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
                                  legend_loc='upper right')
    
    # Initialize output plot elements
    for j in range(p):
        # Get output bounds if provided
        y_bounds = y_bounds_list[i] if y_bounds_list else None
        initialize_data_animation(axis=axs_y[j],
                                  data=y_k[:, j],
                                  setpoint=y_s[j, :],
                                  index=j,
                                  data_line_params=outputs_line_params,
                                  bounds_line_params=bounds_line_params,
                                  setpoint_line_params=setpoints_line_params,
                                  var_symbol="y",
                                  var_label="Output",
                                  initial_text=initial_measurement_text,
                                  control_text=control_text,
                                  fontsize=fontsize,
                                  legend_params=legend_params,
                                  lines=y_lines,
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
                                  legend_loc='lower right')
    
    # Get initial text bounding box width
    init_text_width_input = get_text_width_in_data(
        text_object=u_init_texts[0], axis=axs_u[0], fig=fig)
    init_text_width_output = get_text_width_in_data(
        text_object=y_init_texts[0], axis=axs_y[0], fig=fig)
    
    # Calculate maximum text width between input and
    # output labels to show them at the same time
    init_text_width = max(init_text_width_input, init_text_width_output)
    
    # Get control text bounding box width
    control_text_width = get_text_width_in_data(
        text_object=u_control_texts[0], axis=axs_u[0], fig=fig)

    # Animation update function
    def update(frame):
        # Calculate the current index based on the number of points per frame,
        # ensuring it does not exceed the last valid data index
        current_index = min(frame * points_per_frame, T - 1)

        # Update input plot data
        for i in range(m):
            # Get lower boundary line of the initial measurement region
            # if continuous updates are enabled
            u_left_rect_line = (u_left_rect_lines[i]
                                if continuous_updates else None)
            update_data_animation(index=current_index,
                                  data=u_k[:current_index + 1, i],
                                  data_length=T,
                                  points_per_frame=points_per_frame,
                                  initial_steps=initial_steps,
                                  continuous_updates=continuous_updates,
                                  line=u_lines[i],
                                  rect=u_rects[i],
                                  y_axis_center=u_y_axis_centers[i],
                                  right_rect_line=u_right_rect_lines[i],
                                  left_rect_line=u_left_rect_line,
                                  init_text_obj=u_init_texts[i],
                                  control_text_obj=u_control_texts[i],
                                  display_initial_text=display_initial_text,
                                  display_control_text=display_control_text,
                                  init_text_width=init_text_width,
                                  control_text_width=control_text_width)
        
        # Update output plot data
        for j in range(p):
            # Get lower boundary line of the initial measurement region
            # if continuous updates are enabled
            y_left_rect_line = (y_left_rect_lines[j]
                                if continuous_updates else None)
            update_data_animation(index=current_index,
                                  data=y_k[:current_index + 1, j],
                                  data_length=T,
                                  points_per_frame=points_per_frame,
                                  initial_steps=initial_steps,
                                  continuous_updates=continuous_updates,
                                  line=y_lines[j],
                                  rect=y_rects[j],
                                  y_axis_center=y_y_axis_centers[j],
                                  right_rect_line=y_right_rect_lines[j],
                                  left_rect_line=y_left_rect_line,
                                  init_text_obj=y_init_texts[j],
                                  control_text_obj=y_control_texts[j],
                                  display_initial_text=display_initial_text,
                                  display_control_text=display_control_text,
                                  init_text_width=init_text_width,
                                  control_text_width=control_text_width)

        return (u_lines + y_lines + u_rects + u_right_rect_lines +
                u_left_rect_lines + u_init_texts + u_control_texts + y_rects +
                y_init_texts + y_control_texts + y_right_rect_lines +
                y_left_rect_lines)

    # Calculate the number of animation frames
    n_frames = math.ceil((T - 1) / points_per_frame)  + 1

    # Create animation
    animation = FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True)

    return animation

def initialize_data_animation(
    axis: Axes,
    data: np.ndarray,
    setpoint: float,
    index: int,
    data_line_params: dict[str, Any],
    setpoint_line_params: dict[str, Any],
    bounds_line_params: dict[str, Any],
    var_symbol: str,
    var_label: str,
    initial_text: str,
    control_text: str,
    fontsize: int,
    legend_params: dict[str, Any],
    lines: List[Line2D],
    rects: List[Rectangle],
    right_rect_lines: List[Line2D],
    left_rect_lines: List[Line2D],
    init_texts: List[Text],
    control_texts: List[Text],
    y_axis_centers: List[float],
    bounds: Optional[Tuple[float, float]] = None,
    initial_steps: Optional[int] = None,
    initial_steps_label: Optional[str] = None,
    continuous_updates: bool = False,
    legend_loc: str = 'best'
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
        setpoint (float): The setpoint value for the data.
        index (int): The index of the data used for labeling purposes (e.g.,
            "u_1", "u_2").
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
        lines (List[Line2D]): The list where the initialized plot lines will
            be stored.
        rects (List[Rectangle]): The list where the initialized rectangles
            representing the initial measurement region will be stored.
        right_rect_lines (List[Line2D]): The list where the initialized
            vertical lines representing the upper boundary of the initial
            measurement region will be stored.
        left_rect_lines (List[Line2D]): The list where the initialized
            vertical lines representing the lower boundary of the initial
            measurement region will be stored.
        init_texts (List[Text]): The list where the initialized initial
            measurement label texts will be stored.
        control_texts (List[Text]): The list where the initialized control
            label texts will be stored.
        y_axis_centers (List[float]): The list where the y-axis center from
            the adjusted axis will be stored.
        bounds (Optional[Tuple[float, float]]): A tuple (lower_bound,
            upper_bound) specifying the bounds of the data to be plotted. If
            provided, horizontal lines representing these bounds will be
            plotted. Defaults to `None`.
        initial_steps (Optional[int]): The number of initial time steps during
            which input-output measurements were taken for the data-driven
            characterization of the system. This highlights the initial
            measurement period in the plot. If `None`, no special highlighting
            will be applied. Defaults to `None`.
        initial_steps_label (Optional[str]): Label text to use for the legend
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

    # Initialize plot lines
    lines.append(axis.plot([], [],
                           **data_line_params,
                           label=f'${var_symbol}_{index + 1}$')[0])
    
    # Plot bounds if provided
    if bounds:
        lower_bound, upper_bound = bounds
        # Plot lower bound line
        axis.axhline(y=lower_bound,
                     **bounds_line_params,
                     label='Constraints')
        # Plot upper bound line
        axis.axhline(y=upper_bound,
                     **bounds_line_params)
    
    # Plot setpoint
    axis.plot(range(0, T), np.full(T, setpoint),
              **setpoint_line_params,
              label=f'${var_symbol}_{index + 1}^s$')

    # Define axis limits
    u_lim_min, u_lim_max = get_padded_limits(data, setpoint)
    axis.set_xlim([0, T - 1])
    axis.set_ylim(u_lim_min, u_lim_max)
    y_axis_centers.append((u_lim_min + u_lim_max) / 2)

    # Highlight initial input-output data measurement period if provided
    if initial_steps:
        # Initialize initial measurement rectangle
        rect = axis.axvspan(0, 0, color='gray', alpha=0.1)
        rects.append(rect)
       
        # Initialize initial measurement rectangle boundary lines
        right_rect_lines.append(axis.axvline(
            x=0, color='black', linestyle=(0, (5, 5)), linewidth=1))

        if continuous_updates:
            # Add left boundary line to show continuous updates, if enabled
            left_rect_lines.append(axis.axvline(
                x=0, color='black', linestyle=(0, (5, 5)), linewidth=1))
    
        # Initialize initial measurement text
        init_texts.append(axis.text(
            initial_steps / 2, y_axis_centers[index],
            initial_text, fontsize=fontsize - 1, ha='center',
            va='center', color='black', bbox=dict(facecolor='white',
                                                    edgecolor='black')))

        # Initialize control text
        control_texts.append(axis.text(
            (T + initial_steps) / 2, y_axis_centers[index],
            control_text, fontsize=fontsize - 1, ha='center',
            va='center', color='black', bbox=dict(facecolor='white',
                                                    edgecolor='black')))
        
    # Format labels and ticks
    axis.set_xlabel('Time step $k$', fontsize=fontsize)
    axis.set_ylabel(f'{var_label} ${var_symbol}_{index + 1}$', fontsize=fontsize)
    axis.tick_params(axis='both', labelsize=fontsize)

    # Collect all legend handles and labels
    handles, labels = axis.get_legend_handles_labels()
    custom_handlers = {}

    # Add rectangle to legend with custom handler if used
    if initial_steps and initial_steps_label:
        handles.append(rect)
        labels.append(initial_steps_label)

        # Add legend with custom handlers
        custom_handlers = {Rectangle: HandlerInitMeasurementRect()}
    
    # Format legend
    axis.legend(handles=handles,
                labels=labels,
                handler_map=custom_handlers,
                **legend_params,
                loc=legend_loc)

def update_data_animation(
    index: int,
    data: np.ndarray,
    data_length: int,
    points_per_frame: int,
    initial_steps: Optional[int],
    continuous_updates: bool,
    line: Line2D,
    rect: Rectangle,
    y_axis_center: float,
    right_rect_line: Line2D,
    left_rect_line: Optional[Line2D],
    init_text_obj: Text,
    control_text_obj: Text,
    display_initial_text: bool,
    display_control_text: bool,
    init_text_width: float,
    control_text_width: float
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
        data_length (int): The length of the `data` array.
        points_per_frame (int): The number of data points shown per animation
            frame.
        initial_steps (Optional[int]): The number of initial time steps during
            which input-output measurements were taken for the data-driven
            characterization of the system. This highlights the initial
            measurement period in the plot.
        continuous_updates (bool): Whether the initial measurement period
            highlight should move with the latest data to represent continuous
            input-output measurement updates.
        line (Line2D): The plot line corresponding to the data series plot.
        rect (Rectangle): The rectangle representing the initial measurement
            region.
        y_axis_center (float): The y-axis center of the plot axis.
        right_rect_line (Line2D): The line object representing the upper
            boundary of the initial measurement region.
        left_rect_line (Optional[Line2D]): The line object representing the
            lower boundary of the initial measurement region.
        init_text_obj (Text): The text object containing the initial
            measurement period label.
        control_text_obj (Text): The text object containing the control period
            label.
        display_initial_text (bool): Whether to display the `initial_text`
            label on the plot.
        display_control_text (bool): Whether to display the `control_text`
            label on the plot.
        init_text_width (float): The width of the `init_text_obj` object in
            data coordinates.
        control_text_width (float): The width of the `control_text_obj` object
            in data coordinates.
    """
    # Update plot line data
    line.set_data(range(0, index + 1), data[:index + 1])
    
    # Determine if an update is needed. Always update for continuous updates
    needs_update = (index <= initial_steps + points_per_frame or
                    continuous_updates)
    
    # Update initial measurement rectangle and texts
    if initial_steps and needs_update:
        # Calculate measurement period limit index
        lim_index = (min(index, initial_steps)
                     if not continuous_updates else index)
        
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
    file_path: str
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
    writer = FFMpegWriter(fps=fps,
                          metadata=dict(artist='Me'),
                          bitrate=bitrate)
    
    # Save animation while displaying a progress bar
    with tqdm(total=total_frames, desc="Saving animation") as pbar:
        animation.save(file_path,
                       writer=writer,
                       progress_callback=lambda i, n: pbar.update(1))

def get_padded_limits(
    X: np.ndarray,
    X_s: np.ndarray,
    pad_percentage: float = 0.05
) -> Tuple[float, float]:
    """
    Get the minimum and maximum limits from two data sequences extended by
    a specified percentage of the combined data range.

    Args:
        X (np.ndarray): First data array.
        X_s (np.ndarray): Second data array.
        pad_percentage (float, optional): The percentage of the data range
            to be used as padding. Defaults to 0.05.

    Returns:
        Tuple[float, float]: A tuple containing padded minimum and maximum 
            limits for the combined data from `X` and `X_s`.
    """
    # Get minimum and maximum limits from data sequences
    X_min, X_max = np.min(X), np.max(X)
    X_s_min, X_s_max = np.min(X_s), np.max(X_s)
    X_lim_min = min(X_min, X_s_min)
    X_lim_max = max(X_max, X_s_max)

    # Extend limits by a percentage of the overall data range
    X_range = X_lim_max - X_lim_min
    X_lim_min -= X_range * pad_percentage
    X_lim_max += X_range * pad_percentage

    return (X_lim_min, X_lim_max)

def get_text_width_in_data(
    text_object: Text,
    axis: Axes,
    fig: Figure
) -> float:
    """
    Calculate the bounding box width of a text object in data coordinates.

    Args:
        text_object (Text): A Matplotlib text object.
        axis (Axes): The axis on which the text object is displayed.
        fig (Figure): The Matplotlib figure object containing the axis.

    Returns:
        float: The width of the text object's bounding box in data
            coordinates.
    """
    # Get the bounding box of the text object in pixel coordinates
    text_box = text_object.get_window_extent(
        renderer=fig.canvas.get_renderer())
    # Convert the bounding box from pixel coordinates to data coordinates
    text_box_data = axis.transData.inverted().transform(text_box)
    # Calculate the width of the bounding box in data coordinates
    text_box_width = text_box_data[1][0] - text_box_data[0][0]
    
    return text_box_width

def remove_legend_duplicates(
    axis: Axes,
    legend_params: dict[str, Any],
    last_label: Optional[str] = None
) -> None:
    """
    Remove duplicate entries from the legend of a Matplotlib axis. Optionally,
    move a specified label to the end of the legend.

    Note:
        The appearance of the plot legend can be customized by passing a
        dictionary of Matplotlib legend properties.

    Args:
        axis (Axes): The Matplotlib axis containing the legend to modify.
        legend_params (dict[str, Any]): A dictionary of Matplotlib properties
            for customizing the plot legend (e.g., fontsize, loc,
            handlelength).
        last_label (Optional[str]): The label that should appear last in the
            legend. If not provided, no specific label is moved to the end.
            Defaults to `None`.
    """
    # Get labels and handles from axis without duplicates
    handles, labels = axis.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # If a last_label is provided and exists, move it to the end
    if last_label and last_label in by_label:
        last_handle = by_label.pop(last_label)
        by_label[last_label] = last_handle

    # Update the legend with the unique handles and labels
    axis.legend(by_label.values(), by_label.keys(), **legend_params)

def create_input_output_figure(
    m: int,
    p: int,
    figsize: Tuple[int, int],
    dpi: int,
    fontsize: int,
    title: Optional[str] = None
) -> Tuple[Figure, List[Axes], List[Axes]]:
    """
    Create a Matplotlib figure with two rows of subplots: one for control
    inputs and one for system outputs, and return the created figure and
    axes.

    If a title is provided, it will be set as the overall figure title.
    Each row of subplots will have its own title for 'Control Inputs' and
    'System Outputs'.

    Args:
        m (int): The number of control inputs (subplots in the first row).
        p (int): The number of system outputs (subplots in the second row).
        figsize (Tuple[int, int]): The (width, height) dimensions of the
            created Matplotlib figure.
        dpi (int): The DPI resolution of the figure.
        fontsize (int): The fontsize for suptitles.
        title (Optional[str]): The title for the overall figure. If `None`,
            no title will be added. Defaults to `None`.
    
    Returns:
        Tuple: A tuple containing:
            - Figure: The created Matplotlib figure.
            - List[Axes]: A list of axes for control inputs subplots.
            - List[Axes]: A list of axes for system outputs subplots.
    """
    # Create figure
    fig = plt.figure(
        num=title, layout='constrained', figsize=figsize, dpi=dpi)
    
    # Modify constrained layout padding
    fig.get_layout_engine().set(w_pad=0.1, h_pad=0.1, wspace=0.05, hspace=0)

    # Set overall figure title if provided
    if title:
        fig.suptitle(title, fontsize=fontsize + 3, fontweight='bold')
    
    # Create subfigures for input and output data plots
    subfigs = fig.subfigures(2, 1)

    # Add titles for input and output subfigures
    subfigs[0].suptitle('Control Inputs',
                        fontsize=fontsize + 2,
                        fontweight='bold')
    subfigs[1].suptitle('System Outputs',
                        fontsize=fontsize + 2,
                        fontweight='bold')
    
    # Create subplots
    axs_u = subfigs[0].subplots(1, m)
    axs_y = subfigs[1].subplots(1, p)

    # Ensure axs_u and axs_y are always lists
    if max(m, p) == 1:
        axs_u = [axs_u]
        axs_y = [axs_y]

    return fig, axs_u, axs_y