import math
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from plot_utils import get_x_ticks_positions, get_x_position


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def plot(labels: List[str], pandas_dataframes: List[pd.DataFrame], positions: Dict[str, Tuple[int, int]], group_labels=List[str], fig=None) -> plt.Figure:
    """
    :param labels:
    :param pandas_dataframes:
    :param positions:
    :param group_labels:
    :param fig:
    :return:
    """

    if len(labels) == 0:
        # Nothing to plot
        return None

    # Save current figure to restore later
    previous_figure = plt.gcf()

    # Set the current figure to fig
    # figsize = (int(len(labels) * 0.95), 6)
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    fig_width = 3 * 252 * inches_per_pt  # width in inches
    # fig_height = (fig_width * golden_mean)  # height in inches
    fig_height = 2.5
    figsize = [fig_width * 0.67, fig_height / 1.22]

    config_dpi = 100
    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=config_dpi)
    else:
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["figure.dpi"] = config_dpi

    plt.figure(fig.number)

    # change size and DPI of resulting figure
    # change legend font size
    plt.rcParams["legend.fontsize"] = 8
    # NOTE: Enabling this requires latex to be installed on the Github actions runner
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = 'serif'


    # plt.title('Runtime for Chi-Squared Test Benchmark', fontsize=10)
    plt.ylabel('Time [s]', labelpad=0)

    bar_width = 0.002
    spacer = 0.004
    inner_spacer = 0.0005
    # {\fontsize{30pt}{3em}\selectfont{}{Mean WRFv3.5 LHF\r}{\fontsize{18pt}{3em}\selectfont{}(September 16 - October 30, 2012)}

    x_center, x_start = get_x_ticks_positions(positions, bar_width, inner_spacer, spacer)

    #plt.yscale('log')

    plt.xticks(x_center, group_labels, fontsize=9)  # rotation='35',)
    # adds a thousand separator
    # fig.axes[0].get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    fig.axes[0].get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: human_format(x)))
    # add a grid
    ax = plt.gca()
    ax.grid(which='major', axis='y', linestyle=':')

    def ms_to_sec(num):
        return num / 1_000

    colors = ['#15607a', '#ffbd70', '#e7e7e7', '#ff483a']


    # Plot Bars
    max_y_value = 0
    for i in range(len(labels)):
        if not labels[i] in positions:
            raise ValueError(f"here {labels[i]}")

            continue

        else:
            x_pos = get_x_position(positions[labels[i]], x_start, bar_width, inner_spacer)

        df = pandas_dataframes[i]

        d1 = ms_to_sec(df['t_compile'].mean())
        d1_err = 0 if math.isnan(df['t_compile'].std()) else df['t_compile'].std()
        p1 = plt.bar(x_pos, d1, bar_width * 0.9, color=colors[0])



    max_y_rounded = (int(math.ceil(max_y_value / 10.0)) * 10) + 10
    # plt.yticks(np.arange(0, max_y_rounded, step=10))

    plt.yticks(fontsize=8)

    # Add Legend
    plt.legend(p1, ["Compile Time"], loc='upper left')


    # Restore current figure
    plt.figure(previous_figure.number)

    return fig
