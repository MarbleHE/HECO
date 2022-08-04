import math, glob, os
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


def extract_infos(folder, df_names, param_sizes=None):

    files = glob.glob(f'{folder}/*')

    data  = []
    labels = []

    print(param_sizes)

    for f in sorted(files):
        # extract label from path
        label = os.path.basename(f).split(".")[0]


        if param_sizes is not None:

            app = label.split("_")[0]
            size = label.split("_")[2]

            if size not in param_sizes[app]:
                # ignore unsused param sizes
                print(f"ignore file: label={label}")
                continue

        labels.append(label)

        df = pd.read_csv(f, names=df_names)
        data.append(df)

    groups = set()
    inner_groups = set()
    for label in labels:
        parts = label.split("_")
        groups.add(parts[0])
        inner_groups.add(parts[1])


    groups = sorted(list(groups))
    inner_groups = sorted(list(inner_groups))

    positions = {}

    for label in labels:
        g_idx = groups.index(label.split("_")[0])
        ig_idx = inner_groups.index(label.split("_")[1])
        positions[label] = (g_idx, ig_idx)

    group_labels = []
    for group in groups:
        g_labels = sorted(filter(lambda label: label.startswith(group), labels))
        g_inners = map(lambda l: l.split("_")[1], g_labels)

        group_label = f"{group}" #\n{{\\fontsize{{4pt}}{{3em}}\\selectfont{{}}({'/'.join(g_inners)})}}"
        group_labels.append(group_label)

    return labels, positions, group_labels, data


def plot(labels: List[str], pandas_dataframes: List[pd.DataFrame], positions: Dict[str, Tuple[int, int]], group_labels=List[str], fig=None) -> plt.Figure:
    """
    :param labels:
    :param pandas_dataframes:
    :param positions:
    :param group_labels:
    :param fig:
    :return:
    """

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

    plt.yscale('log')

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


    print(f"labels={labels}")

    # Plot Bars
    max_y_value = 0
    for i in range(len(labels)):
        if not labels[i] in positions:
            raise ValueError(f"here {labels[i]}")

            continue

        else:
            x_pos = get_x_position(positions[labels[i]], x_start, bar_width, inner_spacer)

        df = pandas_dataframes[i]

        d1 = ms_to_sec(df['t_keygen'].mean())
        d1_err = 0 if math.isnan(df['t_keygen'].std()) else df['t_keygen'].std()
        p1 = plt.bar(x_pos, d1, bar_width * 0.9, color=colors[0])
        d2 = ms_to_sec(df['t_input_encryption'].mean())
        d2_err = 0 if math.isnan(df['t_input_encryption'].std()) else df['t_input_encryption'].std()
        p2 = plt.bar(x_pos, d2, bar_width * 0.9, bottom=d1, color=colors[1])
        d3 = ms_to_sec(df['t_computation'].mean())
        d3_err = 0 if math.isnan(df['t_computation'].std()) else df['t_computation'].std()
        p3 = plt.bar(x_pos, d3, bar_width * 0.9, bottom=d1 + d2, color=colors[2])
        d4 = ms_to_sec(df['t_decryption'].mean())
        d4_err = 0 if math.isnan(df['t_decryption'].std()) else df['t_decryption'].std()
        total_err = ms_to_sec(d1_err + d2_err + d3_err + d4_err)
        max_y_value = d1 + d2 + d3 + d4 if (d1 + d2 + d3 + d4) > max_y_value else max_y_value
        p4 = plt.bar(x_pos, d4, bar_width * 0.9, ecolor='black', capsize=3, bottom=d1 + d2 + d3, color=colors[3])
        print(labels[i].replace('\n', ' '), ": \n", d1, '\t', d2, '\t', d3, '\t', d4, '\t( total: ', d1 + d2 + d3 + d4, ')')

        plt.bar_label(p4, labels=[labels[i].split("_")[1]], padding=3, rotation='vertical', fontsize=8)



    max_y_rounded = (int(math.ceil(max_y_value / 10.0)) * 10) + 10
    # plt.yticks(np.arange(0, max_y_rounded, step=10))

    plt.yticks(fontsize=8)



    # Add Legend
    plt.legend((p4[0], p3[0], p2[0], p1[0]),
               ('Decryption', 'Computation', 'Encryption', 'Key Generation'),
               loc='upper right', ncol=4)


    plt.ylim(0, max_y_value*20000) # TODO [nku] need to scale this a bit


    #plt.xlim(ax.get_xlim()[0], ax.get_xlim()[1]*1.2)


    # Restore current figure
    plt.figure(previous_figure.number)

    return fig
