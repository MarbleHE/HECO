import pandas as pd
import glob, os, math
import matplotlib.pyplot as plt


def load_df(data_dir):

    data = []

    for t in ["runtime", "compiletime"]:

        # load run times
        files = glob.glob(f'{data_dir}/{t}/*')


        for f in sorted(files):
            label = os.path.basename(f).split(".")[0]

            app = label.split("_")[0]
            approach = label.split("_")[1]
            size = label.split("_")[2]


            if t == "runtime":
                df1 = pd.read_csv(f, names=['t_keygen', 't_input_encryption', 't_computation', 't_decryption'])

                df1["total_sec"] = (df1["t_keygen"] + df1["t_input_encryption"] + df1["t_computation"] + df1["t_decryption"]) / 1_000

                mean = df1["total_sec"].mean()
                std = 0 if math.isnan(df1['total_sec'].std()) else df1['total_sec'].std()
            elif t == "compiletime":
                df1 = pd.read_csv(f, names=['t_compile'])

                # convert ms to sec
                df1["t_compile_sec"] = df1["t_compile"]/ 1_000

                mean = df1["t_compile_sec"].mean()
                std = 0 if math.isnan(df1['t_compile_sec'].std()) else df1['t_compile_sec'].std()

            else:
                raise ValueError("unknown type")

            d = {"app": app, "approach": approach, "size": int(size), "type": t, "time_sec": mean, "time_err": std}

            data.append(d)



    # load compile times

    df = pd.DataFrame(data)
    df = df.sort_values(by=['app', 'approach', 'type', 'size'], ascending=True)

    return df



def plot(df, app, approach_type_filter, fig=None):


    # Save current figure to restore later
    previous_figure = plt.gcf()

    # Set the current figure to fig
    # figsize = (int(len(labels) * 0.95), 6)
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    fig_width = 252 * inches_per_pt  # width in inches

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

    colors = ['#15607a', '#ffbd70', '#e7e7e7', '#ff483a']

    colors_d = {
        "naive": '#15607a',
        "heco": '#ffbd70',
        "expert": '#e7e7e7',
        "porcupine": '#ff483a'
    }

    linestyles_d = {
        "runtime": "-",
        "compiletime": "--"
    }

    labels_d = {
        "runtime": "",
        "compiletime": "compile"
    }


    # plot the lineplots
    for approach, types in approach_type_filter.items():
        for t in types:

            df_cur = df[(df["app"]==app) & (df["approach"]==approach) & (df["type"]==t)]
            label = f"{approach} {labels_d[t]}"
            plt.errorbar(x=df_cur["size"], y=df_cur["time_sec"], yerr=df_cur["time_err"], label=label, color=colors_d[approach], linestyle=linestyles_d[t], capsize=5, marker=".")


    plt.yscale('log')
    plt.xscale('log', base=2)

    ax = plt.gca()
    ax.grid(which='major', axis='y', linestyle=':')

    #plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    plt.legend()

    plt.xlabel('Size')

    # Restore current figure
    plt.figure(previous_figure.number)

    return fig