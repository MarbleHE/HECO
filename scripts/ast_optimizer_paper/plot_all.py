import pandas as pd
import random, csv, glob, os

from matplotlib import pyplot as plt

import plot_runtime
import plot_line
import plot_compiletime


output_filetypes = ['pdf', 'png']

def save_plot(fig: plt.Figure, filename: str, output_dir: str, use_tight_layout: bool = True):
    for fn, ext in zip(filename, output_filetypes):
        full_filename = f"{output_dir}/{filename}.{ext}"

        bbox_inches = 'tight' if (use_tight_layout and ext == 'pdf') else None
        pad_inches = 0 if (use_tight_layout and ext == 'pdf') else 0.1
        dpi = 300
        fig.savefig(full_filename, format=ext, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi)

        #dst_path_s3 = str(PurePosixPath(urlparse(root_folder).path) / 'plot' / full_filename)
        #upload_file_to_s3_bucket(full_filename, dst_path_s3)



def generate_dummy_data_runtime(data_dir):

    """
    Generates dummy data for the runtime (can be removed once actual data is here)
    -> for each app, system and size, there is a file in `data_dir`/runtime with the following name: {app}_{system}_{size}.csv
    """

    systems = ["heco", "expert", "porcupine", "naive"]

    vector_sizes = [8, 16, 32, 64, 128]
    apps = [("BoxBlur", vector_sizes), ("CardioRiskScore", [1]), ("ChiSquared", [1]), ("GxKernel", vector_sizes), ("GyKernel", vector_sizes), ("HammingDistance", vector_sizes), ("L2Distance", vector_sizes), ("LaplaceSharpening", vector_sizes),
            ("LinearKernel", vector_sizes), ("PolynomialKernel", vector_sizes), ("RobertsCross", vector_sizes)]



    # shuffle systems and apps such that difficulty level is not just in order
    #random.shuffle(systems)
    random.shuffle(apps)


    # remove all files
    files = glob.glob(f'{data_dir}/runtime/*')
    for f in files:
        os.remove(f)

    # generate new files
    for i_sys, system in enumerate(systems):
        for i_app, (app, vector_sizes) in enumerate(apps):

            for size in vector_sizes:

                with open(f'{data_dir}/runtime/{app}_{system}_{size}.csv', 'w', newline='') as csvfile:
                    fieldnames = ['Key Generation', 'Encryption', 'Computation', 'Decryption']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    #writer.writeheader()

                    n_reps = random.choice([1,2,3,4,5,6,7,8,9])

                    difficulty_factor = 100 * (2**i_sys) * (i_app+1) * size # calculate a difficulty level for the system and application
                        # -> use it to derive computation time

                    keygen = difficulty_factor * 12 # scale to be proportional to components
                    enc = difficulty_factor * 0.06
                    comp = difficulty_factor * 85
                    dec = difficulty_factor * 1.5

                    for rep in range(n_reps):

                        rep_factor = random.uniform(0.9, 1.1)

                        writer.writerow({'Key Generation': int(keygen*rep_factor), 'Encryption': int(enc*rep_factor), 'Computation': int(comp*rep_factor), 'Decryption': int(dec*rep_factor)})



def generate_dummy_data_compiletime(data_dir):

    """
    Generates dummy data for the compiletime (can be removed once actual data is here)
    -> for each app, system and size, there is a file in `data_dir`/compiletime with the following name: {app}_{system}_{size}.csv
    """
    systems = ["heco"]

    vector_sizes = [8, 16, 32, 64, 128]
    apps = [("CardioRiskScore", [1]), ("ChiSquared", [1]), ("BoxBlur", vector_sizes), ("GxKernel", vector_sizes), ("GyKernel", vector_sizes), ("HammingDistance", vector_sizes), ("L2Distance", vector_sizes), ("LaplaceSharpening", vector_sizes),
            ("LinearKernel", vector_sizes), ("PolynomialKernel", vector_sizes), ("RobertsCross", vector_sizes)]


    # remove all files
    files = glob.glob(f'{data_dir}/compiletime/*')
    for f in files:
        os.remove(f)

    # generate new files
    for i_sys, system in enumerate(systems):
        for i_app, (app, vector_sizes) in enumerate(apps):

            for size in vector_sizes:

                with open(f'{data_dir}/compiletime/{app}_{system}_{size}.csv', 'w', newline='') as csvfile:
                    fieldnames = ['Compile Time']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    n_reps = random.choice([1,2,3,4,5,6,7,8,9])

                    difficulty_factor = 100 * (2**i_sys) * (i_app+1) # calculate a difficulty level for the system and application
                        # -> use it to derive computation time

                    for rep in range(n_reps):

                        rep_factor = random.uniform(0.9, 1.1)

                        writer.writerow({'Compile Time': int(20 * difficulty_factor * rep_factor)})


def plot_all_compiletime(data_dir, output_dir):
    # TODO [nku] maybe include in other plot

    size = ["64"] # in this plot, only show the compiletimes for this vector size
    param_sizes = {
        "CardioRiskScore": ["1"], # fix -> does not have a size
        "ChiSquared": ["1"], # fix -> does not have a size
        "BoxBlur": size,
        "GxKernel": size,
        "GyKernel": size,
        "HammingDistance": size,
        "L2Distance": size,
        "LaplaceSharpening": size,
        "LinearKernel": size,
        "PolynomialKernel": size,
        "RobertsCross": size
    }

    labels, positions, group_labels, data = plot_runtime.extract_infos(folder=f'{data_dir}/compiletime', df_names=['t_compile'], param_sizes=param_sizes)
    fig = plot_compiletime.plot(labels, data, positions=positions, group_labels=group_labels)

    if fig:
        fig.show()

        save_plot(fig, "compiletime", output_dir)


def plot_all_runtime(data_dir, output_dir):
    size = ["64"] # in this plot, only show the runtimes for this vector size
    param_sizes = {
        "CardioRiskScore": ["1"], # fix -> does not have a size
        "ChiSquared": ["1"], # fix -> does not have a size
        "BoxBlur": size,
        "GxKernel": size,
        "GyKernel": size,
        "HammingDistance": size,
        "L2Distance": size,
        "LaplaceSharpening": size,
        "LinearKernel": size,
        "PolynomialKernel": size,
        "RobertsCross": size
    }

    labels, positions, group_labels, data = plot_runtime.extract_infos(folder=f'{data_dir}/runtime', df_names=['t_keygen', 't_input_encryption', 't_computation', 't_decryption'], param_sizes=param_sizes)


    fig = plot_runtime.plot(labels, data, positions=positions, group_labels=group_labels)
    fig.show()

    save_plot(fig, "runtime", output_dir)
    # TODO [nku] save figure as pdf in output_folder


def plot_all_line(data_dir, output_dir):
    df = plot_line.load_df(data_dir=data_dir)

    apps = ["RobertsCross", "HammingDistance"] # TODO [nku] select which apps to display

    approach_type_filter = {
        "naive": ["runtime"],
        "heco": ["runtime", "compiletime"],
        "expert": ["runtime"],
        "porcupine": ["runtime"]
    }

    for app in apps:
        print(f"Application: {app}")
        fig = plot_line.plot(df, app, approach_type_filter)
        fig.show()
        save_plot(fig, f"runtimeline_{app}", output_dir)

        # TODO [nku] save figure as pdf in output_folder


def ensure_dir_exists(dir):

    if not os.path.exists(dir):
        os.makedirs(dir)


def plot_all():

    data_dir = "./data"
    output_dir = "./out"

    ensure_dir_exists(f"{data_dir}/runtime")
    ensure_dir_exists(f"{data_dir}/compiletime")
    ensure_dir_exists(f"{output_dir}")


    # TODO [nku] these can be removed when actual data exists
    # generate_dummy_data_runtime(data_dir)
    # generate_dummy_data_compiletime(data_dir)

    plot_all_runtime(data_dir, output_dir)
    plot_all_compiletime(data_dir, output_dir)
    plot_all_line(data_dir, output_dir)


if __name__ == "__main__":
    plot_all()
