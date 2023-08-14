import pandas as pd
import pathlib, os

from matplotlib import pyplot as plt

import plot_comparison
import plot_benchmark

output_filetypes = ["pdf", "png"]


def save_plot(
    fig: plt.Figure, filename: str, output_dir: str, use_tight_layout: bool = True
):
    for fn, ext in zip(filename, output_filetypes):
        full_filename = f"{output_dir}/{filename}.{ext}"

        bbox_inches = "tight" if (use_tight_layout and ext == "pdf") else None
        pad_inches = 0 if (use_tight_layout and ext == "pdf") else 0.1
        dpi = 300
        fig.savefig(
            full_filename,
            format=ext,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            dpi=dpi,
        )


def plot_all_comparison(data_dir, output_dir):
    param_sizes = {
        "BoxBlur": ["64"],
        "DotProduct": ["8"],
        "GxKernel": ["64"],
        "HammingDistance": ["4"],
        "L2Distance": ["4"],
        "LinearPolynomial": ["64"],
        "QuadraticPolynomial": ["64"],
        "RobertsCross": ["64"],
    }

    labels, positions, group_labels, data = plot_comparison.extract_infos(
        folder=f"{data_dir}/comparison",
        df_names=["t_keygen", "t_input_encryption", "t_computation", "t_decryption"],
        param_sizes=param_sizes,
    )

    fig = plot_comparison.plot(
        labels, data, positions=positions, group_labels=group_labels
    )
    fig.show()

    save_plot(fig, "comparison", f"{output_dir}/comparison")


def plot_all_benchmark_rc(data_dir, output_dir):
    param_sizes = {
        "RobertsCross": ["2", "4", "8", "16", "32", "64"],
        "HammingDistance": [],
    }

    labels, positions, group_labels, data = plot_benchmark.extract_infos(
        folder=f"{data_dir}/benchmark",
        df_names=["t_keygen", "t_input_encryption", "t_computation", "t_decryption"],
        param_sizes=param_sizes,
    )

    fig = plot_benchmark.plot(
        labels, data, positions=positions, group_labels=group_labels
    )
    fig.show()

    save_plot(fig, "benchmark_rc", f"{output_dir}/benchmark")


def plot_all_benchmark_hd(data_dir, output_dir):
    param_sizes = {
        "RobertsCross": [],
        "HammingDistance": ["4", "16", "64", "256", "1024"],
    }

    labels, positions, group_labels, data = plot_benchmark.extract_infos(
        folder=f"{data_dir}/benchmark",
        df_names=["t_keygen", "t_input_encryption", "t_computation", "t_decryption"],
        param_sizes=param_sizes,
    )

    fig = plot_benchmark.plot(
        labels, data, positions=positions, group_labels=group_labels
    )
    fig.show()

    save_plot(fig, "benchmark_hd", f"{output_dir}/benchmark")


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def plot_all():
    script_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = f"{script_dir}/data"
    output_dir = f"{script_dir}/out"

    ensure_dir_exists(f"{data_dir}/benchmark")
    ensure_dir_exists(f"{output_dir}/benchmark")

    ensure_dir_exists(f"{data_dir}/comparison")
    ensure_dir_exists(f"{output_dir}/comparison")

    plot_all_comparison(data_dir, output_dir)
    plot_all_benchmark_rc(data_dir, output_dir)
    plot_all_benchmark_hd(data_dir, output_dir)


if __name__ == "__main__":
    plot_all()
