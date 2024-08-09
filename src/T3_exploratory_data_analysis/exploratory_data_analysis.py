import time
from pathlib import Path
import argparse

import pandas as pd
import seaborn as sns
import dask.dataframe as dd
import matplotlib.pyplot as plt

from bigdata.utils import read_files
from bigdata.utils import run_with_memory_log
from dask.distributed import LocalCluster, Client

FILE_PATH = Path(__file__).resolve()
FILE_DIR_PATH = FILE_PATH.parent

# Set up the directory for saving figures
fig_dir = FILE_DIR_PATH / "figs"
fig_dir.mkdir(exist_ok=True)

# Create argument parser
parser = argparse.ArgumentParser(description="EDA Script")

# Add arguments
parser.add_argument(
    "--base_path", type=str, help="Base path for file location", required=True
)
parser.add_argument("--file_format", type=str, help="File format", required=True)
parser.add_argument("--years", nargs="+", type=str, help="List of years")
parser.add_argument("--n_workers", type=int, help="Number of workers", required=True)
parser.add_argument(
    "--memory_limit", type=str, help="Memory limit for each worker", default=None
)

# Parse the arguments
args = parser.parse_args()

# Access the values
base_path = args.base_path
file_format = args.file_format
years = args.years
if years is None:
    years = list(range(2014, 2024))
n_workers = args.n_workers
memory_limit = args.memory_limit


def main():
    # Initialize list to store times for each plot
    plot_times = []

    # Function to save times to a .txt file
    def save_times_to_file(times, filename=f"{file_format}_plot_times.txt"):
        with open(filename, "w") as f:
            for i, t in enumerate(times):
                f.write(f"Plot {i+1}: {t:.2f} seconds\n")
            mean_time = sum(times) / len(times)
            std_time = (sum((x - mean_time) ** 2 for x in times) / len(times)) ** 0.5
            f.write(f"\nMean Time: {mean_time:.2f} seconds\n")
            f.write(f"Standard Deviation: {std_time:.2f} seconds\n")

    # Read the data
    ddf = (
        read_files(
            base_path,
            file_format,
            usecols=[
                "Registration State",
                "Vehicle Body Type",
                "Vehicle Year",
                "Violation Code",
                "Feet From Curb",
                "Issue Date",
                "Violation Precinct",
                "Issuing Agency",
                "Vehicle Make",
                "Violation Location",
                "Violation Time",
                "Violation Precinct",
            ],
            years=years,
        )
        .sample(frac=0.1)
        .persist()
    )

    # %%
    # Distribution of Registration States
    start_time = time.time()
    df = ddf[["Registration State"]].compute()
    df_reg_state = df["Registration State"].value_counts()
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="Registration State", order=df_reg_state.index)
    plt.title("Distribution of Registration States")
    plt.xticks(rotation=90)
    plt.savefig(fig_dir / f"(distribution_registration_states_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Distribution of 'Vehicle Body Type'
    start_time = time.time()
    df = ddf[["Vehicle Body Type"]].compute().reset_index(drop=True)
    df_reg_state = df["Vehicle Body Type"].value_counts()
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="Vehicle Body Type", order=list(df_reg_state.index[:10]))
    plt.title("Distribution of Vehicle Body Type")
    plt.xticks(rotation=90)
    plt.savefig(fig_dir / f"(distribution_vehicle_body_type_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Distribution of Vehicle Year
    start_time = time.time()
    df = ddf[["Vehicle Year"]]
    df["Vehicle Year"] = dd.to_numeric(df["Vehicle Year"], errors="coerce")
    df["Vehicle Year"] = df["Vehicle Year"].fillna(0).astype(int)
    df = df[df["Vehicle Year"] != 0]
    plt.figure(figsize=(12, 6))
    sns.histplot(df["Vehicle Year"].compute(), bins=30)
    plt.title("Distribution of Vehicle Year")
    plt.savefig(fig_dir / f"(distribution_vehicle_year_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Correlation Matrix
    start_time = time.time()
    df = ddf[["Vehicle Year", "Feet From Curb"]].compute().dropna()
    df_corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(fig_dir / f"(correlation_matrix_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Top 10 Most Frequent Violations
    start_time = time.time()
    df = ddf[["Violation Code"]].compute().reset_index(drop=True)
    df_violation_code = df["Violation Code"].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="Violation Code", order=df_violation_code.index)
    plt.title("Top 10 Most Frequent Violations")
    plt.xticks(rotation=90)
    plt.savefig(fig_dir / f"(top_10_most_frequent_violations_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Violations Over Time
    start_time = time.time()
    df = ddf[["Issue Date"]].compute()
    df["Issue Date"] = pd.to_datetime(df["Issue Date"], errors="coerce")
    df.dropna(inplace=True)
    violations_over_time = df["Issue Date"].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    violations_over_time.plot()
    plt.title("Violations Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Violations")
    plt.savefig(fig_dir / f"(violations_over_time_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Violations by Precinct
    start_time = time.time()
    df = ddf[["Violation Precinct"]].compute().reset_index(drop=True)
    df["Precinct"] = df["Violation Precinct"].values[:, 0]
    df_precinct = df["Precinct"].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="Violation Precinct", order=df_precinct.index)
    plt.title("Violations by Precinct")
    plt.xticks(rotation=90)
    plt.savefig(fig_dir / f"(violations_by_precinct_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Top 10 Vehicle Makes with Most Violations
    start_time = time.time()
    df = ddf[["Vehicle Make"]].compute().reset_index(drop=True)
    df_make = df["Vehicle Make"].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="Vehicle Make", order=df_make.index)
    plt.title("Top 10 Vehicle Makes with Most Violations")
    plt.xticks(rotation=90)
    plt.savefig(fig_dir / f"(top_10_vehicle_makes_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Violations by Issuing Agency
    start_time = time.time()
    df = ddf[["Issuing Agency"]].compute().reset_index(drop=True)
    df_agency = df["Issuing Agency"].value_counts()
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="Issuing Agency", order=df_agency.index)
    plt.title("Violations by Issuing Agency")
    plt.xticks(rotation=90)
    plt.savefig(fig_dir / f"(violations_by_issuing_agency_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Distribution of Violation Times
    start_time = time.time()
    df = ddf[["Violation Time"]].compute().reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="Violation Time", bins=24, kde=True)
    plt.title("Distribution of Violation Times")
    plt.savefig(fig_dir / f"(distribution_violation_times_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Heatmap of Violation Counts by Location and Precinct
    start_time = time.time()
    df = ddf[["Violation Precinct", "Violation Location"]].compute()
    df["Precinct"] = df["Violation Precinct"].values[:, 0]
    df.drop(columns=["Violation Precinct"], inplace=True)
    df = ddf.groupby(["Precinct", "Violation Location"]).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, cmap="coolwarm")
    plt.title("Heatmap of Violation Counts by Location and Precinct")
    plt.savefig(fig_dir / f"(heatmap_violation_counts_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Box Plot of Feet From Curb
    start_time = time.time()
    df = ddf[["Feet From Curb"]].compute()
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Feet From Curb")
    plt.title("Box Plot of Feet From Curb")
    plt.savefig(fig_dir / f"(box_plot_feet_from_curb_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # %%
    # Box Plot of Vehicle Year
    start_time = time.time()
    df = ddf[["Vehicle Year"]]
    df["Vehicle Year"] = dd.to_numeric(df["Vehicle Year"], errors="coerce")
    df["Vehicle Year"] = df["Vehicle Year"].fillna(0).astype(int)
    df = df[df["Vehicle Year"] != 0].compute()
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df["Vehicle Year"])
    plt.title("Box Plot of Vehicle Year")
    plt.savefig(fig_dir / f"(box_plot_vehicle_year_dask)_{file_format}.png")

    end_time = time.time()
    plot_times.append(end_time - start_time)

    # Save times to file
    save_times_to_file(plot_times)

    # %%


if __name__ == "__main__":
    if memory_limit is None:
        memory_limit = 64 / n_workers

    memory_string = f"{memory_limit}GiB"

    cluster = LocalCluster(n_workers=n_workers, memory_limit=memory_string)
    client = Client(cluster)
    run_with_memory_log(
        main,
        FILE_DIR_PATH
        / f"eda_{file_format}_n_workers_{n_workers}_memory_lim_{memory_limit*n_workers}_memory_log.txt",
    )
