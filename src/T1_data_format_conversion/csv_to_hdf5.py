import os
from pathlib import Path
import time
import pandas as pd
import argparse
from bigdata.utils import run_with_memory_log

FILE_PATH = Path(__file__).resolve()
FILE_DIR_PATH = FILE_PATH.parent

parser = argparse.ArgumentParser(description="Convert CSV files to Parquet format.")
parser.add_argument(
    "--csv_path",
    type=str,
    help="Path to the directory containing CSV files",
    required=True,
)
parser.add_argument(
    "--hdf5_path",
    type=str,
    help="Path to the directory to store HDF5 files",
    required=True,
)
args = parser.parse_args()

CSV_PATH = Path(args.csv_path)
HDF5_PATH = Path(args.hdf5_path)

# create directory if it doesn't exist
if not HDF5_PATH.exists():
    HDF5_PATH.mkdir(parents=True, exist_ok=True)

columns = [
    "Summons Number",
    "Plate ID",
    "Registration State",
    "Plate Type",
    "Issue Date",
    "Violation Code",
    "Vehicle Body Type",
    "Vehicle Make",
    "Issuing Agency",
    "Street Code1",
    "Street Code2",
    "Street Code3",
    "Vehicle Expiration Date",
    "Violation Location",
    "Violation Precinct",
    "Issuer Precinct",
    "Issuer Code",
    "Issuer Command",
    "Issuer Squad",
    "Violation Time",
    "Time First Observed",
    "Violation County",
    "Violation In Front Of Or Opposite",
    "House Number",
    "Street Name",
    "Intersecting Street",
    "Date First Observed",
    "Law Section",
    "Sub Division",
    "Violation Legal Code",
    "Days Parking In Effect",
    "From Hours In Effect",
    "To Hours In Effect",
    "Vehicle Color",
    "Unregistered Vehicle?",
    "Vehicle Year",
    "Meter Number",
    "Feet From Curb",
    "Violation Post Code",
    "Violation Description",
    "No Standing or Stopping Violation",
    "Hydrant Violation",
    "Double Parking Violation",
]


def main():
    time_dict = {}

    # loop over all the csv files in the directory
    for file in os.listdir(CSV_PATH):
        if file.endswith(".csv"):
            print(f"Converting {file} to HDF5.")
            # read the csv file
            start_time = time.time()
            df = pd.read_csv(
                CSV_PATH / file, low_memory=False, names=columns, header=0
            ).sample(frac=0.01)
            for col in df.columns:
                if col == "Issue Date" or col == "Vehicle Expiration Date":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].dtype == "object":
                    df[col] = df[col].astype("|S20")
                elif df[col].dtype == "float64":
                    df[col] = df[col].astype("float32")
            df.to_hdf(
                HDF5_PATH / (file.split(".")[0] + ".h5"),
                key="data",
                mode="w",
                format="table",
                data_columns=True,
            )
            end_time = time.time()
            time_dict[file] = end_time - start_time

    # save the time taken to convert each file to a HDF5 file
    with open(FILE_DIR_PATH / "csv_to_hdf5_times.txt", "w") as f:
        for file, time_taken in time_dict.items():
            f.write(f"{file}: {time_taken:.2f} seconds\n")
        mean_time = sum(time_dict.values()) / len(time_dict)
        std_time = (
            sum((x - mean_time) ** 2 for x in time_dict.values()) / len(time_dict)
        ) ** 0.5
        f.write(f"\nMean Time: {mean_time:.2f} seconds\n")
        f.write(f"Standard Deviation: {std_time:.2f} seconds\n")


if __name__ == "__main__":
    run_with_memory_log(main, FILE_DIR_PATH / "csv_to_hdf5_memory_log.txt")
