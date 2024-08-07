import os
from pathlib import Path
import time
import pandas as pd
import pyarrow as pa
import argparse
import pyarrow.parquet as pq
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
    "--parquet_path",
    type=str,
    help="Path to the directory to store Parquet files",
    required=True,
)
args = parser.parse_args()

CSV_PATH = Path(args.csv_path)
PARQUET_PATH = Path(args.parquet_path)

# create directory if it doesn't exist
if not PARQUET_PATH.exists():
    PARQUET_PATH.mkdir(parents=True, exist_ok=True)

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
            parquet_file_path = PARQUET_PATH / (file.split(".")[0] + ".parquet")
            if parquet_file_path.exists():
                print(f"{parquet_file_path} already exists. Skipping conversion.")
                continue

            print(f"Converting {file} to parquet.")
            # read the csv file
            start_time = time.time()
            df = pd.read_csv(CSV_PATH / file, low_memory=False, names=columns, header=0)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_file_path)
            end_time = time.time()
            time_dict[file] = end_time - start_time


    # save the time taken to convert each file to a HDF5 file
    with open(FILE_DIR_PATH / "csv_to_parquet_times.txt", "w") as f:
        for file, time_taken in time_dict.items():
            f.write(f"{file}: {time_taken:.2f} seconds\n")
        mean_time = sum(time_dict.values()) / len(time_dict)
        std_time = (
            sum((x - mean_time) ** 2 for x in time_dict.values()) / len(time_dict)
        ) ** 0.5
        f.write(f"\nMean Time: {mean_time:.2f} seconds\n")
        f.write(f"Standard Deviation: {std_time:.2f} seconds\n")

if __name__ == "__main__":
    run_with_memory_log(main, FILE_DIR_PATH / "T1_csv_to_parquet_memory_log.txt")
