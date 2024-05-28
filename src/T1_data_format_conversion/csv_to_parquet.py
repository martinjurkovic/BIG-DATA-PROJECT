import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

cwd = Path.cwd()
path = str(cwd).split("BIG-DATA-PROJECT")
CSV_PATH = Path(path[0]) / "BIG-DATA-PROJECT/data/CSV/"

# create a directory to store the parquet files
PARQUET_PATH = Path(path[0]) / "BIG-DATA-PROJECT/data/parquet/"
if not os.path.exists(PARQUET_PATH):
    os.makedirs(PARQUET_PATH)

columns = ['Summons Number', 'Plate ID', 'Registration State', 'Plate Type', 'Issue Date', 'Violation Code', 'Vehicle Body Type', 'Vehicle Make', 'Issuing Agency', 'Street Code1', 'Street Code2', 'Street Code3', 'Vehicle Expiration Date', 'Violation Location', 'Violation Precinct', 'Issuer Precinct', 'Issuer Code', 'Issuer Command', 'Issuer Squad', 'Violation Time', 'Time First Observed', 'Violation County', 'Violation In Front Of Or Opposite', 'House Number', 'Street Name', 'Intersecting Street', 'Date First Observed', 'Law Section', 'Sub Division', 'Violation Legal Code', 'Days Parking In Effect    ', 'From Hours In Effect', 'To Hours In Effect', 'Vehicle Color', 'Unregistered Vehicle?', 'Vehicle Year', 'Meter Number', 'Feet From Curb', 'Violation Post Code', 'Violation Description', 'No Standing or Stopping Violation', 'Hydrant Violation', 'Double Parking Violation']

# loop over all the csv files in the directory
for file in os.listdir(CSV_PATH):
    if file.endswith('.csv'):
        parquet_file_path = PARQUET_PATH / (file.split(".")[0] + ".parquet")
        if parquet_file_path.exists():
            print(f"{parquet_file_path} already exists. Skipping conversion.")
            continue
                                            

        print(f"Converting {file} to parquet.")
        # read the csv file
        df = pd.read_csv(CSV_PATH / file, low_memory=False, names = columns, header=0)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_file_path)