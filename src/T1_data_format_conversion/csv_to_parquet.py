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

# loop over all the csv files in the directory
for file in os.listdir(CSV_PATH):
    if file.endswith('.csv'):
        print(f"Converting {file} to parquet.")
        # read the csv file
        df = pd.read_csv(CSV_PATH / file, low_memory=False)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, PARQUET_PATH / (file.split(".")[0] + ".parquet"))