import os
from pathlib import Path

import pandas as pd

cwd = Path.cwd()
path = str(cwd).split("BIG-DATA-PROJECT")
CSV_PATH = Path(path[0]) / "BIG-DATA-PROJECT/data/CSV/"

# create a directory to store the HDF5 files
HDF5_PATH = Path(path[0]) / "BIG-DATA-PROJECT/data/HDF5/"
if not os.path.exists(HDF5_PATH):
    os.makedirs(HDF5_PATH)

# loop over all the csv files in the directory
for file in os.listdir(CSV_PATH):
    if file.endswith('.csv'):
        print(f"Converting {file} to HDF5.")
        # read the csv file
        df = pd.read_csv(CSV_PATH / file, low_memory=False)
        df.to_hdf(HDF5_PATH / (file.split(".")[0] + ".h5"), key='df', mode='w')