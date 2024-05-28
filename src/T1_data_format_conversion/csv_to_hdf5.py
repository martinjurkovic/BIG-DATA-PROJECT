import os
from pathlib import Path

import pandas as pd

cwd = Path.cwd()
path = str(cwd).split("BIG-DATA-PROJECT")
CSV_PATH = Path(path[0]) / "BIG-DATA-PROJECT/data/CSV/"

columns = ['Summons Number', 'Plate ID', 'Registration State', 'Plate Type', 'Issue Date', 'Violation Code', 'Vehicle Body Type', 'Vehicle Make', 'Issuing Agency', 'Street Code1', 'Street Code2', 'Street Code3', 'Vehicle Expiration Date', 'Violation Location', 'Violation Precinct', 'Issuer Precinct', 'Issuer Code', 'Issuer Command', 'Issuer Squad', 'Violation Time', 'Time First Observed', 'Violation County', 'Violation In Front Of Or Opposite', 'House Number', 'Street Name', 'Intersecting Street', 'Date First Observed', 'Law Section', 'Sub Division', 'Violation Legal Code', 'Days Parking In Effect    ', 'From Hours In Effect', 'To Hours In Effect', 'Vehicle Color', 'Unregistered Vehicle?', 'Vehicle Year', 'Meter Number', 'Feet From Curb', 'Violation Post Code', 'Violation Description', 'No Standing or Stopping Violation', 'Hydrant Violation', 'Double Parking Violation']

# create a directory to store the HDF5 files
HDF5_PATH = Path(path[0]) / "BIG-DATA-PROJECT/data/HDF5/"
if not os.path.exists(HDF5_PATH):
    os.makedirs(HDF5_PATH)

# loop over all the csv files in the directory
for file in os.listdir(CSV_PATH):
    if file.endswith('.csv'):
        print(f"Converting {file} to HDF5.")
        # read the csv file
        df = pd.read_csv(CSV_PATH / file, low_memory=False, names = columns, header=0)
        df.to_hdf(HDF5_PATH / (file.split(".")[0] + ".h5"), key='df', mode='w')