import os
import dask.dataframe as dd


def read_csv_files(base_path="../../data/csv", usecols=None, dtype=None, years=None):
    file_paths = []
    if years is None:
        file_paths = [
            os.path.join(base_path, file)
            for file in os.listdir(base_path)
            if file.endswith(".csv")
        ]
    else:
        year_strs = [str(year) for year in years]
        all_files = [file for file in os.listdir(base_path) if file.endswith(".csv")]
        filtered_files = [
            file
            for file in all_files
            if any(year_str in file for year_str in year_strs)
        ]
        if not filtered_files:
            raise ValueError("No files found for the specified years.")
        file_paths = [os.path.join(base_path, file) for file in filtered_files]
    ddf = dd.read_csv(file_paths, usecols=usecols, assume_missing=True, dtype=dtype)
    return ddf


county_map = {
    # Bronx
    "BX": "Bronx",
    "BRONX": "Bronx",
    # Brooklyn
    "BK": "Brooklyn",
    "BROOKLYN": "Brooklyn",
    "K": "Brooklyn",
    "KINGS": "Brooklyn",
    # Manhattan
    "MN": "Manhattan",
    "MANHATTAN": "Manhattan",
    "NY": "Manhattan",
    # Queens
    "Q": "Queens",
    "QS": "Queens",
    "QN": "Queens",
    "QNS": "Queens",
    "QUEENS": "Queens",
    "QUEEN": "Queens",
    # Staten Island
    "SI": "Staten Island",
    "ST": "Staten Island",
    "STATEN ISLAND": "Staten Island",
    "RICH": "Staten Island",
    "R": "Staten Island",
}
