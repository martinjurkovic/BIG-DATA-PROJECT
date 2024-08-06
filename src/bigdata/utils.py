import os
import dask.dataframe as dd
from pathlib import Path
import duckdb


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


def read_parquet_files(base_path="../../data/parquet", usecols=None, years=None):
    file_paths = []
    if years is None:
        file_paths = [
            os.path.join(base_path, file)
            for file in os.listdir(base_path)
            if file.endswith(".parquet")
        ]
    else:
        year_strs = [str(year) for year in years]
        all_files = [
            file for file in os.listdir(base_path) if file.endswith(".parquet")
        ]
        filtered_files = [
            file
            for file in all_files
            if any(year_str in file for year_str in year_strs)
        ]
        if not filtered_files:
            raise ValueError("No files found for the specified years.")
        file_paths = [os.path.join(base_path, file) for file in filtered_files]
    ddf = dd.read_parquet(file_paths, engine="pyarrow", columns=usecols)
    return ddf


def read_duckdb_files(base_path, usecols=None, years=None):
    """
    Reads data from DuckDB and filters based on specified columns and years using only SQL.

    Parameters:
    - columns: list of str or None, optional, list of column names to read from the dataset
    - years: list of int or None, optional, list of years to filter the data

    Returns:
    - Pandas DataFrame containing the queried data
    """
    # Create a DuckDB connection
    conn = duckdb.connect(str(Path(base_path) / "nyc_database.db"))
    # Handle columns with spaces
    columns = usecols
    if columns:
        columns = [f'"{col}"' for col in columns]
        columns_str = ", ".join(columns)
    else:
        columns_str = "*"

    # Construct the base SQL query
    query = f"SELECT {columns_str} FROM nyc_data"

    if years:
        # Create a string of years for SQL IN clause
        years_str = ", ".join(map(str, years))

        # Split 'Issue Date' on '/' and extract the year
        query += f"""
            WHERE SPLIT_PART("Issue Date", '/', 3)::INTEGER IN ({years_str})
        """

    # Execute the query and return the result as a DataFrame
    df = conn.execute(query).df()
    return df


def read_hdf5_files(base_path="../../data/hdf5", usecols=None, years=None):
    file_paths = []
    if years is None:
        file_paths = [
            os.path.join(base_path, file)
            for file in os.listdir(base_path)
            if file.endswith(".h5")
        ]
    else:
        year_strs = [str(year) for year in years]
        all_files = [file for file in os.listdir(base_path) if file.endswith(".h5")]
        filtered_files = [
            file
            for file in all_files
            if any(year_str in file for year_str in year_strs)
        ]
        if not filtered_files:
            raise ValueError("No files found for the specified years.")
        file_paths = [os.path.join(base_path, file) for file in filtered_files]
    ddf = dd.read_hdf(file_paths, key="data", columns=usecols)
    return ddf


def read_files(base_path, file_format, usecols=None, dtype=None, years=None):
    if file_format == "csv":
        return read_csv_files(base_path, usecols, dtype, years)
    elif file_format == "parquet":
        return read_parquet_files(base_path, usecols, years)
    elif file_format == "duckdb":
        return read_duckdb_files(base_path, usecols, years)
    elif file_format == "hdf5":
        return read_hdf5_files(base_path, usecols, years)
    raise ValueError(
        "Invalid file format. Please use one of ['csv', 'parquet', 'duckdb', 'hdf5']."
    )


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
