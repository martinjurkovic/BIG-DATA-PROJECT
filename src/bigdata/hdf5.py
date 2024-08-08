import unicodedata

import numpy as np
import h5py


def clean_string(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def process_type(type):
    if type == np.int64:
        return "int"
    if type == np.dtype("O") or str(type) == "string":
        return "S20"
    if type == np.float64:
        return "float"
    if type == np.datetime64 or type == np.dtype("<M8[ns]"):
        return "S20"
    raise ValueError(f"Unknown type {type}")


def save_to_hdf5(path, df):
    data_types = [(name, process_type(type)) for name, type in df.dtypes.items()]
    for col, dtype in data_types:
        if dtype == "S20":
            df[col] = df[col].apply(clean_string)

    structured_array = np.core.records.fromarrays(
        df.values.T, names=",".join(df.columns), dtype=data_types
    )

    with h5py.File(path, "w") as hdf5_file:
        # Create a dataset for the table
        hdf5_file.create_dataset("table", data=structured_array)


def read_hdf5(path):
    with h5py.File(path, "r") as f:
        data = f["table"][()]
    return data
