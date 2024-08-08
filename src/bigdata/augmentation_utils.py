import dask.dataframe as dd
import duckdb
from pathlib import Path

from .hdf5 import save_to_hdf5, read_hdf5


def get_duckdb_connection(path):
    dir_path = Path(path).parent
    table_name = Path(path).name.split(".")[0]
    conn = duckdb.connect(str(dir_path / "nyc_database.db"))
    return conn, table_name


def read_data(path, format, dtype=None, **kwargs):
    if format == "csv":
        return dd.read_csv(
            path,
            dtype=dtype,
            **kwargs,
        )
    elif format == "parquet":
        return dd.read_parquet(
            path,
            engine="pyarrow",
            **kwargs,
        )
    elif format == "hdf5":
        return read_hdf5(path)
    elif format == "duckdb":
        conn, table_name = get_duckdb_connection(path.replace(" ", "_"))
        df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        return dd.from_pandas(df)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_data(path, df, format):
    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "parquet":
        df.to_parquet(path, engine="pyarrow")
    elif format == "hdf5":
        save_to_hdf5(path, df)
    elif format == "duckdb":
        conn, table_name = get_duckdb_connection(path.replace(" ", "_"))
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    else:
        raise ValueError(f"Unsupported format: {format}")
