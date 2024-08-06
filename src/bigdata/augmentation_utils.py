import dask.dataframe as dd


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
        return dd.read_hdf(path, key="data")
    elif format == "duckdb":
        raise NotImplementedError("DuckDB is not supported yet")
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_data(path, df, format):
    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "parquet":
        df.to_parquet(path, engine="pyarrow")
    elif format == "hdf5":
        df.to_hdf(path, key="data")
    elif format == "duckdb":
        raise NotImplementedError("DuckDB is not supported yet")
    else:
        raise ValueError(f"Unsupported format: {format}")
