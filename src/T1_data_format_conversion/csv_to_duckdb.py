from pathlib import Path
import argparse
import duckdb
import time
from bigdata.utils import run_with_memory_log

LOG_PATH = Path("logs").resolve()

parser = argparse.ArgumentParser(description="Convert CSV files to Parquet format.")
parser.add_argument(
    "--parquet_path",
    type=str,
    help="Path to the directory containing CSV files",
    required=True,
)
parser.add_argument(
    "--duckdb_path",
    type=str,
    help="Path to the directory to store HDF5 files",
    required=True,
)
args = parser.parse_args()

PARQUET_PATH = Path(args.parquet_path)
DUCKDB_PATH = Path(args.duckdb_path)

# create directory if it doesn't exist
if not DUCKDB_PATH.exists():
    DUCKDB_PATH.mkdir(parents=True, exist_ok=True)

def main():
    start_time = time.time()
    # Create a DuckDB connection
    conn = duckdb.connect(str(DUCKDB_PATH / "nyc_database.db"))
    # load the parquet files into conn
    conn.execute(f"CREATE TABLE nyc_data AS SELECT * FROM parquet_scan('{PARQUET_PATH / '*.parquet'}', union_by_name = true)")
    end_time = time.time()

    with open(LOG_PATH / "csv_to_duckdb_times.txt", "w") as f:
        f.write(f"Time to convert CSV files to DuckDB: {end_time - start_time:.2f} seconds\n, or per file: {(end_time - start_time) / len(list(PARQUET_PATH.glob('*.parquet'))):.2f} seconds\n")

if __name__ == "__main__":
    run_with_memory_log(main, LOG_PATH / "T1_csv_to_duckdb_memory_log.txt")