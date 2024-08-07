import os

from bigdata.augmentation_utils import read_data, save_data

FILE_PATH = __file__

DATA_DIR = FILE_PATH.split("src")[0] + "data"

os.makedirs(exist_ok=True, name=os.path.join(DATA_DIR, "parquet", "raw"))
os.makedirs(exist_ok=True, name=os.path.join(DATA_DIR, "hdf5", "raw"))
os.makedirs(exist_ok=True, name=os.path.join(DATA_DIR, "duckdb", "raw"))

# School data
school_data = read_data(
    os.path.join(DATA_DIR, "CSV", "raw", "school_locations.csv"),
    format="csv",
    dtype={
        "HighSchool_Network_Location_Code": "object",
        "HighSchool_Network_Name": "object",
        "HighSchool_Network_Superintendent": "object",
    },
    assume_missing=True,
)

school_data = school_data.compute()
school_data.to_parquet(
    os.path.join(DATA_DIR, "parquet", "raw", "school_locations.parquet")
)

school_data.to_hdf(
    os.path.join(DATA_DIR, "hdf5", "raw", "school_locations.hdf5"),
    key="data",
    format="table",
)

save_data(
    os.path.join(DATA_DIR, "duckdb", "raw", "school_locations.duckdb"),
    school_data,
    format="duckdb",
)

# Business data
business_data = read_data(
    os.path.join(DATA_DIR, "CSV", "raw", "business_locations.csv"),
    format="csv",
    dtype={
        "Address Building": "object",
        "BBL": "object",
        "BIN": "object",
        "Contact Phone Number": "object",
        "Detail": "object",
        "NTA": "object",
        "Secondary Address Street Name": "object",
        "Location": "object",
    },
    assume_missing=True,
)

business_data = business_data.compute()
business_data.to_parquet(
    os.path.join(DATA_DIR, "parquet", "raw", "business_locations.parquet")
)
business_data.to_hdf(
    os.path.join(DATA_DIR, "hdf5", "raw", "business_locations.hdf5"),
    key="data",
    format="table",
)
save_data(
    os.path.join(DATA_DIR, "duckdb", "raw", "business_locations.duckdb"),
    business_data,
    format="duckdb",
)

# Event Data
event_data = read_data(
    os.path.join(DATA_DIR, "CSV", "raw", "events.csv"),
    format="csv",
    dtype={"Street Closure Type": "object", "Event Street Side": "object"},
    assume_missing=True,
)

event_data = event_data.compute()
event_data.to_parquet(os.path.join(DATA_DIR, "parquet", "raw", "events.parquet"))
event_data.to_hdf(
    os.path.join(DATA_DIR, "hdf5", "raw", "events.hdf5"), key="data", format="table"
)
save_data(
    os.path.join(DATA_DIR, "duckdb", "raw", "events.duckdb"),
    event_data,
    format="duckdb",
)


# Landmark Data
landmark_data = read_data(
    os.path.join(DATA_DIR, "CSV", "raw", "landmarks.csv"),
    format="csv",
    dtype={"Street Closure Type": "object"},
    assume_missing=True,
)

landmark_data = landmark_data.compute()
landmark_data.to_parquet(os.path.join(DATA_DIR, "parquet", "raw", "landmarks.parquet"))
landmark_data.to_hdf(
    os.path.join(DATA_DIR, "hdf5", "raw", "landmarks.hdf5"), key="data", format="table"
)
save_data(
    os.path.join(DATA_DIR, "duckdb", "raw", "landmarks.duckdb"),
    landmark_data,
    format="duckdb",
)

# Weather Data
for borough in ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]:
    weather_data = read_data(
        os.path.join(DATA_DIR, "CSV", "raw", f"{borough}_weather.csv"),
        format="csv",
        assume_missing=True,
    )

    weather_data = weather_data.compute()
    weather_data.to_parquet(
        os.path.join(DATA_DIR, "parquet", "raw", f"{borough}_weather.parquet")
    )
    weather_data.to_hdf(
        os.path.join(DATA_DIR, "hdf5", "raw", f"{borough}_weather.hdf5"),
        key="data",
        format="table",
    )
    save_data(
        os.path.join(DATA_DIR, "duckdb", "raw", f"{borough}_weather.duckdb"),
        weather_data,
        format="duckdb",
    )
