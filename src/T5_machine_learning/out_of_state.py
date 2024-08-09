import os
import time
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from dask_ml.linear_model import LogisticRegression
from dask.distributed import LocalCluster, Client
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from bigdata.utils import read_files, county_map, run_with_memory_log
from bigdata.ml_utils import save_confusion_matrix
from bigdata.augmentation_utils import read_data


parser = argparse.ArgumentParser(description="High Ticket Days Prediction")
parser.add_argument(
    "--format",
    type=str,
    help="Format of the data",
    choices=["csv", "parquet", "hdf5", "duckdb"],
    default="csv",
)
parser.add_argument("--n_workers", type=int, help="Number of workers", required=True)
parser.add_argument(
    "--memory_limit", type=str, help="Memory limit for each worker", default=None
)

args = parser.parse_args()
fmt = args.format
memory_limit = args.memory_limit
n_workers = args.n_workers

CHUNKSIZE = 10_000
FILE_PATH = __file__
ROOT_DIR = FILE_PATH.split("src")[0]
DATA_DIR = ROOT_DIR + "data"

START_DATE = "2014-01-01"
END_DATE = "2024-01-01"
TEST_DATE = "2023-01-01"


def main():
    processing_times = dict()
    model_performance = dict()

    # Read the data
    start_time = time.time()
    columns = [
        "Registration State",
        "Plate Type",
        "Issue Date",
        "Violation Time",
        "Violation County",
        "Violation Code",
        "Vehicle Body Type",
        "Vehicle Make",
        "Issuing Agency",
        "Unregistered Vehicle?",
        "Violation Legal Code",
        "From Hours In Effect",
        "To Hours In Effect",
        "Vehicle Year",
        "Feet From Curb",
    ]

    dtype = {
        "Registration State": "str",
        "Plate Type": "str",
        "Issue Date": "str",
        "Violation Code": "str",
        "Vehicle Body Type": "str",
        "Vehicle Make": "str",
        "Issuing Agency": "str",
        "Violation Time": "str",
        "Violation County": "str",
        "Violation Legal Code": "str",
        "From Hours In Effect": "str",
        "To Hours In Effect": "str",
        "Unregistered Vehicle?": "str",
        "Vehicle Year": "float32",
        "Feet From Curb": "float32",
    }

    ddf = read_files(
        os.path.join(DATA_DIR, fmt),
        file_format=fmt,
        usecols=columns,
        dtype=dtype,
        years=list(range(2014, 2024)),
    )

    end_time = time.time()
    print(f"Time to read data: {end_time - start_time:.2f} seconds")
    processing_times["Read Data"] = end_time - start_time

    # Data Augmentation
    start_time = time.time()

    def convert24(str1):
        # Checking if last two elements of time
        # is AM and first two elements are 12
        if str1[-1:] == "A" and str1[:2] == "12":
            return "00" + str1[2:-2]
        # remove the AM
        elif str1[-1:] == "A":
            return str1[:-1]
        # Checking if last two elements of time
        # is PM and first two elements are 12
        elif str1[-1:] == "P" and str1[:2] == "12":
            return str1[:-1]
        else:
            # add 12 to hours and remove PM
            return str(int(str1[:2]) + 12) + str1[2:-1]

    def round_hour(x):
        # 1123
        hours, minutes = divmod(int(x), 100)
        if minutes >= 30:
            hours += 1
        return hours

    def get_hour(x):
        try:
            return round_hour(convert24(x))
        except Exception:
            return float("nan")

    ddf["Violation Time"] = ddf["Violation Time"].apply(
        get_hour, meta=("Violation Time", "float64")
    )

    ddf = ddf.dropna(subset=["Violation Time"])
    ddf["Issue Time"] = dd.to_datetime(
        ddf["Issue Date"], format="mixed"
    ) + dd.to_timedelta(ddf["Violation Time"], unit="h")
    ddf = ddf.drop(columns=["Violation Time", "Issue Date"])

    ddf = ddf.loc[(ddf["Issue Time"] >= START_DATE) & (ddf["Issue Time"] < END_DATE)]

    def aggregate_body_types(x: str):
        if pd.isna(x):
            return "other"
        x = x.upper()
        # ordered by frequency for speed
        if x.startswith("S"):
            return (
                x.replace("SEDN", "SD")
                .replace("SDN", "SD")
                .replace("SEDA", "SD")
                .replace("SEDN", "SD")
                .replace("SUBN", "SU")
            )
        elif x.startswith("V"):
            return x.replace("VAN", "VN")
        elif x.startswith("2"):
            return x.replace("2D", "SD").replace("2S", "SD")
        elif x.startswith("4"):
            return x.replace("4D", "SD").replace("4S", "SD")
        elif x.startswith("M"):
            return x.replace("MOPD", "MP")
        elif x.startswith("P"):
            return x.replace("PICK", "PK")
        elif x.startswith("U"):
            return x.replace("UTIL", "UT")
        return x

    def filter_body_types(x: str):
        if x in [
            "SU",
            "SDSD",
            "VN",
            "DELV",
            "SD",
            "PK",
            "UT",
            "REFG",
            "TRAC",
            "CONV",
            "TAXI",
            "SW",
            "MCY",
            "BUS",
            "4 DR",
            "TRLR",
            "other",
            "WAGO",
            "MP",
            "TK",
        ]:
            return x
        return "other"

    ddf["Vehicle Body Type"] = ddf["Vehicle Body Type"].apply(
        aggregate_body_types, meta=("Vehicle Body Type", "str")
    )
    ddf["Vehicle Body Type"] = ddf["Vehicle Body Type"].apply(
        filter_body_types, meta=("Vehicle Body Type", "str")
    )

    def convert_vehicle_make(x: str):
        if x == "other":
            return x
        if x[0].isalpha():
            # this will cause collisions but we don't care
            return x.upper()[:1]
        else:
            return "other"

    ddf["Vehicle Make"] = ddf["Vehicle Make"].fillna("other")
    ddf["Vehicle Make"] = ddf["Vehicle Make"].apply(
        convert_vehicle_make, meta=("Vehicle Make", "str")
    )

    def get_parking_hour(x: str):
        if x == "ALL":
            return 24
        return get_hour(x)

    ddf["From Hours In Effect"] = (
        ddf["From Hours In Effect"]
        .apply(get_hour, meta=("From Hours In Effect", "float64"))
        .fillna(0)
    )
    ddf["To Hours In Effect"] = (
        ddf["To Hours In Effect"]
        .apply(get_hour, meta=("To Hours In Effect", "float64"))
        .fillna(24)
    )

    def remap_county_codes(code):
        if str(code).upper() not in county_map.keys():
            return "-"
        return county_map[code.upper()]

    ddf["Violation County"] = ddf["Violation County"].apply(
        remap_county_codes, meta=("Violation County", "str")
    )
    ddf = ddf.loc[ddf["Violation County"] != "-"]

    def reindex(df, date_column):
        df[date_column] = dd.to_datetime(df[date_column])
        df = df.set_index(date_column).compute()
        return df

    def get_augmented_data_borough(borough):
        busineses_df = read_data(
            os.path.join(
                DATA_DIR, fmt, "augmented", f"{borough}_business_openings.{fmt}"
            ),
            format=fmt,
        )
        busineses_df = reindex(busineses_df, "datetime")

        # Events Data
        events_df = read_data(
            os.path.join(DATA_DIR, fmt, "augmented", f"{borough}_events.{fmt}"),
            format=fmt,
        )
        events_df = reindex(events_df, "datetime")

        # Landmarks Data
        landmarks_df = read_data(
            os.path.join(DATA_DIR, fmt, "augmented", f"{borough}_landmarks.{fmt}"),
            format=fmt,
        )
        landmarks_df = reindex(landmarks_df, "datetime")

        # School open Data
        schools_df = read_data(
            os.path.join(
                DATA_DIR, fmt, "augmented", f"{borough}_school_openings.{fmt}"
            ),
            format=fmt,
        )
        schools_df = reindex(schools_df, "datetime")

        # Weather Data
        weather_df = read_data(
            os.path.join(DATA_DIR, fmt, "augmented", f"{borough}_weather.{fmt}"),
            format=fmt,
        )
        weather_df = reindex(weather_df, "time")

        return busineses_df, events_df, landmarks_df, schools_df, weather_df

    def get_augmented_data(borough=None):
        holidays = read_data(
            os.path.join(DATA_DIR, fmt, "augmented", f"holidays.{fmt}"),
            format=fmt,
        )
        holidays = reindex(holidays, "datetime")
        if borough:
            busineses_df, events_df, landmarks_df, schools_df, weather_df = (
                get_augmented_data_borough(borough)
            )
            # combine all data
            df = (
                busineses_df.join(events_df, how="outer")
                .join(landmarks_df, how="outer")
                .join(schools_df, how="outer")
                .join(weather_df, how="outer")
                .join(holidays, how="outer")
            )
            return df
        else:
            dfs = dict()
            for borough in [
                "Bronx",
                "Brooklyn",
                "Manhattan",
                "Queens",
                "Staten Island",
            ]:
                busineses, events, landmarks, schools, weather = (
                    get_augmented_data_borough(borough)
                )
                df = (
                    busineses.join(events, how="outer")
                    .join(landmarks, how="outer")
                    .join(schools, how="outer")
                    .join(weather, how="outer")
                    .join(holidays, how="outer")
                )
                dfs[borough] = df
            return dfs

    dfs = get_augmented_data()

    ddf = ddf.persist()
    county_ddfs = []
    for county, aug_df in dfs.items():
        county_ddf = ddf.loc[ddf["Violation County"] == county]
        county_ddf = county_ddf.merge(
            aug_df, left_on="Issue Time", right_index=True, how="left"
        )
        county_ddfs.append(county_ddf)

    ddf = dd.concat(county_ddfs)
    del county_ddfs

    # convert Issue Time to yearly sine and cosine
    ddf["month_sine"] = np.sin(2 * np.pi * ddf["Issue Time"].dt.month / 12)
    ddf["month_cosine"] = np.cos(2 * np.pi * ddf["Issue Time"].dt.month / 12)
    ddf["day_sine"] = np.sin(2 * np.pi * ddf["Issue Time"].dt.day / 31)
    ddf["day_cosine"] = np.cos(2 * np.pi * ddf["Issue Time"].dt.day / 31)
    ddf["hour_sine"] = np.sin(2 * np.pi * ddf["Issue Time"].dt.hour / 24)
    ddf["hour_cosine"] = np.cos(2 * np.pi * ddf["Issue Time"].dt.hour / 24)

    ddf = ddf.set_index("Issue Time")

    categoricals = [
        "Plate Type",
        "Violation Code",
        "Vehicle Body Type",
        "Vehicle Make",
        "Issuing Agency",
        "Violation County",
        "Violation Legal Code",
        "Unregistered Vehicle?",
    ]
    ddf = ddf.categorize(columns=categoricals)
    X = dd.get_dummies(ddf, columns=categoricals, drop_first=True).persist()
    y = X.pop("Registration State")
    X = X.fillna(0).astype(np.float32)
    y = (y == "NY").astype(int)

    X_train, X_test = X.loc[X.index < TEST_DATE], X.loc[X.index >= TEST_DATE]
    y_train, y_test = y.loc[X.index < TEST_DATE], y.loc[X.index >= TEST_DATE]

    # drop constant columns
    usecols = []
    stds = (X_train - X_train.mean(axis=0)).compute().std(axis=0)
    for col, std in stds.items():
        if std > 0:
            usecols.append(col)
    X_train = X_train[usecols]
    X_test = X_test[usecols]

    X_train, X_test = (
        X_train.to_dask_array().persist(),
        X_test.to_dask_array().persist(),
    )
    y_train, y_test = (
        y_train.to_dask_array().persist(),
        y_test.to_dask_array().persist(),
    )

    end_time = time.time()
    print(f"Time to augment data: {end_time - start_time:.2f} seconds")
    processing_times["Data Augmentation"] = end_time - start_time

    # XGBoost
    start_time = time.time()

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    gt = y_test.compute()
    acc = accuracy_score(gt, pred)

    save_confusion_matrix(gt, pred, "XGBoost")

    print("XGBoost Accuracy: ", acc)
    end_time = time.time()

    print(f"Time to train XGBoost: {end_time - start_time:.2f} seconds")
    processing_times["XGBoost"] = end_time - start_time
    model_performance["XGBoost"] = acc

    # Dask Logistic Regression
    start_time = time.time()

    scaler = StandardScaler()
    X_test.compute_chunk_sizes()  # need to do this to avoid a bug in dask-ml
    X_train = scaler.fit_transform(X_train).persist()
    X_test = scaler.transform(X_test).persist()

    clf = LogisticRegression(solver_kwargs={"normalize": False})
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test).compute().astype(int)
    gt = y_test.compute()

    acc = accuracy_score(gt, pred)
    save_confusion_matrix(gt, pred, "LogisticRegression")

    print("Logistic Regression Accuracy: ", acc)
    end_time = time.time()

    print(f"Time to train Logistic Regression: {end_time - start_time:.2f} seconds")
    processing_times["Logistic Regression"] = end_time - start_time
    model_performance["Logistic Regression"] = acc

    # Dask SGDClassifier with partial_fit
    start_time = time.time()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train.compute_chunk_sizes()
    y_train.compute_chunk_sizes()
    X_train = X_train.rechunk((CHUNKSIZE, X_train.shape[1])).persist()
    y_train = y_train.rechunk(CHUNKSIZE).persist()

    clf = SGDClassifier()

    for i in range(X_train.blocks.shape[0]):
        clf.partial_fit(
            X_train.blocks[i].compute(), y_train.blocks[i].compute(), classes=[0, 1]
        )

    pred = clf.predict(X_test.compute())
    gt = y_test.compute()

    acc = accuracy_score(gt, pred)
    save_confusion_matrix(gt, pred, "SGDClassifier")

    print("SGDClassifier Accuracy: ", acc)
    end_time = time.time()
    print(f"Time to train models: {end_time - start_time:.2f} seconds")
    processing_times["SGDClassifier"] = end_time - start_time
    model_performance["SGDClassifier"] = acc

    # Save processing times
    times_log_path = os.path.join("logs", f"T5a_{fmt}_times.txt")
    with open(times_log_path, "w") as f:
        for key, value in processing_times.items():
            print(f"{key :<19}: {value} seconds")
            f.write(f"{key :<19}: {value} seconds\n")

    # Save model performance
    performance_log_path = os.path.join("logs", f"T5a_{fmt}_performance.txt")
    with open(performance_log_path, "w") as f:
        for key, value in model_performance.items():
            print(f"{key :<19}: {value}")
            f.write(f"{key :<19}: {value}\n")


if __name__ == "__main__":
    if memory_limit is None:
        memory_limit = 64 / n_workers

    memory_string = f"{memory_limit}GiB"

    cluster = LocalCluster(n_workers=n_workers, memory_limit=memory_string)
    client = Client(cluster)
    run_with_memory_log(
        main,
        os.path.join(
            "logs", f"T5a_{fmt}_memory_lim_{memory_limit*n_workers}_memory_log.txt"
        ),
    )
