import os
import time
import argparse

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask_ml.linear_model import LinearRegression
from dask_ml.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from bigdata.ml_utils import plot_results
from bigdata.utils import read_files, county_map, run_with_memory_log
from bigdata.augmentation_utils import read_data

# TODO: support all formats and add logging
args = argparse.ArgumentParser(description="High Ticket Days Prediction")
args.add_argument(
    "--format",
    type=str,
    help="Format of the data",
    choices=["csv", "parquet", "hdf5", "duckdb"],
    default="csv",
)

args = args.parse_args()
fmt = args.format

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
    columns = ["Issue Date", "Violation County"]

    ddf = read_files(
        os.path.join(DATA_DIR, fmt),
        file_format=fmt,
        usecols=columns,
        dtype={"Violation County": "str"},
        years=list(range(2014, 2024)),
    )
    end_time = time.time()
    print(f"Time to read data: {end_time - start_time:.2f} seconds")
    processing_times["Read Data"] = end_time - start_time

    # Data Augmentation
    start_time = time.time()

    def remap_county_codes(code):
        if str(code).upper() not in county_map.keys():
            return "-"
        return county_map[code.upper()]

    # Filter dates
    ddf["Issue Date"] = dd.to_datetime(ddf["Issue Date"], format="mixed")
    ddf = ddf.loc[(ddf["Issue Date"] >= START_DATE) & (ddf["Issue Date"] < END_DATE)]
    # Unify county codes
    ddf["Violation County"] = ddf["Violation County"].map(remap_county_codes)
    # Drop rows with missing county codes
    ddf = ddf.loc[ddf["Violation County"] != "-"]

    def reindex(df, date_column, resample_freq="D"):
        df[date_column] = dd.to_datetime(df[date_column])
        df = df.set_index(date_column).compute()
        df = df.resample(resample_freq).mean()
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

    ddf["tickets"] = 1
    df = ddf.groupby(["Issue Date", "Violation County"]).sum().compute().reset_index()

    county_dfs = []
    for county, aug_df in dfs.items():
        mask = df["Violation County"] == county
        merged = df.loc[mask].merge(
            aug_df, left_on="Issue Date", right_index=True, how="left"
        )
        county_dfs.append(merged)

    df = pd.concat(county_dfs)
    df.set_index("Issue Date", inplace=True)
    df.sort_index(inplace=True)

    X = pd.get_dummies(df, columns=["Violation County"]).astype(float)
    y = X.pop("tickets")

    # drop constant columns
    X = X.loc[:, X.apply(pd.Series.nunique) != 1]

    X_train, X_test = X.iloc[X.index < TEST_DATE], X.iloc[X.index >= TEST_DATE]
    y_train, y_test = y.iloc[X.index < TEST_DATE], y.iloc[X.index >= TEST_DATE]

    end_time = time.time()
    print(f"Time to augment data: {end_time - start_time:.2f} seconds")
    processing_times["Data Augmentation"] = end_time - start_time

    # XGBoost
    start_time = time.time()

    clf = xgb.XGBRegressor()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    plot_results(y_test, y_pred, "XGBoost", fmt=fmt, title=f"XGBoost RMSE: {rmse :.2f}")
    print("XGBoost RMSE: ", rmse)
    end_time = time.time()
    print(f"Time to train XGBoost: {end_time - start_time:.2f} seconds")
    processing_times["XGBoost"] = end_time - start_time
    model_performance["XGBoost"] = rmse

    # Dask Linear Regression
    start_time = time.time()

    # convert x_train to dask array
    X_train = dd.from_pandas(X_train, npartitions=10)
    y_train = dd.from_pandas(y_train, npartitions=10)
    X_test = dd.from_pandas(X_test, npartitions=10)
    y_test = dd.from_pandas(y_test, npartitions=10)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LinearRegression()
    clf.fit(X_train.to_dask_array(), y_train.to_dask_array())
    y_pred = clf.predict(X_test.to_dask_array())
    gt = y_test.compute()
    pred = y_pred.compute()

    rmse = np.sqrt(mean_squared_error(gt, pred))
    plot_results(
        gt, pred, "LinearRegression", fmt=fmt, title=f"Linear reg. RMSE: {rmse:.2f}"
    )
    print("Lin re. RMSE: ", rmse)
    end_time = time.time()
    print(f"Time to train Linear Regression: {end_time - start_time:.2f} seconds")
    processing_times["Linear Regression"] = end_time - start_time
    model_performance["Linear Regression"] = rmse

    # SGDRegressor with partial_fit
    start_time = time.time()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SGDRegressor()
    for partition in range(10):
        clf.partial_fit(
            X_train.get_partition(partition).compute(),
            y_train.get_partition(partition).compute(),
        )

    y_pred = clf.predict(X_test.compute())
    gt = y_test.compute()

    rmse = np.sqrt(mean_squared_error(gt, y_pred))
    plot_results(gt, y_pred, "SGD", fmt=fmt, title=f"Batch SGD RMSE: {rmse:.2f}")
    print("SGD RMSE: ", rmse)
    end_time = time.time()
    print(f"Time to train SGD: {end_time - start_time:.2f} seconds")
    processing_times["SGD"] = end_time - start_time
    model_performance["SGD"] = rmse

    # Save processing times
    times_log_path = os.path.join("logs", f"T3_{fmt}_times.txt")
    with open(times_log_path, "w") as f:
        for key, value in processing_times.items():
            print(f"{key :<17}: {value} seconds")
            f.write(f"{key :<17}: {value} seconds\n")

    # Save model performance
    performance_log_path = os.path.join("logs", f"T3_{fmt}_performance.txt")
    with open(performance_log_path, "w") as f:
        for key, value in model_performance.items():
            print(f"{key :<17}: {value}")
            f.write(f"{key :<17}: {value}\n")


if __name__ == "__main__":
    run_with_memory_log(
        main,
        os.path.join("logs", f"T3_{fmt}_memory_log.txt"),
    )
