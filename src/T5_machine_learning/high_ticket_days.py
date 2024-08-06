import os
import time

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask_ml.linear_model import LinearRegression
from dask_ml.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from bigdata.utils import read_csv_files, county_map

FILE_PATH = __file__
ROOT_DIR = FILE_PATH.split("src")[0]
DATA_DIR = ROOT_DIR + "data"

START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
TEST_DATE = "2023-01-01"


# Read the data
start_time = time.time()
columns = ["Issue Date", "Violation County"]

ddf = read_csv_files(
    os.path.join(DATA_DIR, "CSV"), usecols=columns, years=list(range(2014, 2024))
)
df = ddf.compute()
end_time = time.time()
print(f"Time to read data: {end_time - start_time:.2f} seconds")


# Data Augmentation
start_time = time.time()


def remap_county_codes(code):
    if code != code:
        return code
    if code.upper() not in county_map.keys():
        return float("nan")
    return county_map[code.upper()]


# Unify county codes
df["Violation County"] = df["Violation County"].apply(remap_county_codes)
# Drop rows with missing county codes
df = df.loc[~df["Violation County"].isna()]


def reindex(df, date_column, resample_freq="D"):
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    df = df.resample(resample_freq).mean()
    return df


def get_augmented_data_borough(borough):
    # Open Business Data
    busineses_df = pd.read_csv(
        os.path.join(DATA_DIR, "CSV", f"{borough}_business_openings.csv")
    )
    busineses_df = reindex(busineses_df, "datetime")

    # Events Data
    events_df = pd.read_csv(os.path.join(DATA_DIR, "CSV", f"{borough}_events.csv"))
    events_df = reindex(events_df, "datetime")

    # Landmarks Data
    landmarks_df = pd.read_csv(
        os.path.join(DATA_DIR, "CSV", f"{borough}_landmarks.csv")
    )
    landmarks_df = reindex(landmarks_df, "datetime")

    # School open Data
    schools_df = pd.read_csv(
        os.path.join(DATA_DIR, "CSV", f"{borough}_school_openings.csv")
    )
    schools_df = reindex(schools_df, "datetime")

    # Weather Data
    weather_df = pd.read_csv(os.path.join(DATA_DIR, "CSV", f"{borough}_weather.csv"))
    weather_df = reindex(weather_df, "time")

    return busineses_df, events_df, landmarks_df, schools_df, weather_df


def get_augmented_data(borough=None):
    holidays = weather_df = pd.read_csv(os.path.join(DATA_DIR, "CSV", "holidays.csv"))
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
        for borough in ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]:
            busineses, events, landmarks, schools, weather = get_augmented_data_borough(
                borough
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

df["tickets"] = 1
df = df.groupby(["Issue Date", "Violation County"]).sum()
df.reset_index(inplace=True)

df["Issue Date"] = pd.to_datetime(df["Issue Date"], format="mixed")
df = df.loc[(df["Issue Date"] >= START_DATE) & (df["Issue Date"] < END_DATE)]

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

# XGBoost
start_time = time.time()

clf = xgb.XGBRegressor()
clf.fit(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, clf.predict(X_test)))
print("XGBoost RMSE: ", rmse)
end_time = time.time()
print(f"Time to train XGBoost: {end_time - start_time:.2f} seconds")

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
print("Lin re. RMSE: ", rmse)
end_time = time.time()
print(f"Time to train Linear Regression: {end_time - start_time:.2f} seconds")

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
print("SGD RMSE: ", rmse)
end_time = time.time()
print(f"Time to train SGD: {end_time - start_time:.2f} seconds")
