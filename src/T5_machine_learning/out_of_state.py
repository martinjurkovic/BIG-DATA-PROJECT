import os
import time

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask_ml.linear_model import LogisticRegression
from dask_ml.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from bigdata.utils import read_csv_files, county_map
from bigdata.ml_utils import save_confusion_matrix

FILE_PATH = __file__
ROOT_DIR = FILE_PATH.split("src")[0]
DATA_DIR = ROOT_DIR + "data"

NPARTITIONS = 10
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
TEST_DATE = "2023-01-01"

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
    "Registration State": "object",
    "Plate Type": "object",
    "Issue Date": "object",
    "Violation Code": "object",
    "Vehicle Body Type": "object",
    "Vehicle Make": "object",
    "Issuing Agency": "object",
    "Violation Time": "object",
    "Violation County": "object",
    "Violation Legal Code": "object",
    "From Hours In Effect": "object",
    "To Hours In Effect": "object",
    "Unregistered Vehicle?": "object",
    "Vehicle Year": "float32",
    "Feet From Curb": "float32",
}

ddf = read_csv_files(
    os.path.join(DATA_DIR, "CSV"),
    usecols=columns,
    dtype=dtype,
    years=list(range(2014, 2024)),
)
ddf = ddf.sample(frac=0.001)  # TODO: remove this line
df = ddf.compute()

end_time = time.time()
print(f"Time to read data: {end_time - start_time:.2f} seconds")


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


df["Violation Time"] = df["Violation Time"].apply(get_hour).fillna(12)

df = df.loc[~df["Violation Time"].isna()]
df["Issue Time"] = pd.to_datetime(df["Issue Date"], format="mixed") + pd.to_timedelta(
    df["Violation Time"], unit="h"
)
df.drop(columns=["Violation Time", "Issue Date"], inplace=True)

df = df.loc[(df["Issue Time"] >= START_DATE) & (df["Issue Time"] < END_DATE)]


def aggregate_body_types(x: str):
    if x != x:
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


df["Vehicle Body Type"] = df["Vehicle Body Type"].apply(aggregate_body_types)
vc = df["Vehicle Body Type"].value_counts()
df.loc[df["Vehicle Body Type"].isin(vc[vc < 10000].index), "Vehicle Body Type"] = (
    "other"
)


def convert_vehicle_make(x: str):
    if x != x:
        return "other"
    elif x[0].isalpha():
        # this will cause collisions but we don't care
        return x.upper()[:1]
    else:
        return "other"


df["Vehicle Make"] = df["Vehicle Make"].apply(convert_vehicle_make)


def get_parking_hour(x: str):
    if x != x:
        return x
    if x == "ALL":
        return 24
    return get_hour(x)


df["From Hours In Effect"] = df["From Hours In Effect"].apply(get_hour).fillna(0)
df["To Hours In Effect"] = df["To Hours In Effect"].apply(get_hour).fillna(24)


def remap_county_codes(code):
    if code != code:
        return code
    if code.upper() not in county_map.keys():
        return float("nan")
    return county_map[code.upper()]


df["Violation County"] = df["Violation County"].apply(remap_county_codes)

df = df.loc[~df["Violation County"].isna()]


def reindex(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
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

county_dfs = []
for county, aug_df in dfs.items():
    mask = df["Violation County"] == county
    merged = df.loc[mask].merge(
        aug_df, left_on="Issue Time", right_index=True, how="left"
    )
    county_dfs.append(merged)

df = pd.concat(county_dfs)
del county_dfs

# convert Issue Time to yearly sine and cosine
df["month_sine"] = np.sin(2 * np.pi * df["Issue Time"].dt.month / 12)
df["month_cosine"] = np.cos(2 * np.pi * df["Issue Time"].dt.month / 12)
df["day_sine"] = np.sin(2 * np.pi * df["Issue Time"].dt.day / 31)
df["day_cosine"] = np.cos(2 * np.pi * df["Issue Time"].dt.day / 31)
df["hour_sine"] = np.sin(2 * np.pi * df["Issue Time"].dt.hour / 24)
df["hour_cosine"] = np.cos(2 * np.pi * df["Issue Time"].dt.hour / 24)

df.drop(columns=["Issue Time"], inplace=True)


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

X = pd.get_dummies(df, columns=categoricals, drop_first=True)
y = X.pop("Registration State")
X = X.fillna(0).astype(np.float32)
y = (y == "NY").astype(int)

# drop constant columns
X = X.loc[:, X.apply(pd.Series.nunique) != 1]

n = len(X)
train_size = int(n * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

end_time = time.time()
print(f"Time to augment data: {end_time - start_time:.2f} seconds")

# XGBoost
start_time = time.time()

clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

save_confusion_matrix(y_test, y_pred, "XGBoost")

print("XGBoost Accuracy: ", acc)
end_time = time.time()

print(f"Time to train XGBoost: {end_time - start_time:.2f} seconds")

# Dask Logistic Regression
start_time = time.time()

# TODO: handle number of partitions
# convert x_train to dask array
X_train = dd.from_pandas(X_train, npartitions=NPARTITIONS)
y_train = dd.from_pandas(y_train, npartitions=NPARTITIONS)
X_test = dd.from_pandas(X_test, npartitions=NPARTITIONS)
y_test = dd.from_pandas(y_test, npartitions=NPARTITIONS)

x_train_sd = X_train.std().compute()
const = []
for col, sd in x_train_sd.items():
    if sd == 0:
        print(col)
        const.append(col)

X_train = X_train.drop(columns=const)
X_test = X_test.drop(columns=const)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


clf = LogisticRegression()
clf.fit(X_train.to_dask_array(), y_train.to_dask_array())
y_pred = clf.predict(X_test.to_dask_array())
gt = y_test.compute()

acc = accuracy_score(gt, y_pred)
save_confusion_matrix(gt, y_pred, "LogisticRegression")

print("Logistic Regression Accuracy: ", acc)
end_time = time.time()

print(f"Time to train Logistic Regression: {end_time - start_time:.2f} seconds")

# Dask SGDClassifier with partial_fit
start_time = time.time()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SGDClassifier()

for partition in range(NPARTITIONS):
    clf.partial_fit(
        X_train.get_partition(partition).compute(),
        y_train.get_partition(partition).compute(),
        classes=[0, 1],
    )

y_pred = clf.predict(X_test.compute())
gt = y_test.compute()

acc = accuracy_score(gt, y_pred)
save_confusion_matrix(gt, y_pred, "SGDClassifier")

print("SGDClassifier Accuracy: ", acc)
end_time = time.time()
print(f"Time to train models: {end_time - start_time:.2f} seconds")
