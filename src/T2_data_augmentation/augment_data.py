import os
import time
import argparse

import holidays
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from tqdm import tqdm
import geopandas
from geodatasets import get_path
import matplotlib.pyplot as plt
import dask.dataframe as dd

from bigdata.augmentation_utils import read_data, save_data
from bigdata.utils import run_with_memory_log

FILE_PATH = __file__

DATA_DIR = FILE_PATH.split("src")[0] + "data"


def parse_args():
    args = argparse.ArgumentParser(description="Convert CSV files to Parquet format.")
    args.add_argument(
        "--format",
        type=str,
        help="Format of the data",
        choices=["csv", "parquet", "hdf5", "duckdb"],
        default="hdf5",
    )
    return args.parse_args()


args = parse_args()
fmt = args.format


def main():
    os.makedirs(exist_ok=True, name=os.path.join(DATA_DIR, fmt, "augmented"))
    processing_times = dict()

    # Load the school data
    start_time = time.time()
    school_data = read_data(
        os.path.join(DATA_DIR, fmt, "raw", f"school_locations.{fmt}"),
        format=fmt,
        dtype={
            "HighSchool_Network_Location_Code": "object",
            "HighSchool_Network_Name": "object",
            "HighSchool_Network_Superintendent": "object",
        },
        assume_missing=True,
    )

    # Convert dates
    school_data["open_date"] = dd.to_datetime(school_data["open_date"], errors="coerce")
    school_data = school_data.dropna(subset=["open_date", "LONGITUDE", "LATITUDE"])

    # Filter closed schools
    school_data["Status_descriptions"] = school_data["Status_descriptions"].fillna(
        "Open"
    )
    school_data = school_data[school_data["Status_descriptions"] == "Open"]
    school_data = school_data.drop(columns=["Status_descriptions"])
    school_data = school_data[
        ["open_date", "LONGITUDE", "LATITUDE", "Location_Category_Description"]
    ]

    boroughs = geopandas.read_file(get_path("nybb")).to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(10, 10))
    boroughs.plot("BoroName", legend=True, ax=ax, cmap="viridis")

    for i, row in boroughs.iterrows():
        ax.annotate(
            row["BoroName"],
            xy=row["geometry"].representative_point().coords[0],
            ha="center",
            color="red",
            fontsize=14,
        )

    colors = list(plt.get_cmap("tab10").colors)
    categories = school_data["Location_Category_Description"].compute()
    for category in categories.unique():
        school_data.compute()[categories == category].plot.scatter(
            x="LONGITUDE",
            y="LATITUDE",
            alpha=0.5,
            ax=ax,
            color=colors.pop(),
            label=category,
        )

    ax.legend(loc="upper left")
    plt.savefig("figs/schools.png")
    plt.close()

    school_data["Borough"] = None
    school_data = school_data.compute()
    for i, row in boroughs.iterrows():
        geom = boroughs.at[i, "geometry"]
        mask = geom.contains(
            geopandas.points_from_xy(school_data["LONGITUDE"], school_data["LATITUDE"])
        )
        school_data.loc[mask, "Borough"] = boroughs.at[i, "BoroName"]

    school_data = dd.from_pandas(school_data, npartitions=10)
    # school opening times based on
    # https://www.aaastateofplay.com/the-average-school-start-times-in-every-state/
    # are approx. at 8:00 am for NY

    # school durations based on
    # https://nces.ed.gov/programs/statereform/tab5_14.asp
    # rounded down as we are using hourly data

    five_hour = [
        "Elementary" "Junior High-Intermediate-Middle",
        "K-8",
        "K-12 all grades",
        "Early Childhood",
        "Ungraded",
    ]
    six_hour = [
        "High school",
        "Secondary School",
        "High School",
    ]

    def get_schools_hourly_df(school_data, borough):
        # Generate hourly datetime range from 2013 to 2024
        hourly_dates = pd.date_range(
            start="2013-01-01", end="2024-12-31 23:00:00", freq="h"
        )
        hourly_df = pd.DataFrame(hourly_dates, columns=["datetime"])

        school_data = school_data[school_data["Borough"] == borough].compute()
        hourly_df["open_schools"] = 0
        school_months = hourly_df["datetime"].dt.month.isin(
            [9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
        )
        for school_type in five_hour:
            open_schools = len(
                school_data[school_data["Location_Category_Description"] == school_type]
            )
            hourly_df.loc[
                school_months
                & (hourly_df["datetime"].dt.hour >= 8)
                & (hourly_df["datetime"].dt.hour < 13),
                "open_schools",
            ] += open_schools

        for school_type in six_hour:
            open_schools = len(
                school_data[school_data["Location_Category_Description"] == school_type]
            )
            hourly_df.loc[
                school_months
                & (hourly_df["datetime"].dt.hour >= 8)
                & (hourly_df["datetime"].dt.hour < 14),
                "open_schools",
            ] += open_schools

        return hourly_df

    print("Processing Shool data")
    for borough in boroughs["BoroName"]:
        print(f"{borough}")
        hourly_df = get_schools_hourly_df(school_data, borough)
        save_data(
            os.path.join(
                DATA_DIR, fmt, "augmented", f"{borough}_school_openings.{fmt}"
            ),
            hourly_df,
            fmt,
        )
    print()
    end_time = time.time()
    processing_times["Schools"] = end_time - start_time

    # Holiday data
    start_time = time.time()
    cal = USFederalHolidayCalendar()

    # Define function to categorize holidays
    def get_holiday_categories(date):
        us_holidays = holidays.US(years=date.year)
        ny2_holidays = holidays.US(subdiv="NY", years=date.year)

        last_day_of_april = pd.Timestamp(year=date.year, month=4, day=30)
        arbor_day = last_day_of_april - pd.DateOffset(
            days=(last_day_of_april.weekday() - 4) % 7
        )
        ny_holidays = {
            "Lincoln's Birthday": pd.Timestamp(year=date.year, month=2, day=12),
            "Election Day": pd.Timestamp(
                year=date.year,
                month=11,
                day=(1 + (2 + date.replace(day=1).weekday()) % 7) + 1,
            )
            if date.year % 2 == 0
            else None,
            "Juneteenth": pd.Timestamp(year=date.year, month=6, day=19),
            "Valentine's Day": pd.Timestamp(year=date.year, month=2, day=14),
            "St. Patrick's Day": pd.Timestamp(year=date.year, month=3, day=17),
            "April Fool's Day": pd.Timestamp(year=date.year, month=4, day=1),
            "Mother's Day": pd.Timestamp(
                year=date.year,
                month=5,
                day=8
                + (6 - pd.Timestamp(year=date.year, month=5, day=1).weekday()) % 7,
            ),
            "Father's Day": pd.Timestamp(
                year=date.year,
                month=6,
                day=15
                + (6 - pd.Timestamp(year=date.year, month=6, day=1).weekday()) % 7,
            ),
            "Flag Day": pd.Timestamp(year=date.year, month=6, day=14),
            "Halloween": pd.Timestamp(year=date.year, month=10, day=31),
            "Groundhog Day": pd.Timestamp(year=date.year, month=2, day=2),
            "Arbor Day": arbor_day,
            "Patriot Day": pd.Timestamp(year=date.year, month=9, day=11),
            "Constitution Day": pd.Timestamp(year=date.year, month=9, day=17),
            "Christmas Eve": pd.Timestamp(year=date.year, month=12, day=24),
            "New Year's Eve": pd.Timestamp(year=date.year, month=12, day=31),
        }

        religious_holidays = {
            "Hanukkah": pd.Timestamp(year=date.year, month=12, day=10),  # Example date
            "Ramadan": pd.Timestamp(year=date.year, month=4, day=12),  # Example date
            "Eid al-Fitr": pd.Timestamp(
                year=date.year, month=5, day=12
            ),  # Example date
            "Eid al-Adha": pd.Timestamp(
                year=date.year, month=7, day=20
            ),  # Example date
            "Diwali": pd.Timestamp(year=date.year, month=11, day=4),  # Example date
            "Vesak": pd.Timestamp(year=date.year, month=5, day=26),  # Example date
            "Lunar New Year": pd.Timestamp(
                year=date.year, month=2, day=12
            ),  # Example date
            "Passover": pd.Timestamp(year=date.year, month=3, day=27),  # Example date
            "Rosh Hashanah": pd.Timestamp(
                year=date.year, month=9, day=6
            ),  # Example date
            "Yom Kippur": pd.Timestamp(year=date.year, month=9, day=15),  # Example date
            "Good Friday": pd.Timestamp(year=date.year, month=4, day=2),  # Example date
            "Easter": pd.Timestamp(year=date.year, month=4, day=4),  # Example date
        }

        school_holidays = {
            "Summer Vacation": (
                pd.Timestamp(year=date.year, month=6, day=25),
                pd.Timestamp(year=date.year, month=9, day=6),
            ),
            "Winter Recess": (
                pd.Timestamp(year=date.year, month=12, day=24),
                pd.Timestamp(year=date.year + 1, month=1, day=2),
            ),
            "Midwinter Recess": (
                pd.Timestamp(year=date.year, month=2, day=15),
                pd.Timestamp(year=date.year, month=2, day=19),
            ),
            "Spring Recess": (
                pd.Timestamp(year=date.year, month=4, day=1),
                pd.Timestamp(year=date.year, month=4, day=9),
            ),
            "Labor Day": (
                pd.Timestamp(
                    year=date.year,
                    month=9,
                    day=(
                        1
                        + (0 - pd.Timestamp(year=date.year, month=9, day=1).weekday())
                        % 7
                    ),
                ),
                0,
            ),
            "Rosh Hashanah": (
                pd.Timestamp(year=date.year, month=9, day=6),
                0,
            ),  # Example date
            "Yom Kippur": (
                pd.Timestamp(year=date.year, month=9, day=15),
                0,
            ),  # Example date
            "Columbus Day": (
                pd.Timestamp(
                    year=date.year,
                    month=10,
                    day=(
                        8
                        + (0 - pd.Timestamp(year=date.year, month=10, day=1).weekday())
                        % 7
                    ),
                ),
                0,
            ),
            "Election Day": (
                pd.Timestamp(
                    year=date.year,
                    month=11,
                    day=(
                        1
                        + (2 + pd.Timestamp(year=date.year, month=11, day=1).weekday())
                        % 7
                    )
                    + 1,
                ),
                0,
            ),
            "Veterans Day": (pd.Timestamp(year=date.year, month=11, day=11), 0),
            "Thanksgiving Break": (
                pd.Timestamp(
                    year=date.year,
                    month=11,
                    day=(
                        22
                        + (3 - pd.Timestamp(year=date.year, month=11, day=1).weekday())
                        % 7
                    ),
                ),
                pd.Timestamp(
                    year=date.year,
                    month=11,
                    day=(
                        23
                        + (3 - pd.Timestamp(year=date.year, month=11, day=1).weekday())
                        % 7
                    ),
                ),
            ),
            "Dr. Martin Luther King Jr. Day": (
                pd.Timestamp(
                    year=date.year,
                    month=1,
                    day=(
                        15
                        + (0 - pd.Timestamp(year=date.year, month=1, day=1).weekday())
                        % 7
                    ),
                ),
                0,
            ),
            "Lunar New Year": (
                pd.Timestamp(year=date.year, month=2, day=12),
                0,
            ),  # Example date
            "Memorial Day": (
                pd.Timestamp(
                    year=date.year,
                    month=5,
                    day=(
                        31
                        - (pd.Timestamp(year=date.year, month=5, day=31).weekday() - 0)
                    ),
                ),
                0,
            ),
        }

        categories = {
            "national_holiday": 0,
            "religious_holiday": 0,
            "special_day": 0,
            "school_holiday": 0,
        }

        if date in us_holidays:
            categories["national_holiday"] = 1
        if any(date == d for d in religious_holidays.values() if d is not None):
            categories["religious_holiday"] = 1
        if date in ny2_holidays and date not in ny_holidays:
            categories["special_day"] = 1
        if any(date == d for d in ny_holidays.values() if d is not None):
            categories["special_day"] = 1
        if any(
            start <= date <= end if isinstance(end, pd.Timestamp) else date == start
            for start, end in school_holidays.values()
        ):
            categories["school_holiday"] = 1

        return categories

    # Generate hourly datetime range from 2013 to 2024
    hourly_dates = pd.date_range(
        start="2013-01-01", end="2024-12-31 23:00:00", freq="h"
    )
    hourly_df = pd.DataFrame(hourly_dates, columns=["datetime"])
    hourly_df["federal_holiday"] = 0
    hourly_df["national_holiday"] = 0
    hourly_df["religious_holiday"] = 0
    hourly_df["special_day"] = 0
    hourly_df["school_holiday"] = 0

    print("Processing holidays")
    # Add federal holidays
    for rule in cal.rules:
        holidays_filter = (hourly_df["datetime"].dt.month == rule.month) & (
            hourly_df["datetime"].dt.day == rule.day
        )
        hourly_df.loc[holidays_filter, "federal_holiday"] = 1

    # Add special holidays
    for index, row in hourly_df.iterrows():
        categories = get_holiday_categories(row["datetime"])
        hourly_df.at[index, "national_holiday"] = categories["national_holiday"]
        hourly_df.at[index, "religious_holiday"] = categories["religious_holiday"]
        hourly_df.at[index, "special_day"] = categories["special_day"]
        hourly_df.at[index, "school_holiday"] = categories["school_holiday"]

    save_data(
        os.path.join(DATA_DIR, fmt, "augmented", f"holidays.{fmt}"),
        hourly_df,
        fmt,
    )
    print()
    end_time = time.time()
    processing_times["Holidays"] = end_time - start_time

    # Business data
    start_time = time.time()

    business_data = read_data(
        os.path.join(DATA_DIR, fmt, "raw", f"business_locations.{fmt}"),
        format=fmt,
        dtype={"Location": "object"},
        assume_missing=True,
    )
    # Convert License Creation Date and License Expiration Date to datetime
    business_data["License Creation Date"] = dd.to_datetime(
        business_data["License Creation Date"], errors="coerce"
    )
    business_data["License Expiration Date"] = dd.to_datetime(
        business_data["License Expiration Date"], errors="coerce"
    )

    # Filter for relevant columns
    business_data = business_data[
        [
            "License Expiration Date",
            "License Status",
            "License Creation Date",
            "Industry",
            "Location",
        ]
    ]

    # Create LATITUDE and LONGITUDE columns from tuple location
    business_data = business_data.dropna(subset="Location")
    business_data["Location"] = (
        business_data["Location"].str.replace("(", "").str.replace(")", "")
    )
    lat_lon = business_data["Location"].str.split(",", expand=True, n=2).compute()
    business_data["LATITUDE"] = lat_lon[0].astype(float)
    business_data["LONGITUDE"] = lat_lon[1].astype(float)
    business_data = business_data.loc[
        business_data["License Expiration Date"] > pd.Timestamp("2014-01-01")
    ]
    business_data["Borough"] = None

    print("Locating businesses")
    business_data = business_data.compute()
    for i, row in boroughs.iterrows():
        print(boroughs.at[i, "BoroName"])
        geom = boroughs.at[i, "geometry"]
        mask = geom.contains(
            geopandas.points_from_xy(business_data.LONGITUDE, business_data.LATITUDE)
        )
        business_data.loc[mask, "Borough"] = boroughs.at[i, "BoroName"]
    print()
    business_data = dd.from_pandas(business_data, npartitions=10)

    industry_hours = {
        "Laundries": (7, 21),
        "Sidewalk Cafe": (8, 22),
        "Secondhand Dealer - General": (10, 18),
        "Electronic & Appliance Service": (9, 19),
        "Employment Agency": (9, 17),
        "Home Improvement Contractor": (8, 18),
        "Tobacco Retail Dealer": (9, 21),
        "Electronic Cigarette Dealer": (10, 20),
        "Newsstand": (6, 21),
        "Garage": (0, 24),
        "Electronics Store": (10, 20),
        "Stoop Line Stand": (7, 20),
        "Tow Truck Company": (0, 24),
        "Secondhand Dealer - Auto": (9, 19),
        "Garage and Parking Lot": (0, 24),
        "Bingo Game Operator": (18, 23),
        "Pawnbroker": (9, 18),
        "Process Serving Agency": (9, 17),
        "Car Wash": (8, 20),
        "Dealer In Products": (10, 18),
        "Laundry": (7, 21),
        "Parking Lot": (0, 24),
        "Laundry Jobber": (7, 17),
        "Construction Labor Provider": (7, 17),
        "Pedicab Business": (10, 18),
        "Special Sale": (10, 18),
        "Third Party Food Delivery": (0, 24),
        "Debt Collection Agency": (9, 17),
        "Scrap Metal Processor": (8, 17),
        "Catering Establishment": (9, 21),
        "Tow Truck Exemption": (0, 24),
        "Ticket Seller Business": (10, 18),
        "Horse Drawn Cab Owner": (9, 17),
        "Auction House Premises": (10, 18),
        "Amusement Device Temporary": (10, 22),
        "Games of Chance": (10, 22),
        "Storage Warehouse": (8, 17),
        "Amusement Device Portable": (10, 22),
        "Cabaret": (19, 4),
        "Booting Company": (8, 20),
        "Commercial Lessor": (9, 17),
        "Gaming Cafe": (10, 22),
        "Amusement Arcade": (10, 22),
        "Amusement Device Permanent": (10, 22),
        "Pool or Billiard Room": (12, 2),
        "Scale Dealer Repairer": (9, 17),
        "Sightseeing Bus": (8, 20),
        "General Vendor Distributor": (10, 18),
        "Secondhand Dealer - Firearms": (10, 18),
    }

    industry_weekend_service = {
        "Laundries": True,
        "Sidewalk Cafe": True,
        "Secondhand Dealer - General": True,
        "Electronic & Appliance Service": True,
        "Employment Agency": False,
        "Home Improvement Contractor": True,
        "Tobacco Retail Dealer": True,
        "Electronic Cigarette Dealer": True,
        "Newsstand": True,
        "Garage": True,
        "Electronics Store": True,
        "Stoop Line Stand": True,
        "Tow Truck Company": True,
        "Secondhand Dealer - Auto": True,
        "Garage and Parking Lot": True,
        "Bingo Game Operator": True,
        "Pawnbroker": True,
        "Process Serving Agency": False,
        "Car Wash": True,
        "Dealer In Products": True,
        "Laundry": True,
        "Parking Lot": True,
        "Laundry Jobber": True,
        "Construction Labor Provider": False,
        "Pedicab Business": True,
        "Special Sale": True,
        "Third Party Food Delivery": True,
        "Debt Collection Agency": False,
        "Scrap Metal Processor": False,
        "Catering Establishment": True,
        "Tow Truck Exemption": True,
        "Ticket Seller Business": True,
        "Horse Drawn Cab Owner": False,
        "Auction House Premises": False,
        "Amusement Device Temporary": True,
        "Games of Chance": True,
        "Storage Warehouse": False,
        "Amusement Device Portable": True,
        "Cabaret": True,
        "Booting Company": True,
        "Commercial Lessor": False,
        "Gaming Cafe": True,
        "Amusement Arcade": True,
        "Amusement Device Permanent": True,
        "Pool or Billiard Room": True,
        "Scale Dealer Repairer": False,
        "Sightseeing Bus": True,
        "General Vendor Distributor": True,
        "Secondhand Dealer - Firearms": False,
    }

    def get_business_hourly_df(business_data, borough):
        # Generate hourly datetime range from 2013 to 2024
        hourly_dates = pd.date_range(
            start="2013-01-01", end="2024-12-31 23:00:00", freq="h"
        )
        hourly_df = pd.DataFrame(hourly_dates, columns=["datetime"])

        business_data = business_data[business_data["Borough"] == borough].compute()

        hourly_df["open_businesses"] = 0

        # for index, row in tqdm(hourly_df.iterrows(), total=hourly_df.shape[0]):
        #     businesses = business_data[(business_data['License Creation Date'] < row['datetime']) & (business_data['License Expiration Date'] > row['datetime'])]
        #     for industry, hours in industry_hours.items():
        #         if row.datetime.hour < hours[0] or row.datetime.hour >= hours[1]:
        #             continue
        #         else:
        #             open_businesses = len(businesses[businesses['Industry'] == industry])
        #             hourly_df.at[index, 'open_businesses'] += open_businesses

        pbar = tqdm(total=business_data.shape[0], desc="Processing businesses")
        for industry, hours in industry_hours.items():
            industry_filter = hourly_df["datetime"].dt.hour.isin(
                range(hours[0], hours[1] + 1)
            )
            if not industry_weekend_service[industry]:
                industry_filter = industry_filter & hourly_df[
                    "datetime"
                ].dt.weekday.isin(range(0, 5))
            businesses = business_data[business_data["Industry"] == industry]
            for index, row in businesses.iterrows():
                hourly_df.loc[
                    industry_filter
                    & (hourly_df["datetime"] >= row["License Creation Date"])
                    & (hourly_df["datetime"] <= row["License Expiration Date"]),
                    "open_businesses",
                ] += 1
                pbar.update(1)
        pbar.close()
        return hourly_df

    print("Processing businesses")
    for borough in boroughs["BoroName"]:
        if borough is None:
            continue
        print(f"{borough}")
        hourly_df = get_business_hourly_df(business_data, borough)
        save_data(
            os.path.join(
                DATA_DIR, fmt, "augmented", f"{borough}_business_openings.{fmt}"
            ),
            hourly_df,
            fmt,
        )

    print()
    end_time = time.time()
    processing_times["Businesses"] = end_time - start_time

    # Event Data
    start_time = time.time()

    event_data = read_data(
        os.path.join(DATA_DIR, fmt, "raw", f"events.{fmt}"),
        format=fmt,
        dtype={"Street Closure Type": "object"},
        assume_missing=True,
    )

    # Convert Start Date/Time and End Date/Time to datetime
    event_data["Start Date/Time"] = dd.to_datetime(
        event_data["Start Date/Time"], errors="coerce"
    )
    event_data["End Date/Time"] = dd.to_datetime(
        event_data["End Date/Time"], errors="coerce"
    )

    event_data = event_data[
        [
            "Start Date/Time",
            "End Date/Time",
            "Event Type",
            "Street Closure Type",
            "Event Borough",
        ]
    ]

    def get_events_hourly_df(event_data, borough):
        # Generate hourly datetime range from 2013 to 2024
        hourly_dates = pd.date_range(
            start="2013-01-01", end="2024-12-31 23:00:00", freq="h"
        )
        hourly_df = pd.DataFrame(hourly_dates, columns=["datetime"])

        event_data = event_data[event_data["Event Borough"] == borough].compute()

        hourly_df["num_events"] = 0
        hourly_df["num_constructions"] = 0
        hourly_df["num_street_closures"] = 0

        for index, row in tqdm(
            event_data.iterrows(),
            total=event_data.shape[0],
            desc=f"Events in {borough}",
        ):
            start = row["Start Date/Time"]
            end = row["End Date/Time"]
            mask = (hourly_df["datetime"] >= start) & (hourly_df["datetime"] <= end)
            hourly_df.loc[mask, "num_events"] += 1
            if row["Event Type"] == "Construction":
                hourly_df.loc[mask, "num_constructions"] += 1
            if row["Street Closure Type"] is not None:
                hourly_df.loc[mask, "num_street_closures"] += 1

        return hourly_df

    print("Processing events")
    for borough in event_data["Event Borough"].unique():
        print(f"{borough}")
        hourly_df = get_events_hourly_df(event_data, borough)
        save_data(
            os.path.join(DATA_DIR, fmt, "augmented", f"{borough}_events.{fmt}"),
            hourly_df,
            fmt,
        )
    print()
    end_time = time.time()
    processing_times["Events"] = end_time - start_time

    # Landmark Data
    start_time = time.time()

    landmark_data = read_data(
        os.path.join(DATA_DIR, fmt, "raw", f"landmarks.{fmt}"),
        format=fmt,
        dtype={"Street Closure Type": "object"},
        assume_missing=True,
    )

    landmark_data = landmark_data.dropna(subset=["Borough"])
    landmark_data = landmark_data.drop_duplicates(subset=["OBJECTID"])
    landmark_data = landmark_data[["Shape_Leng", "Shape_Area", "Borough"]]

    boroughs_map = {
        "SI": "Staten Island",
        "QN": "Queens",
        "MN": "Manhattan",
        "BK": "Brooklyn",
        "BX": "Bronx",
    }

    landmark_data.Borough = landmark_data.Borough.map(boroughs_map)

    print("Processing landmarks")
    for borough in landmark_data["Borough"].unique():
        print(f"{borough}")
        hourly_dates = pd.date_range(
            start="2013-01-01", end="2024-12-31 23:00:00", freq="h"
        )
        hourly_df = pd.DataFrame(hourly_dates, columns=["datetime"])

        landmarks = landmark_data[landmark_data["Borough"] == borough].compute()
        borough_area = boroughs[boroughs["BoroName"] == borough].Shape_Area.values[0]
        total_landmark_area = landmarks.Shape_Area.sum()
        hourly_df["landmark_density"] = total_landmark_area / borough_area
        hourly_df["total_landmark_length"] = landmarks.Shape_Leng.sum()
        hourly_df["total_landmarks"] = len(landmarks)
        save_data(
            os.path.join(DATA_DIR, fmt, "augmented", f"{borough}_landmarks.{fmt}"),
            hourly_df,
            fmt,
        )

    print()
    end_time = time.time()
    processing_times["Landmarks"] = end_time - start_time

    # Weather Data
    start_time = time.time()
    print("Processing weather data")
    for borough in boroughs["BoroName"]:
        print(f"{borough}")
        weather_data = read_data(
            os.path.join(DATA_DIR, fmt, "raw", f"{borough}_weather.{fmt}"),
            format=fmt,
            assume_missing=True,
        )

        save_data(
            os.path.join(DATA_DIR, fmt, "augmented", f"{borough}_weather.{fmt}"),
            weather_data.compute(),
            fmt,
        )
    end_time = time.time()
    processing_times["Weather"] = end_time - start_time
    print("Done!")

    times_log_path = os.path.join("logs", f"T2_{fmt}_times.txt")
    with open(times_log_path, "w") as f:
        times = []
        for key, value in processing_times.items():
            print(f"{key :<10}: {value} seconds")
            f.write(f"{key :<10}: {value} seconds\n")
            times.append(value)
        mean_time = sum(times) / len(times)
        std_time = (sum((x - mean_time) ** 2 for x in times) / len(times)) ** 0.5
        f.write(f"\nMean Time: {mean_time:.2f} seconds\n")
        f.write(f"Standard Deviation: {std_time:.2f} seconds\n")
        print()
        print(f"Mean Time: {mean_time:.2f} seconds")
        print(f"Standard Deviation: {std_time:.2f} seconds")


if __name__ == "__main__":
    run_with_memory_log(
        main,
        os.path.join("logs", f"T2_{fmt}_memory_log.txt"),
    )
