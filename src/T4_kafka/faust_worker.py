import time
from datetime import datetime, timedelta

import faust
import numpy as np

class WeatherData(faust.Record):
    WBANNO: str
    UTC_DATE: str
    UTC_TIME: str
    LST_DATE: str
    LST_TIME: str
    CRX_VN: str
    LONGITUDE: float
    LATITUDE: float
    AIR_TEMPERATURE: float
    PRECIPITATION: float
    SOLAR_RADIATION: float
    SR_FLAG: str
    SURFACE_TEMPERATURE: float
    ST_TYPE: str
    ST_FLAG: str
    RELATIVE_HUMIDITY: float
    RH_FLAG: str
    SOIL_MOISTURE_5: float
    SOIL_TEMPERATURE_5: float
    WETNESS: float
    WET_FLAG: str
    WIND_1_5: float
    WIND_FLAG: str


class OutlierData(faust.Record):
    temp: float
    mean: float
    std: float


class HourlyData(faust.Record):
    hour: int
    temperature: float


class HighestTemperature(faust.Record):
    station: str
    temperature: float

app = faust.App('weather_app', broker='kafka://localhost:29092')

weather_data_topic = app.topic('weather_data', key_type=str, value_type=WeatherData)
outliers_topic = app.topic('outliers', key_type=str, value_type=OutlierData)
hourly_data_topic = app.topic('hourly_temperature', key_type=str, value_type=HourlyData)
highest_temperature_topic = app.topic('highest_temperature', key_type=str, value_type=HighestTemperature)

hourly_temperature_table = app.Table('hourly_temperature', key_type=str, value_type=float, partitions=1, default=float)
hourly_count_table       = app.Table('hourly_count', key_type=str, value_type=int, partitions=1, default=int)


N = 36 # 3 hours
MAX_TEMP = 50
reference_values = {
    'Aleknagik_1_NNE': [],
    'Bethel_87_WNW': [],
    'Cordova_14_ESE': [],
}
SMOOTHING_FACTOR = 0.1
SCALE_STD = 1.25
momentum = 0

@app.agent(weather_data_topic)
async def weather_data_stream(data):
    global reference_values, N, MAX_TEMP, momentum, SCALE_STD
    async for key, record in data.items():
        # Extract station name from the key
        air_temp = float(record.AIR_TEMPERATURE)

        if len(reference_values[key]) < N:
            reference_values[key].append(air_temp)
        else:
            mean = np.mean(reference_values[key])
            std = np.std(reference_values[key], ddof=1)
            # first check feasibility
            if np.abs(air_temp) > MAX_TEMP:
                # print(f"Anomaly detected: {air_temp}")
                # sample a point from the normal distribution with mean and std
                sample = np.random.normal(mean, std)
                reference_values[key].append(sample)
                # dampen the momentum
                momentum = SMOOTHING_FACTOR * momentum
                # send the record to the outliers topic
                await outliers_topic.send(key=key, value={'temp': np.round(air_temp, 2), 'mean': np.round(mean, 2), 'std': np.round(std, 2)})
                continue
            # check if the record is an outlier
            # project the record back based on the momentum
            projected = air_temp - momentum
            adj_mean = np.mean(reference_values[key] + [projected])
            adj_std = np.std(reference_values[key] + [projected], ddof=1)
            z_score_proj = (projected - adj_mean) / (adj_std * SCALE_STD)
            z_score = (air_temp - adj_mean) / (adj_std * SCALE_STD)
            if np.abs(z_score) > 3 and np.abs(z_score_proj) > 3:
                # print(f"Anomaly detected: {air_temp}, Z-Score: {z_score}, {z_score_proj}*")
                # print(f"Temp: {air_temp} -> {projected}, Mean: {mean}, {adj_mean}*, Std: {std}, {adj_std}*")
                # time.sleep(2)
                
                # some sudden shifts are possible
                # if the z-score is less than 4.5, use the new record in mean calculation
                if z_score < 4.5:
                    sample = air_temp
                elif z_score_proj < 4.5:
                    sample = projected
                else:
                    sample = np.random.normal(mean, std) + momentum
                reference_values[key].pop(0)
                reference_values[key].append(sample)

                # dampen the momentum
                momentum = SMOOTHING_FACTOR * momentum
                # send the record to the outliers topic
                await outliers_topic.send(key=key, value={'temp': np.round(air_temp, 2), 'mean': np.round(mean, 2), 'std': np.round(std, 2)})
                continue
            # update the momentum
            momentum = SMOOTHING_FACTOR * momentum + (1-SMOOTHING_FACTOR) * (air_temp - reference_values[key][-1])
            # print(f"{reference_values[key][-1]} -> {air_temp}, Momentum: {momentum}")
            # time.sleep(0.5)
            
            # remove the oldest value
            reference_values[key].pop(0)
            reference_values[key].append(air_temp)

        hour = int(record.UTC_TIME[:2])
        date = f"{record.UTC_DATE}"
        hourly_temperature_table[f"{key}", f"{hour}", date] += air_temp
        hourly_count_table[f"{key}", f"{hour}", date] += 1

CURRENT_HOUR = 0
CURRENT_DATE = datetime.strptime('20210101', '%Y%m%d')
@app.timer(interval=10)  # Timer to trigger hourly computation
async def compute_hourly_temperature():
    global CURRENT_HOUR, CURRENT_DATE

    hourly_data = dict()
    hour = f"{CURRENT_HOUR}"
    date = datetime.strftime(CURRENT_DATE, '%Y%m%d')

    for station_name in ['Aleknagik_1_NNE', 'Bethel_87_WNW', 'Cordova_14_ESE']:
        if (station_name, hour, date) in hourly_temperature_table:
            temperature_sum = hourly_temperature_table[station_name, hour, date]
            temperature_count = hourly_count_table[station_name, hour, date]
            # print(f"{station_name} {hour} {temperature_sum} {temperature_count}")
            if temperature_count > 0:
                hourly_mean_temperature = temperature_sum / temperature_count
                print(f"Hourly mean temperature for station {station_name} @ {date}-{hour}: {hourly_mean_temperature}")
                await hourly_data_topic.send(key=station_name, value={'hour': hour, 'temperature': np.round(hourly_mean_temperature, 3)})
                hourly_data[station_name] = hourly_mean_temperature
    
    if len(hourly_data):
        # get hottest station
        hottest_station = max(hourly_data, key=hourly_data.get)
        print(f"Hottest station at hour {CURRENT_HOUR} is {hottest_station} with temperature {hourly_data[hottest_station]}")
        await highest_temperature_topic.send(key=hour, value={'station': hottest_station, 'temperature': np.round(hourly_data[hottest_station], 4)})

        CURRENT_HOUR += 1
        if CURRENT_HOUR == 24:
            CURRENT_HOUR = 0
            CURRENT_DATE += timedelta(days=1)
