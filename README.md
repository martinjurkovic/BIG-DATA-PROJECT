# BIG-DATA-PROJECT
**Authors:**
- Martin Jurkovič  
Email: mj5835@student.uni-lj.si  
Faculty of Computer and Information Science  
Student ID: 63180015  

- Valter Hudovernik  
Email: vh0153@student.uni-lj.si  
Faculty of Computer and Information Science  
Student ID: 63160134  

Project Repository:
[https://github.com/martinjurkovic/BIG-DATA-PROJECT](https://github.com/martinjurkovic/BIG-DATA-PROJECT)


# Project Overview

The Big Data Project Report 2024 is an extensive exploration into the performance and suitability of various data formats and distributed computing frameworks for handling large datasets. The study specifically focuses on three file formats—Parquet, DuckDB, and HDF5—in combination with the distributed computing framework Dask. The project also includes a streaming analysis using the Kafka framework, comparing it with traditional distributed computing approaches and ML tasks on the NYC parking violations dataset.

## Repository Structure

├ Instructions.pdf  
├ README.md  
├ T3_figures  
├ T4_dashboard_gifs  
├ data  
   ├ CSV  
   ├ HDF5  
   ├ duckdb  
   └ parquet  
├ logs  
├ pyproject.toml  
└ src  
   ├ T1_data_format_conversion  
   ├ T2_data_augmentation  
   ├ T3_exploratory_data_analysis  
   ├ T4_kafka  
   ├ T5_machine_learning  
   └ bigdata  

### T3_figures

This directory contains various figures generated during exploratory data analysis. These figures include visualizations such as correlation matrices, distributions of registration states, and heatmaps of violation counts.

### T4_dashboard_gifs

## Statistics Dashboard
Below you can see the statistics dashboard for the parking violations dataset. The dashboard is created using the `dash` library in Python. The dashboard consists of 3 tabs, Overall, boroughs and top 10 streets. All of the statistics are updated live when data is streamed into kafka in the webapp.
![Dashboard Gif](https://github.com/martinjurkovic/BIG-DATA-PROJECT/blob/main/T4_dashboard_gifs/statistics.gif)

## Streaming Clustering Dashboard
Similarly, we also built a dashboard for accuracy of the clustering algorithm.

![Clustering gif](https://github.com/martinjurkovic/BIG-DATA-PROJECT/blob/main/T4_dashboard_gifs/accuracy.gif)


### data

The `data` directory contains the datasets used in the project in various formats:

- **CSV:** Contains CSV files from 2014 to 2024 and associated raw datasets such as weather and business locations.
- **HDF5:** Stores the data in HDF5 format, with files organized by year and type.
- **duckdb:** DuckDB databases with raw, augmented, and standard data.
- **parquet:** Parquet files organized by year and type, including raw and augmented data.

### logs

The `logs` directory contains log files capturing the memory usage and execution times of various tasks. This includes logs from data conversions, exploratory data analysis, and machine learning tasks.

### src

The `src` directory is the main source directory for the project, containing the following subdirectories:

- **T1_data_format_conversion:** Contains Python scripts for converting data between different formats (CSV to DuckDB, HDF5, and Parquet).
  - `csv_to_duckdb.py`
  - `csv_to_hdf5.py`
  - `csv_to_parquet.py`
  
- **T2_data_augmentation:** Contains scripts for augmenting raw data and converting it into formats suitable for analysis.
  - `augment_data.py`
  - `convert_raw_files.py`

- **T3_exploratory_data_analysis:** Contains scripts and Jupyter notebooks for conducting exploratory data analysis.
  - `exploratory_data_analysis.py`
  - `interesting_streets.ipynb`
  - `visualize_schools.ipynb`

- **T4_kafka:** Contains Python scripts and configuration files for Kafka streaming, including producers and consumers for data streams.
  - `README.md`
  - `clustering_consumer.py`
  - `dash_consumer.py`
  - `docker-compose.yml`
  - `producer.py`
  
- **T5_machine_learning:** Contains machine learning scripts for predictive modeling on high ticket days and out-of-state vehicles.
  - `high_ticket_days.py`
  - `out_of_state.py`

- **bigdata:** Contains the utility python package for the project.
  - `augmentation_utils.py`
  - `hdf5.py`
  - `ml_utils.py`
  - `utils.py`

## Setup Instructions

1. **Install the bigdata python package and depencencies:**
   ```bash
   pip install -e .
   ```


