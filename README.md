# BIG-DATA-PROJECT

# REPORT STRUCTURE (CRISP-DM)
[CRISP-DM INSTRUCTIONS](https://s2.smu.edu/~mhd/8331f03/crisp.pdf)

# NYC Open Data

New York City open data - Parking Violations Issued: https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2022/pvqr-7yc4; for previous years see https://catalog.data.gov/dataset/parking-violations-issued-fiscal-year-2023 (you can change the year from 2014 to 2023)

HPC: `/d/hpc/projects/FRI/bigdata/data/NYTickets`


# T4 DASHBOARD

## Statistics Dashboard
Below you can see the statistics dashboard for the parking violations dataset. The dashboard is created using the `dash` library in Python. The dashboard consists of 3 tabs, Overall, boroughs and top 10 streets. All of the statistics are updated live when data is streamed into kafka in the webapp.
![Dashboard Gif](https://github.com/martinjurkovic/BIG-DATA-PROJECT/blob/main/T4_dashboard_gifs/statistics.gif)

## Streaming Clustering Dashboard
Similarly, we also built a dashboard for accuracy of the clustering algorithm.

![Clustering gif](https://github.com/martinjurkovic/BIG-DATA-PROJECT/blob/main/T4_dashboard_gifs/accuracy.gif)

