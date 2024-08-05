# %%
import os
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path

# Set up the directory for saving figures
fig_dir = Path.cwd() / 'figs'
fig_dir.mkdir(exist_ok=True)

# Initialize list to store times for each plot
plot_times = []

# Function to save times to a .txt file
def save_times_to_file(times, filename='dask_plot_times.txt'):
    with open(filename, 'w') as f:
        for i, t in enumerate(times):
            f.write(f'Plot {i+1}: {t:.2f} seconds\n')
        mean_time = sum(times) / len(times)
        std_time = (sum((x - mean_time) ** 2 for x in times) / len(times)) ** 0.5
        f.write(f'\nMean Time: {mean_time:.2f} seconds\n')
        f.write(f'Standard Deviation: {std_time:.2f} seconds\n')

# Define function to read CSV files into a Dask DataFrame
def read_csv_files(base_path="../../data/csv", usecols=None, years=None):
    file_paths = []
    if years is None:
        file_paths = [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith('.csv')]
    else:
        year_strs = [str(year) for year in years]
        all_files = [file for file in os.listdir(base_path) if file.endswith('.csv')]
        filtered_files = [file for file in all_files if any(year_str in file for year_str in year_strs)]
        if not filtered_files:
            raise ValueError("No files found for the specified years.")
        file_paths = [os.path.join(base_path, file) for file in filtered_files]
    ddf = dd.read_csv(file_paths, usecols=usecols, assume_missing=True)
    return ddf

# %%
# Distribution of Registration States
start_time = time.time()
df = read_csv_files(usecols=['Registration State'], years=["2023"]).compute()
df_reg_state = df['Registration State'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Registration State', order=df_reg_state.index)
plt.title('Distribution of Registration States')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'distribution_registration_states_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Distribution of 'Vehicle Body Type'
start_time = time.time()
df = read_csv_files(usecols=['Vehicle Body Type'], years=["2023"]).compute()
df = df.reset_index(drop=True)
df_reg_state = df['Vehicle Body Type'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Vehicle Body Type', order=list(df_reg_state.index[:10]))
plt.title('Distribution of Vehicle Body Type')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'distribution_vehicle_body_type_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Distribution of Vehicle Year
start_time = time.time()
df = read_csv_files(usecols=['Vehicle Year'], years=["2023"])
df['Vehicle Year'] = dd.to_numeric(df['Vehicle Year'], errors='coerce')
df['Vehicle Year'] = df['Vehicle Year'].fillna(0).astype(int)
df = df[df['Vehicle Year'] != 0]
plt.figure(figsize=(12, 6))
sns.histplot(df['Vehicle Year'].compute(), bins=30)
plt.title('Distribution of Vehicle Year')
plt.savefig(fig_dir / 'distribution_vehicle_year_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Correlation Matrix
start_time = time.time()
df = read_csv_files(usecols=['Vehicle Year', 'Feet From Curb'], years=["2023"])
df = df.dropna()
df_corr = df.corr().compute()
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig(fig_dir / 'correlation_matrix_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Top 10 Most Frequent Violations
start_time = time.time()
df = read_csv_files(usecols=['Violation Code'], years=["2023"]).compute()
df = df.reset_index(drop=True)
df_violation_code = df['Violation Code'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Violation Code', order=df_violation_code.index)
plt.title('Top 10 Most Frequent Violations')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'top_10_most_frequent_violations_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Violations Over Time
start_time = time.time()
df = read_csv_files(usecols=['Issue Date'], years=["2023"])
df['Issue Date'] = dd.to_datetime(df['Issue Date'])
violations_over_time = df['Issue Date'].value_counts().compute().sort_index()
plt.figure(figsize=(12, 6))
violations_over_time.plot()
plt.title('Violations Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Violations')
plt.savefig(fig_dir / 'violations_over_time_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Violations by Precinct
start_time = time.time()
df = read_csv_files(usecols=['Violation Precinct'], years=["2023"]).compute()
df = df.reset_index(drop=True)
df_precinct = df['Violation Precinct'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Violation Precinct', order=df_precinct.index)
plt.title('Violations by Precinct')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'violations_by_precinct_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Top 10 Vehicle Makes with Most Violations
start_time = time.time()
df = read_csv_files(usecols=['Vehicle Make'], years=["2023"]).compute()
df = df.reset_index(drop=True)
df_make = df['Vehicle Make'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Vehicle Make', order=df_make.index)
plt.title('Top 10 Vehicle Makes with Most Violations')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'top_10_vehicle_makes_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Violations by Issuing Agency
start_time = time.time()
df = read_csv_files(usecols=['Issuing Agency'], years=["2023"]).compute()
df = df.reset_index(drop=True)
df_agency = df['Issuing Agency'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Issuing Agency', order=df_agency.index)
plt.title('Violations by Issuing Agency')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'violations_by_issuing_agency_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Distribution of Violation Times
start_time = time.time()
df = read_csv_files(usecols=['Violation Time'], years=["2023"]).compute()
df = df.reset_index(drop=True)
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Violation Time', bins=24, kde=True)
plt.title('Distribution of Violation Times')
plt.savefig(fig_dir / 'distribution_violation_times_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Heatmap of Violation Counts by Location and Precinct
start_time = time.time()
df = read_csv_files(usecols=['Violation Precinct', 'Violation Location'], years=["2023"])
violation_location_precinct = df.groupby(['Violation Precinct', 'Violation Location']).size().compute().unstack(fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(violation_location_precinct, cmap='coolwarm')
plt.title('Heatmap of Violation Counts by Location and Precinct')
plt.savefig(fig_dir / 'heatmap_violation_counts_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Box Plot of Feet From Curb
start_time = time.time()
df = read_csv_files(usecols=['Feet From Curb'], years=["2023"]).compute()
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Feet From Curb')
plt.title('Box Plot of Feet From Curb')
plt.savefig(fig_dir / 'box_plot_feet_from_curb_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Box Plot of Vehicle Year
start_time = time.time()
df = read_csv_files(usecols=['Vehicle Year'], years=["2023"])
df['Vehicle Year'] = dd.to_numeric(df['Vehicle Year'], errors='coerce')
df['Vehicle Year'] = df['Vehicle Year'].fillna(0).astype(int)
df = df[df['Vehicle Year'] != 0]
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Vehicle Year'].compute())
plt.title('Box Plot of Vehicle Year')
plt.savefig(fig_dir / 'box_plot_vehicle_year_dask.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# Save times to file
save_times_to_file(plot_times)

# %%
