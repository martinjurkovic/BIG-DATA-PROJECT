# %%
from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set_theme(style="whitegrid")

# Define the directory for saving figures

fig_dir = Path.cwd() / 'figs'
fig_dir.mkdir(exist_ok=True)

# Define paths
cwd = Path.cwd()
path = str(cwd).split("BIG-DATA-PROJECT")
PARQUET_PATH = Path(path[0]) / "BIG-DATA-PROJECT/data/parquet/"
DUCKDB_PATH = Path(path[0]) / "BIG-DATA-PROJECT/data/duckdb/"

# Create a DuckDB connection
conn = duckdb.connect(str(DUCKDB_PATH / "nyc_database.db"))

# Load the parquet files into DuckDB
# conn.execute(f"CREATE TABLE nyc_data AS SELECT * FROM parquet_scan('{PARQUET_PATH / '*.parquet'}')")

def read_data_from_duckdb(columns=None, years=None):
    """
    Reads data from DuckDB and filters based on specified columns and years using only SQL.
    
    Parameters:
    - columns: list of str or None, optional, list of column names to read from the dataset
    - years: list of int or None, optional, list of years to filter the data
    
    Returns:
    - Pandas DataFrame containing the queried data
    """
    # Handle columns with spaces
    if columns:
        columns = [f'"{col}"' for col in columns]
        columns_str = ", ".join(columns)
    else:
        columns_str = "*"
    
    # Construct the base SQL query
    query = f"SELECT {columns_str} FROM nyc_data"
    
    if years:
        # Create a string of years for SQL IN clause
        years_str = ', '.join(map(str, years))
        
        # Split 'Issue Date' on '/' and extract the year
        query += f"""
            WHERE SPLIT_PART("Issue Date", '/', 3)::INTEGER IN ({years_str})
        """
    
    # Execute the query and return the result as a DataFrame
    df = conn.execute(query).df()
    return df


# Initialize list to store times for each plot
plot_times = []

# Function to save times to a .txt file
def save_times_to_file(times, filename='plot_times.txt'):
    with open(filename, 'w') as f:
        for i, t in enumerate(times):
            f.write(f'Plot {i+1}: {t:.2f} seconds\n')
        mean_time = sum(times) / len(times)
        std_time = (sum((x - mean_time) ** 2 for x in times) / len(times)) ** 0.5
        f.write(f'\nMean Time: {mean_time:.2f} seconds\n')
        f.write(f'Standard Deviation: {std_time:.2f} seconds\n')


# %%
# Distribution of Registration States
start_time = time.time()
df = read_data_from_duckdb(columns=['Registration State'])
df_reg_state = df['Registration State'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Registration State', order=df_reg_state.index)
plt.title('Distribution of Registration States')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'distribution_registration_states.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Distribution of 'Vehicle Body Type'
start_time = time.time()
df = read_data_from_duckdb(columns=['Vehicle Body Type'])
df_reg_state = df['Vehicle Body Type'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Vehicle Body Type', order=list(df_reg_state.index[:10]))
plt.title('Distribution of Vehicle Body Type')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'distribution_vehicle_body_type.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Distribution of Vehicle Year
start_time = time.time()
df = read_data_from_duckdb(columns=['Vehicle Year'])
df['Vehicle Year'] = pd.to_numeric(df['Vehicle Year'], errors='coerce')
df['Vehicle Year'] = df['Vehicle Year'].fillna(0).astype(int)
df = df[df['Vehicle Year'] != 0]
plt.figure(figsize=(12, 6))
sns.histplot(df['Vehicle Year'], bins=30)
plt.title('Distribution of Vehicle Year')
plt.savefig(fig_dir / 'distribution_vehicle_year.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Correlation Matrix
start_time = time.time()
df = read_data_from_duckdb(columns=['Vehicle Year', 'Feet From Curb'])
df = df.dropna()
df_corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig(fig_dir / 'correlation_matrix.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Top 10 Most Frequent Violations
start_time = time.time()
df = read_data_from_duckdb(columns=['Violation Code'])
df_violation_code = df['Violation Code'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Violation Code', order=df_violation_code.index)
plt.title('Top 10 Most Frequent Violations')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'top_10_most_frequent_violations.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Violations Over Time
start_time = time.time()
df = read_data_from_duckdb(columns=['Issue Date'])
df['Issue Date'] = pd.to_datetime(df['Issue Date'])
violations_over_time = df['Issue Date'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
violations_over_time.plot()
plt.title('Violations Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Violations')
plt.savefig(fig_dir / 'violations_over_time.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Violations by Precinct
start_time = time.time()
df = read_data_from_duckdb(columns=['Violation Precinct'])
df_precinct = df['Violation Precinct'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Violation Precinct', order=df_precinct.index)
plt.title('Violations by Precinct')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'violations_by_precinct.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Top 10 Vehicle Makes with Most Violations
start_time = time.time()
df = read_data_from_duckdb(columns=['Vehicle Make'])
df_make = df['Vehicle Make'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Vehicle Make', order=df_make.index)
plt.title('Top 10 Vehicle Makes with Most Violations')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'top_10_vehicle_makes.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Violations by Issuing Agency
start_time = time.time()
df = read_data_from_duckdb(columns=['Issuing Agency'])
df_agency = df['Issuing Agency'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Issuing Agency', order=df_agency.index)
plt.title('Violations by Issuing Agency')
plt.xticks(rotation=90)
plt.savefig(fig_dir / 'violations_by_issuing_agency.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Distribution of Violation Times
start_time = time.time()
df = read_data_from_duckdb(columns=['Violation Time'])
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Violation Time', bins=24, kde=True)
plt.title('Distribution of Violation Times')
plt.savefig(fig_dir / 'distribution_violation_times.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Heatmap of Violation Counts by Location and Precinct
start_time = time.time()
df = read_data_from_duckdb(columns=['Violation Precinct', 'Violation Location'])
violation_location_precinct = df.groupby(['Violation Precinct', 'Violation Location']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
sns.heatmap(violation_location_precinct, cmap='coolwarm')
plt.title('Heatmap of Violation Counts by Location and Precinct')
plt.savefig(fig_dir / 'heatmap_violation_counts.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Box Plot of Feet From Curb
start_time = time.time()
df = read_data_from_duckdb(columns=['Feet From Curb'])
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Feet From Curb')
plt.title('Box Plot of Feet From Curb')
plt.savefig(fig_dir / 'box_plot_feet_from_curb.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# %%
# Box Plot of Vehicle Year
start_time = time.time()
df = read_data_from_duckdb(columns=['Vehicle Year'])
df['Vehicle Year'] = pd.to_numeric(df['Vehicle Year'], errors='coerce')
df['Vehicle Year'] = df['Vehicle Year'].fillna(0).astype(int)
df = df[df['Vehicle Year'] != 0]
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Vehicle Year'])
plt.title('Box Plot of Vehicle Year')
plt.savefig(fig_dir / 'box_plot_vehicle_year.png')
plt.close()
end_time = time.time()
plot_times.append(end_time - start_time)

# Save times to file
save_times_to_file(plot_times)