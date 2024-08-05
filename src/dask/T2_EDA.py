# %%
import os
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

DATA_PATH = '../../data/csv/2014.csv'

def read_csv_files(base_path="../../data/CSV_new", usecols=None, years=None,):
    """
    Reads CSV files into a single Dask DataFrame.
    
    Parameters:
    - base_path: str, the directory where the CSV files are located
    - years: list of int or None, optional, list of years to filter which CSV files to read
    - columns: list of str or None, optional, list of column names to read from the CSV files
    
    Returns:
    - Dask DataFrame containing the combined data from the CSV files
    """
    
    # List to store file paths
    file_paths = []

    # If no years are specified, read all CSV files
    if years is None:
        file_paths = [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith('.csv')]
    else:
        year_strs = [str(year) for year in years]
        all_files = [file for file in os.listdir(base_path) if file.endswith('.csv')]
        filtered_files = [file for file in all_files if any(year_str in file for year_str in year_strs)]
        
        if not filtered_files:
            raise ValueError("No files found for the specified years.")
        
        file_paths = [os.path.join(base_path, file) for file in filtered_files]

    # Read the CSV files into a Dask DataFrame with specified columns
    ddf = dd.read_csv(file_paths, usecols=usecols)
    
    return ddf


# %%
# Distribution of Registration States
df = read_csv_files(usecols=['Registration State'])
df = df.compute()
df_reg_state = df['Registration State'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Registration State', order=df_reg_state.index)
plt.title('Distribution of Registration States')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Distribution of 'Vehicle Body Type'
df = read_csv_files(usecols=['Vehicle Body Type']).compute()
df = df.reset_index(drop=True)
df_reg_state = df['Vehicle Body Type'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Vehicle Body Type', order=list(df_reg_state.index[:10]))
plt.title('Distribution of Registration States')
plt.xticks(rotation=90)
plt.show()
del df


# %%
# Distribution of Vehicle Year
df = read_csv_files(usecols=['Vehicle Year'], assume_missing=True)
df['Vehicle Year'] = dd.to_numeric(df['Vehicle Year'], errors='coerce')
df['Vehicle Year'] = df['Vehicle Year'].fillna(0).astype(int)
df = df[df['Vehicle Year'] != 0]
plt.figure(figsize=(12, 6))
sns.histplot(df['Vehicle Year'].compute(), bins=30)
plt.title('Distribution of Vehicle Year')
plt.show()
del df

# %%
# Correlation Matrix
df = read_csv_files(usecols=['Vehicle Year', 'Feet From Curb'], assume_missing=True)
df = df.dropna()
df_corr = df.corr().compute()
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
del df

# %%
# Top 10 Most Frequent Violations
df = read_csv_files(usecols=['Violation Code'], assume_missing=True).compute()
df_violation_code = df['Violation Code'].value_counts().head(10)
df = df.reset_index(drop=True)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Violation Code', order=df_violation_code.index)
plt.title('Top 10 Most Frequent Violations')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Violations Over Time
df = read_csv_files(usecols=['Issue Date'], assume_missing=True)
df['Issue Date'] = dd.to_datetime(df['Issue Date'])
violations_over_time = df['Issue Date'].value_counts().compute().sort_index()

plt.figure(figsize=(12, 6))
violations_over_time.plot()
plt.title('Violations Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Violations')
plt.show()
del df

# %%
# Violations by Precinct
df = read_csv_files(usecols=['Violation Precinct']).compute()
df = df.reset_index(drop=True)
df_precinct = df['Violation Precinct'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Violation Precinct', order=df_precinct.index)
plt.title('Violations by Precinct')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Top 10 Vehicle Makes with Most Violations
df = read_csv_files(usecols=['Vehicle Make']).compute()
df = df.reset_index(drop=True)
df_make = df['Vehicle Make'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Vehicle Make', order=df_make.index)
plt.title('Top 10 Vehicle Makes with Most Violations')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Violations by Issuing Agency
df = read_csv_files(usecols=['Issuing Agency']).compute()
df = df.reset_index(drop=True)
df_agency = df['Issuing Agency'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Issuing Agency', order=df_agency.index)
plt.title('Violations by Issuing Agency')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Distribution of Violation Times
df = read_csv_files(usecols=['Violation Time']).compute()
df = df.reset_index(drop=True)
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Violation Time', bins=24, kde=True)
plt.title('Distribution of Violation Times')
plt.show()
del df

# %%
# Heatmap of Violation Counts by Location and Precinct
df = read_csv_files(usecols=['Violation Precinct', 'Violation Location'])
violation_location_precinct = df.groupby(['Violation Precinct', 'Violation Location']).size().compute().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
sns.heatmap(violation_location_precinct, cmap='coolwarm')
plt.title('Heatmap of Violation Counts by Location and Precinct')
plt.show()
del df

# %%
# Box Plot of Feet From Curb
df = read_csv_files(usecols=['Feet From Curb'])
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.compute(), x='Feet From Curb')
plt.title('Box Plot of Feet From Curb')
plt.show()
del df

# %%
# Box Plot of Vehicle Year
df = read_csv_files(usecols=['Vehicle Year'])
df['Vehicle Year'] = dd.to_numeric(df['Vehicle Year'], errors='coerce')
df['Vehicle Year'] = df['Vehicle Year'].fillna(0).astype(int)
df = df[df['Vehicle Year'] != 0]
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Vehicle Year'].compute())
plt.title('Box Plot of Vehicle Year')
plt.show()
del df
