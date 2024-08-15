import os
import pandas as pd

# Define the paths to the directories
csv_dir = './CSV'
parquet_dir = './parquet'
hdf5_dir = './HDF5'

# Function to get file sizes from a directory and convert to GB
def get_file_sizes(directory, extension):
    files = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(extension):
            file_path = os.path.join(directory, file_name)
            files[file_name] = os.path.getsize(file_path) / 1e9  # Convert to GB
    return files

# Get the file sizes
csv_sizes = get_file_sizes(csv_dir, '.csv')
parquet_sizes = get_file_sizes(parquet_dir, '.parquet')
hdf5_sizes = get_file_sizes(hdf5_dir, '.h5')

file_names = list(csv_sizes.keys())
# Strip .csv extension
file_names = [f.replace('.csv', '') for f in file_names]

# Create a DataFrame
df = pd.DataFrame({
    'File Name': file_names,
    'CSV Size (GB)': list(csv_sizes.values()),
    'Parquet Size (GB)': [parquet_sizes.get(f.replace('.csv', '.parquet'), None) for f in csv_sizes.keys()],
    'HDF5 Size (GB)': [hdf5_sizes.get(f.replace('.csv', '.h5'), None) for f in csv_sizes.keys()]
})

# Sort by file name
df = df.sort_values(by='File Name')

# Add a "TOTAL" row at the bottom
total_row = pd.DataFrame({
    'File Name': ['TOTAL'],
    'CSV Size (GB)': [df['CSV Size (GB)'].sum()],
    'Parquet Size (GB)': [df['Parquet Size (GB)'].sum()],
    'HDF5 Size (GB)': [df['HDF5 Size (GB)'].sum()]
})

df = pd.concat([df, total_row], ignore_index=True)

# Generate LaTeX table
latex_table = df.to_latex(index=False, float_format="%.2f")

# Print the LaTeX table
print(latex_table)
