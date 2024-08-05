# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

DATA_PATH = '../../data/csv/2014.csv'

# %%
# Distribution of Registration States
df = pd.read_csv(DATA_PATH, usecols=['Registration State'])
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Registration State', order=df['Registration State'].value_counts().index)
plt.title('Distribution of Registration States')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Distribution of Vehicle Body Types
df = pd.read_csv(DATA_PATH, usecols=['Vehicle Body Type'])
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Vehicle Body Type', order=df['Vehicle Body Type'].value_counts().index[:10])
plt.title('Distribution of Vehicle Body Types')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Distribution of Vehicle Year
df = pd.read_csv(DATA_PATH, usecols=['Vehicle Year'])
df['Vehicle Year'] = pd.to_numeric(df['Vehicle Year'], errors='coerce')
df['Vehicle Year'] = df['Vehicle Year'].fillna(0).astype(int)
plt.figure(figsize=(12, 6))
sns.histplot(x=df['Vehicle Year'][df['Vehicle Year'] != 0], bins=30)
plt.title('Distribution of Vehicle Year')
plt.show()
del df

# %%
# Correlation Matrix
df = pd.read_csv(DATA_PATH, usecols=['Vehicle Year', 'Feet From Curb'])
df.dropna(inplace=True)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
del df

# %%
# Top 10 Most Frequent Violations
df = pd.read_csv(DATA_PATH, usecols=['Violation Code'])
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Violation Code', order=df['Violation Code'].value_counts().index[:10])
plt.title('Top 10 Most Frequent Violations')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Violations Over Time
df = pd.read_csv(DATA_PATH, usecols=['Issue Date'])
df['Issue Date'] = pd.to_datetime(df['Issue Date'])
violations_over_time = df['Issue Date'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
violations_over_time.plot()
plt.title('Violations Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Violations')
plt.show()
del df

# %%
# Violations by Precinct
df = pd.read_csv(DATA_PATH, usecols=['Violation Precinct'])
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Violation Precinct', order=df['Violation Precinct'].value_counts().index[:10])
plt.title('Violations by Precinct')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Top 10 Vehicle Makes with Most Violations
df = pd.read_csv(DATA_PATH, usecols=['Vehicle Make'])
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Vehicle Make', order=df['Vehicle Make'].value_counts().index[:10])
plt.title('Top 10 Vehicle Makes with Most Violations')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Violations by Issuing Agency
df = pd.read_csv(DATA_PATH, usecols=['Issuing Agency'])
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Issuing Agency', order=df['Issuing Agency'].value_counts().index)
plt.title('Violations by Issuing Agency')
plt.xticks(rotation=90)
plt.show()
del df

# %%
# Distribution of Violation Times
df = pd.read_csv(DATA_PATH, usecols=['Violation Time'])
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Violation Time', bins=24, kde=True)
plt.title('Distribution of Violation Times')
plt.show()
del df

# %%
# Heatmap of Violation Counts by Location and Precinct
df = pd.read_csv(DATA_PATH, usecols=['Violation Precinct', 'Violation Location'])
violation_location_precinct = df.groupby(['Violation Precinct', 'Violation Location']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
sns.heatmap(violation_location_precinct, cmap='coolwarm')
plt.title('Heatmap of Violation Counts by Location and Precinct')
plt.show()
del df

# %%
# Box Plot of Feet From Curb
df = pd.read_csv(DATA_PATH, usecols=['Feet From Curb'])
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Feet From Curb')
plt.title('Box Plot of Feet From Curb')
plt.show()
del df

# %%
# Box Plot of Vehicle Year
df = pd.read_csv(DATA_PATH, usecols=['Vehicle Year'])
df['Vehicle Year'] = pd.to_numeric(df['Vehicle Year'], errors='coerce')
df['Vehicle Year'] = df['Vehicle Year'].fillna(0).astype(int)
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Vehicle Year'][df['Vehicle Year'] != 0])
plt.title('Box Plot of Vehicle Year')
plt.show()
del df
