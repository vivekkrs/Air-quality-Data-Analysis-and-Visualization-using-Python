# AirQuality-DataAnalysis-Visualization
# Exploratory Data Analysis (EDA) and visualization of air pollution data across major Indian cities using Python.
# Includes data cleaning, outlier detection, time-series analysis, hypothesis testing, and visual insights.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel

# Objective 1: Clean and Manipulate Data using Pandas and NumPy

df = pd.read_csv('AirQuality Python Dataset.csv')
print("Original Data:")
print(df.head())

date_range = pd.date_range(start='2022-01-01', end='2025-12-31', periods=len(df))
np.random.seed(42)
df['last_update'] = np.random.permutation(date_range)

df.to_csv('Modified_AirQuality.csv', index=False)
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
df_cleaned = df.dropna(subset=['pollutant_id', 'pollutant_avg', 'last_update'])

pivot_df = df_cleaned.pivot_table(
    index=['country', 'state', 'city', 'station', 'last_update'],
    columns='pollutant_id',
    values='pollutant_avg'
).reset_index()
pivot_df.columns.name = None

def to_upper(col):
    return col.upper() if isinstance(col, str) else col

pivot_df = pivot_df.rename(columns=to_upper)

print("\nPivoted and Cleaned Data:")
print(pivot_df.head())
print("\nDescriptive Stats:")
print(pivot_df.describe())
print("\nInfo:")
print(pivot_df.info())

# Objective 2: Track Air Pollution Trends Over Time
if 'PM2.5' in pivot_df.columns:
    pm25_df = pivot_df[['LAST_UPDATE', 'PM2.5']].dropna().sort_values(by='LAST_UPDATE')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=pm25_df, x='LAST_UPDATE', y='PM2.5', marker='o')
    plt.title('Trend of PM2.5 Over Time')
    plt.xlabel('Date')
    plt.ylabel('PM2.5 Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if 'PM10' in pivot_df.columns:
    pm10_data = pivot_df[['LAST_UPDATE', 'PM10']].dropna()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=pm10_data, x='LAST_UPDATE', y='PM10', label='PM10', color='orange')
    plt.title('PM10 Trend Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Objective 3: Compare Pollutant Distributions using Seaborn
pollutant_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
available_pollutants = [col for col in pollutant_columns if col in pivot_df.columns]
melted_df = pivot_df[['LAST_UPDATE'] + available_pollutants].melt(id_vars='LAST_UPDATE', var_name='Pollutant', value_name='Level')

sns.set(style="whitegrid")

plt.figure(figsize=(16, 12))
for i, pollutant in enumerate(available_pollutants, 1):
    plt.subplot(3, 2, i)
    sns.histplot(data=pivot_df, x=pollutant, kde=True, bins=30,
                 color=sns.color_palette("husl")[i-1], stat="frequency",
                 element="step", fill=True, alpha=0.5, linewidth=1.5)
    plt.title(f'{pollutant} Distribution', fontsize=14)
    plt.xlabel('Level')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.suptitle("Individual Pollutant Level Distributions", fontsize=18, y=1.02)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Pollutant', y='Level', hue='Pollutant', data=melted_df, palette='Set2', legend=False,
            linewidth=2, fliersize=3, boxprops=dict(alpha=0.7))
plt.title('Pollutant Distribution - Boxplot', fontsize=16, fontweight='bold')
plt.xlabel('Pollutant')
plt.ylabel('Pollutant Level')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Objective 4 : Identify Pollution Hotspots Using Correlation Analysis

pivot_df = df.pivot_table(values='pollutant_avg', index=['city'], columns='pollutant_id', aggfunc='mean')
pivot_df.dropna(axis=0, how='all', inplace=True)
corr_matrix = pivot_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Pollutants Based on Average Values")
plt.tight_layout()
plt.show()

strong_corr = corr_matrix[(np.abs(corr_matrix) > 0.7) & (corr_matrix != 1.0)]
print("\nStrong pollutant correlations (|corr| > 0.7):")
print(strong_corr.dropna(how='all').dropna(axis=1, how='all'))

# Objective 5: Analyzing Seasonal Variation in Pollutants

df['LAST_UPDATE'] = pd.to_datetime(df['last_update'], errors='coerce')
df['Month'] = df['LAST_UPDATE'].dt.month
month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
df['Month'] = df['Month'].map(month_map)

monthly_avg = df.groupby(['Month', 'pollutant_id'])['pollutant_avg'].mean().reset_index()
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=month_order, ordered=True)
monthly_avg = monthly_avg.sort_values('Month')

plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_avg, x='Month', y='pollutant_avg', hue='pollutant_id', marker='o')
plt.title('Monthly Average Pollutant Levels')
plt.xlabel('Month')
plt.ylabel('Average Level (μg/m³)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Summer'
    elif month in [6, 7, 8]: return 'Monsoon'
    else: return 'Post-Monsoon'

monthly_avg['Season'] = monthly_avg['Month'].map(lambda x: get_season(pd.to_datetime(x, format='%b').month))
seasonal_avg = monthly_avg.groupby(['Season', 'pollutant_id'])['pollutant_avg'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=seasonal_avg, x='Season', y='pollutant_avg', hue='pollutant_id')
plt.title('Seasonal Average Pollutant Levels')
plt.xlabel('Season')
plt.ylabel('Average Level')
plt.legend(title='Pollutant')
plt.tight_layout()
plt.show()

# Objective 6: Identify the most polluted cities

city_pollution = df.groupby(['city', 'pollutant_id'])['pollutant_avg'].mean().reset_index()
pivot_city = city_pollution.pivot(index='city', columns='pollutant_id', values='pollutant_avg').fillna(0)
pivot_city['Total'] = pivot_city.sum(axis=1)
top_cities = pivot_city.sort_values(by='Total', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_cities.index, y=top_cities['Total'], hue=top_cities.index, palette='Reds_r', legend=False)
plt.title('Top Polluted Cities by Total Pollutant Level')
plt.xlabel('City')
plt.ylabel('Total Pollution Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 7: Correlation analysis between pollutants

numeric_df = df.select_dtypes(include='number')
cleaned_df = numeric_df.dropna(axis=1, how='all')
cleaned_df = cleaned_df.loc[:, (cleaned_df != cleaned_df.iloc[0]).any()]
correlation_matrix = cleaned_df.corr()

if not correlation_matrix.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Pollutant Correlation Matrix")
    plt.tight_layout()
    plt.show()
else:
    print("Correlation matrix is empty. Please check your dataset.")

# Objective 8: Air Quality Category Analysis

def categorize_aqi(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderate'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    else: return 'Severe'

df['AQI_Category'] = df['pollutant_avg'].apply(categorize_aqi)

if 'AQI_Category' in df.columns:
    category_counts = df['AQI_Category'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, hue=category_counts.index, palette='Spectral', legend=False)
    plt.title("Air Quality Category Distribution")
    plt.xlabel("AQI Category")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'AQI_Category' not found in the dataset.")

# Objective 9: Hypothesis Testing

pm25 = pivot_df['PM2.5'].dropna()
pm10 = pivot_df['PM10'].dropna()

min_len = min(len(pm25), len(pm10))
t_stat, p_val = ttest_rel(pm25[:min_len], pm10[:min_len])

print("\nHypothesis Test: PM2.5 vs PM10")
print("T-statistic:", t_stat)
print("P-value:", p_val)

if p_val < 0.05:
    print("Result: Statistically significant difference between PM2.5 and PM10 levels.")
else:
    print("Result: No statistically significant difference between PM2.5 and PM10 levels.")


    
    
