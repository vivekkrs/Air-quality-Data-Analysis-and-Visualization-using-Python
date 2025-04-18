# Air Quality Data Analysis and Visualization using Python

## 🧠 Project Overview

This project aims to analyze and visualize real-time air quality data across various cities in India. By using Python libraries like **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn**, we perform **Exploratory Data Analysis (EDA)**, **trend tracking**, **seasonal and correlation analysis**, and **hypothesis testing** on pollutant levels.

The goal is to gain insights into:
- Pollution trends over time
- Seasonal effects on air quality
- Most polluted regions
- Correlations among various pollutants
- Categorization of air quality

---

## 📂 Dataset Source

- **Link:** [data.gov.in – Real-Time Air Quality Index at Various Locations](https://www.data.gov.in/resource/real-time-air-quality-index-various-locations)

---

## 🧾 Dataset Structure

The dataset is in CSV format and includes the following columns:

| Column Name       | Description                                                             |
|-------------------|-------------------------------------------------------------------------|
| `country`         | Country name (e.g., India)                                              |
| `state`           | State name                                                              |
| `city`            | City name                                                               |
| `station`         | Monitoring station name                                                 |
| `last_update`     | Timestamp of the observation                                            |
| `latitude`        | Latitude of the monitoring station                                      |
| `longitude`       | Longitude of the monitoring station                                     |
| `pollutant_id`    | Type of pollutant (e.g., PM2.5, PM10, SO2, CO, NO2, OZONE)              |
| `pollutant_min`   | Minimum concentration of the pollutant                                  |
| `pollutant_max`   | Maximum concentration of the pollutant                                  |
| `pollutant_avg`   | Average concentration of the pollutant (used for analysis and visuals)  |

---

## 🎯 Project Objectives and Visualizations

### ✅ Objective 1: Data Cleaning and Transformation
- Handle missing values
- Convert timestamps
- Pivot data for pollutant-wise analysis

### ✅ Objective 2: Track Air Pollution Trends Over Time
- Line plots showing the time-series trends of PM2.5 and PM10  
![Trend - PM2.5](/Figure_1.png)  
![Trend - PM10](/Figure_2.png)

### ✅ Objective 3: Pollutant Distribution Comparison
- Histogram and boxplot analysis for PM2.5, PM10, SO2, CO, NO2  
![Distribution Histograms](/fig_3.png)

### ✅ Objective 4: Correlation Analysis for Pollution Hotspots
- Heatmap to understand relationships among pollutants across cities  
![Correlation Heatmap](/Figure_4.png)

### ✅ Objective 5: Seasonal Variation in Pollutants
- Line plot for monthly variation
- Bar plot for seasonal averages  
![Seasonal Trend Line](/Figure_5.png)

### ✅ Objective 6: Identify the Most Polluted Cities
- Bar chart showing top 10 cities based on total pollution levels  
![Top Polluted Cities](/Figure_6.png)  
![Bar Plot](/Figure_7.png)

### ✅ Objective 7: Inter-Pollutant Correlation Analysis
- Numerical correlation matrix using heatmaps  
![Pollutant Correlation](/Figure_8.png)

### ✅ Objective 8: Air Quality Category Analysis
- AQI-based categorization: Good, Satisfactory, Moderate, Poor, etc.  
![AQI Category Distribution](/Figure_9.png)

### ✅ Objective 9: Hypothesis Testing
- T-test to compare PM2.5 and PM10 levels statistically  
![Hypothesis Testing Result](/Figure_10.png)

---

## 💡 Key Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`

---

## 📊 Summary

By analyzing and visualizing the dataset:
- We identified cities and seasons with the highest pollution levels.
- Found strong correlations between certain pollutants.
- Validated statistical differences using hypothesis testing.
- Provided clear, visual insights to make data-driven environmental decisions.

---

> 📌 **Note:** This analysis is based on historical and synthetic timestamp reshuffling for demonstration purposes. For real-time monitoring and policy-making, refer to official live data sources.
