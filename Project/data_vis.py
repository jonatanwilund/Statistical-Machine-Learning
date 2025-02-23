import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Load the data
df = pd.read_csv('Project/training_data_vt2025.csv')

# Plot the data
# Increase stock vs hour of day
fig = plt.figure(1)
sns.countplot(data=df, x='hour_of_day', hue='increase_stock', hue_order=['low_bike_demand', 'high_bike_demand'])
plt.title('Increase stock vs hour of day')
plt.xlabel('Hour of day')
plt.ylabel('Count of low or high demand')

# Increase stock vs day of week
fig = plt.figure(2)
sns.countplot(data=df, x='day_of_week', hue='increase_stock', hue_order=['low_bike_demand', 'high_bike_demand'])
plt.title('Increase stock vs day of week')
plt.xlabel('Day of week')
plt.ylabel('Count of low or high demand')

# Increase stock vs month
fig = plt.figure(3)
sns.countplot(data=df, x='month', hue='increase_stock', hue_order=['low_bike_demand', 'high_bike_demand'])
plt.title('Increase stock vs month')
plt.xlabel('Month')
plt.ylabel('Count of low or high demand')

# Increase stock vs weekday
fig = plt.figure(4)
sns.countplot(data=df, x='weekday', hue='increase_stock', hue_order=['low_bike_demand', 'high_bike_demand'])
plt.title('Increase stock vs weekday')
plt.xlabel('Weekday')
plt.ylabel('Count of low or high demand')

# Increase stock vs holiday
fig = plt.figure(5)
sns.countplot(data=df, x='holiday', hue='increase_stock', hue_order=['low_bike_demand', 'high_bike_demand'])
plt.title('Increase stock vs holiday')
plt.xlabel('Holiday')
plt.ylabel('Count of low or high demand')

# Increase stock vs temperature and humidity
fig = plt.figure(6)
sns.scatterplot(df, x='temp', y='humidity', hue='increase_stock', hue_order=['low_bike_demand', 'high_bike_demand'])
plt.title('Increase stock vs temperature and humidity')
plt.xlabel('Temperature')
plt.ylabel('Humidity')

# Increase stock vs temperature and precipitation
fig = plt.figure(7)
sns.scatterplot(df, x='temp', y='precip', hue='increase_stock', hue_order=['low_bike_demand', 'high_bike_demand'])
plt.title('Increase stock vs temperature and precipitation')
plt.xlabel('Temperature')
plt.ylabel('Precipitation')

# Increase stock vs summertime
fig = plt.figure(8)
sns.countplot(data=df, x='summertime', hue='increase_stock', hue_order=['low_bike_demand', 'high_bike_demand'])
plt.title('Increase stock vs summertime')
plt.xlabel('Summertime')
plt.ylabel('Count of low or high demand')

# Increase stock vs temperature and dew
fig = plt.figure(9)
sns.scatterplot(df, x='temp', y='dew', hue='increase_stock', hue_order=['low_bike_demand', 'high_bike_demand'])
plt.title('Increase stock vs temperature and dew')
plt.xlabel('Temperature')
plt.ylabel('Dew')

# Percentage of high demand holidays/non-holidays to number of holidays
num_holidays = df['holiday'].sum()
num_non_holidays = len(df) - num_holidays

num_high_demand_holidays = df[(df['holiday'] == 1) & (df['increase_stock'] == 'high_bike_demand')].shape[0]
num_high_demand_non_holidays = df[(df['holiday'] == 0) & (df['increase_stock'] == 'high_bike_demand')].shape[0]

print(f'Percantage of high demand holidays to number of holidays {num_high_demand_holidays/num_holidays*100:.2f}%')
print(f'Percantage of high demand non-holidays to number of non-holidays {num_high_demand_non_holidays/num_non_holidays*100:.2f}%')

# Percentage of high demand weekdays/non-weekdays to number of weekdays
num_weekdays = df['weekday'].sum()
num_non_weekdays = len(df) - num_holidays

num_high_demand_weekdays = df[(df['weekday'] == 1) & (df['increase_stock'] == 'high_bike_demand')].shape[0]
num_high_demand_non_weekdays = df[(df['weekday'] == 0) & (df['increase_stock'] == 'high_bike_demand')].shape[0]

print(f'Percantage of high demand weekdays to number of holidays {num_high_demand_weekdays/num_weekdays*100:.2f}%')
print(f'Percantage of high demand non-weekdays to number of non-weekdays {num_high_demand_non_weekdays/num_non_weekdays*100:.2f}%')

plt.show()