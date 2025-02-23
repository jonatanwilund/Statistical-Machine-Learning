import sklearn.neighbors as skl_neighbors
import pandas as pd

df = pd.read_csv('Project/training_data_vt2025.csv')
df['increase_stock'] = df['increase_stock'].apply(lambda x: 1 if x == 'high_bike_demand' else 0)

num_neighbors = 68

columns = ['hour_of_day', 'day_of_week', 'month', 'weekday', 'temp', 'dew', 'humidity', 'precip', 'snowdepth', 'windspeed', 'visibility', 'cloudcover']
model = skl_neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
model.fit(df[columns], df['increase_stock'])