import sklearn.neighbors as skl_neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('Project/training_data_vt2025.csv')
df['increase_stock'] = df['increase_stock'].apply(lambda x: 1 if x == 'high_bike_demand' else 0)

training_data, test_data = train_test_split(df , test_size=0.3, random_state=42)

num_neighbors = 68

columns = ['hour_of_day', 'day_of_week', 'month', 'weekday', 'temp', 'dew', 'humidity', 'precip', 'snowdepth', 'windspeed', 'visibility', 'cloudcover']

model = skl_neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
model.fit(df[columns], df['increase_stock'])