import numpy as np
import pandas as pd
import sklearn.neighbors as skl_neighbors
import sklearn.metrics as sklm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Project/training_data_vt2025.csv')
df['increase_stock'] = df['increase_stock'].apply(lambda x: 1 if x == 'high_bike_demand' else 0)

df['is_summer'] = df['month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
df['is_fall'] = df['month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)
df['is_winter'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
df['is_spring'] = df['month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
df['high_activity'] = df['hour_of_day'].apply(lambda x: 1 if 15 <= x <= 19 else 0)
df['middle_activity'] = df['hour_of_day'].apply(lambda x: 1 if 7 <= x <= 14 or x == 20 else 0)
df['low_activity'] = df['hour_of_day'].apply(lambda x: 1 if x >= 20 or x <= 7 else 0)
df['is_there_snow'] = df['snowdepth'].apply(lambda x: 1 if x > 0 else 0)

training_data, test_data = train_test_split(df , test_size=0.3, random_state=42)

columns = ['hour_of_day', 'day_of_week', 'month', 'temp', 'dew', 'humidity', 'precip', 'snowdepth', 'windspeed', 'visibility', 'cloudcover']

# Grid search for optimal hyperparameter: 
model = skl_neighbors.KNeighborsClassifier()
hyperparamater_grid = [{'n_neighbors': list(range(5, 80, 1)), 'weights': ['uniform', 'distance'], 'p': [1, 2]}]
gridsearch = GridSearchCV(model, hyperparamater_grid, scoring='accuracy', n_jobs=-1, cv=10)
gridsearch.fit(training_data[columns], training_data['increase_stock'])
print('--------------------RESULTS OF GRID SEARCH------------------------')
print("Best hyperparameters:", gridsearch.best_params_)
print(f'Best cross-validated accuracy: {gridsearch.best_score_*100} %')
print('------------------------------------------------------------------')

# Use best model on test set: 
final_model = gridsearch.best_estimator_
y_pred = final_model.predict(test_data[columns])
accuracy = accuracy_score(test_data['increase_stock'], y_pred)
confmatrix = confusion_matrix(test_data['increase_stock'], y_pred)
print('-----------EVALUATION OF BEST MODEL----------')
print('Confusion matrix:')
print(confmatrix)
print('Accuracy score:')
print(f'{100*accuracy} %')
print('---------------------------------------------')

# Validation curve
train_scores, valid_scores = validation_curve(
    model, training_data[columns], training_data['increase_stock'],
    param_name='n_neighbors', param_range=list(range(5, 151)), cv=10, scoring='accuracy'
)

# Plot validation curve
plt.figure(2)
plt.plot(range(5, 151), np.mean(train_scores, axis=1), label='Training score')
plt.plot(range(5, 151), np.mean(valid_scores, axis=1), label='Validation score')
plt.title('Validation Curve with KNN')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()
