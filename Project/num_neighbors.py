import numpy as np
import pandas as pd
import sklearn.neighbors as skl_neighbors
import sklearn.metrics as sklm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

df = pd.read_csv('Project/training_data_vt2025.csv')
df['increase_stock'] = df['increase_stock'].apply(lambda x: 1 if x == 'high_bike_demand' else 0)

def data_k_fold(df, k_fold):
    # Shuffle the data
    df = shuffle(df)
    
    # Split the data into k folds
    df_folds = np.array_split(df, k_fold)

    df_matrix = []
    for i in range(k_fold):
        df_train = pd.DataFrame()
        for j in range(k_fold):
            if i != j:
                df_train = pd.concat([df_train, df_folds[j]])
        df_matrix.append([df_train, df_folds[i]])
    
    return df_matrix

def E_hold_out(model, test_data, columns):
    # Calculate the hold out error of the model
    y_pred = model.predict(test_data[columns])
    y_true = test_data['increase_stock']
    return sklm.mean_squared_error(y_true, y_pred)

def E_k_fold(model, df_matrix, columns):
    # Calculate the k-fold error of the model
    E = 0
    for df_train, df_test in df_matrix:
        model.fit(df_train[columns], df_train['increase_stock'])
        E += E_hold_out(model, df_test, columns)
    return E / len(df_matrix)

k_fold = 20
columns = ['hour_of_day', 'day_of_week', 'month', 'weekday', 'temp', 'dew', 'humidity', 'precip', 'snowdepth', 'windspeed', 'visibility', 'cloudcover']
E_dict = {}

for n in range(5, 150, 1):
    model = skl_neighbors.KNeighborsClassifier(n_neighbors=n)
    df_matrix = data_k_fold(df, k_fold)
    E = E_k_fold(model, df_matrix, columns)
    E_dict[n] = E
    print('n =', n, 'E =', E)

fig = plt.figure()
plt.plot(list(E_dict.keys()), list(E_dict.values()))
plt.title('KNN Classifier')
plt.xlabel('Number of neighbors')
plt.ylabel('Mean squared error (E_k_fold)')
plt.show()
