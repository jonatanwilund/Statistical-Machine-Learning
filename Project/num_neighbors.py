import numpy as np
import pandas as pd
import sklearn.neighbors as skl_neighbors
import sklearn.metrics as sklm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('Project/training_data_vt2025.csv')
df['increase_stock'] = df['increase_stock'].apply(lambda x: 1 if x == 'high_bike_demand' else 0)

training_data, test_data = train_test_split(df , test_size=0.3, random_state=42)

def data_k_fold(df, k_fold):
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
columns = ['hour_of_day', 'day_of_week', 'month', 'temp', 'dew', 'humidity', 'precip', 'snowdepth', 'windspeed', 'visibility', 'cloudcover']
E_dict = {}

'''
for n in range(5, 150, 5):
    model = skl_neighbors.KNeighborsClassifier(n_neighbors=n)
    df_matrix = data_k_fold(training_data, k_fold)
    E = E_k_fold(model, df_matrix, columns)
    E_dict[n] = (1-E)*100
    # print('n =', n, 'E =', E)

fig = plt.figure()
plt.plot(list(E_dict.keys()), list(E_dict.values()))
plt.title('KNN Classifier')
plt.xlabel('Number of neighbors')
plt.ylabel('Mean squared error (E_k_fold)')
plt.show()
'''

# Grid search for optimal hyperparameter: 
model = skl_neighbors.KNeighborsClassifier()
hyperparamater_grid = [{'n_neighbors': list(range(5, 150, 1))}]
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

# Own test: 
final_model = skl_neighbors.KNeighborsClassifier(n_neighbors=68)
final_model.fit(training_data[columns], training_data['increase_stock'])
y_pred = final_model.predict(test_data[columns])
accuracy = accuracy_score(test_data['increase_stock'], y_pred)
confmatrix = confusion_matrix(test_data['increase_stock'], y_pred)
print('-----------EVALUATION OF OWN MODEL----------')
print('Confusion matrix:')
print(confmatrix)
print('Accuracy score:')
print(f'{100*accuracy} %')
print('---------------------------------------------')