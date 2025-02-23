import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

#LOAD
rawdata = pd.read_csv('training_data_vt2025.csv')

#SEPARATE FEATURES AND TARGET + LABEL TARGET STRINGS: 
X = rawdata.drop('increase_stock', axis=1)
y = rawdata['increase_stock']
#It is intuitive that 1 implies 'yes' and 0 'no', to some action (stocking up on bikes):
encoder = {'low_bike_demand': 0, 'high_bike_demand': 1}
y_encoded = y.map(encoder)

#TO GET OPTIMAL COVARIANCE MATRIX, NORMALIZE ALL FEATURES TO SAME SCALE. SPLIT DATA: 
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# GRID SEARCH FOR OPTIMAL HYPERPARAMETER COMBO (USING ONLY SHRINKAGE FOR QDA): 
LDAmodel = LinearDiscriminantAnalysis()
hyperparamater_grid = [{'solver': ['svd']}, {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto']+list(np.linspace(0, 1, 1000))}]
gridsearch = GridSearchCV(LDAmodel, hyperparamater_grid, scoring='accuracy', n_jobs=-1, cv=10)
gridsearch.fit(X_train, y_train)
print('--------------------RESULTS OF GRID SEARCH------------------------')
print("Best hyperparameters:", gridsearch.best_params_)
print(f'Best cross-validated accuracy: {gridsearch.best_score_*100} %')
print('------------------------------------------------------------------')

#USE BEST MODEL ON TEST SET:  
bestmodel = gridsearch.best_estimator_
y_pred = bestmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confmatrix = confusion_matrix(y_test, y_pred)
print('--------------------EVALUATION OF BEST MODEL----------------------')
print('Confusion matrix:')
print(confmatrix)
print('Accuracy score:')
print(f'{100*accuracy} %')
print('------------------------------------------------------------------')
 

