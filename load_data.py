import numpy as np
import matplotlib.pyplot as plp
import pandas as pd
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('formatted_data.csv')
#print(dataset)

X = dataset.iloc[:,1:8].values
Y = dataset.iloc[:,0].values

#print(X)
#print(Y)
print(X[32,2])
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)
print(X[32,2])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#print(len(x_train))
#print(len(x_test))

#print("X_Training_Data")
#print(x_train)


#print("Y_Training_Data")
#print(y_train)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
