import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.svm import SVR

def load_dataset():
    global X, Y
    dataset = pd.read_csv('formatted_data_original.csv')
    X = dataset.iloc[:,1:8].values
    Y = dataset.iloc[:,0].values
    Y = Y.reshape(-1,1)

def generate_missing_values():
    global X
    imp = Imputer('NaN', 'mean', 0)
    X=imp.fit_transform(X)

def scale_data():
    global X, Y, x_sc, y_sc
    from sklearn.preprocessing import StandardScaler
    x_sc = StandardScaler()
    x_sc.fit(X)
    X = x_sc.transform(X)
    y_sc = StandardScaler()
    y_sc.fit(Y)
    Y = y_sc.transform(Y)

def split_data():
    global x_train, x_test, y_train, y_test, X, Y
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

def train_model():
    global regressor
    regressor = SVR(kernel='rbf')
    regressor.fit(x_train, y_train)

def predict_value():
    global y_pred
    y_pred = y_sc.inverse_transform(regressor.predict(x_test))

def display():
    print(y_sc.inverse_transform(y_test))
    print(y_pred)

load_dataset()
generate_missing_values()
scale_data()
split_data()
train_model()
predict_value()
display()
