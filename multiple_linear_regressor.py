import numpy as np
import matplotlib.pyplot as plp
import pandas as pd
from sklearn.preprocessing import Imputer

def load_dataset():
    global dataset, X, Y
    dataset = pd.read_csv('formatted_data.csv')
    X = dataset.iloc[:,1:8].values
    Y = dataset.iloc[:,0].values

def generate_missing_values():
    global X
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imp.fit_transform(X)
    #print(X[32,2])

def split_data():
    global x_train, x_test, y_train, y_test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

def create_model():
    global regressor
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

def predict():
    print("ORIGINAL VALUES")
    print(y_test)
    print("PREDICTED VALUES")
    print(regressor.predict(x_test))

def display():
    print(X)
    print(x_train)
    print(x_test)
    print(Y)
    print(y_train)
    print(y_test)

load_dataset()
generate_missing_values()
split_data()
create_model()
predict()
#display()
