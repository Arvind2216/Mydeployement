from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def predict(request):
    slength = float(request.GET['SLength'])
    swidth = float(request.GET['SWidth'])
    plength = float(request.GET['PLength'])
    pwidth = float(request.GET['PWidth'])
    rawdata = staticfiles_storage.path('IRIS.csv')
    dataset = pd.read_csv(rawdata)
    X = dataset[["sepal_length","sepal_width","petal_length","petal_width"]]
    y= dataset["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=0)
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    yet_to_predict = np.array([[slength,swidth,plength,pwidth]])
    y_pred = regressor.predict(yet_to_predict)
    print(y_pred[0])
    accuracy = regressor.score(X_test, y_test)
    print(type(y_pred))

    return render(request,'index.html',{"predicted":y_pred[0]})

def home(request):
    return render(request,'index.html',{"predicted":""})
