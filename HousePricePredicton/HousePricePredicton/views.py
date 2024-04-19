from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import load

model = load(
    '/Users/lindaberhe/Desktop/HousePricePredicton/HousePricePredicton/HousePricePredicton/trained_model.joblib')


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    if request.method == 'GET':
        bedrooms = float(request.GET['Bedrooms'])
        bathrooms = float(request.GET['Bathrooms'])
        garages = float(request.GET['Garages'])
        area = float(request.GET['Area'])

        data = pd.read_csv(
            "/Users/lindaberhe/Desktop/HousePricePredicton/HousePricePredicton/HousePricePredicton/columbus_houses.csv")

        x_test = [bedrooms, bathrooms, garages, area]
        x_test = np.array(x_test).reshape(1, -1)
        prediction = model.predict(x_test)
        print(prediction)

        price = f"The predicted price is ${prediction[0]:,.2f}"
        return render(request, 'predict.html', {"valuation_result": price})

    return render(request, 'predict.html')