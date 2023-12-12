"""Python script containing some useful functions"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

##SP - tinker with dummy variables
from statsmodels.formula.api import ols


def hr_func(ts):
    """Returns the hour from ts"""
    return ts.hour 

def min_func(ts):
    """Returns the current minute from ts"""
    return(ts.minute)

def create_cyclical_features(data):
    """Creates categorical variables representing current hour and day of the week or weekday/weekend"""
    data["minute"] = data["date"].apply(min_func)
    data["hour"] = data["date"].apply(hr_func)
    data["weekday"] = data["date"].apply(datetime.weekday)
    data["weekend"] = data["weekday"] >=5 
    data.describe()
    return data

def fit_cyclical_model(data, weekdays = True, var = "Prietok"):
    """Fits a simple LR model to predict values for categorical variables used to eliminate cyclcity"""
    if weekdays:
        relev = "weekday"
    else:
        relev = "weekend"
    fit = ols(f"{var}~ C(hour) * C({relev})", data = data).fit()
    return(fit)

def get_cyclical_adjustment(data, fit, var = "Prietok"):
    """Uses the fitted model to seasonaly adjust"""
    if 'hour' not in data: ##Create cyclical features, if the don't already exist
        data = create_cyclical_features(data)
    data["cyclical_adjustment"] = fit.predict(data) - data[var].mean() 
    data["flow_cyclicaly_adjusted"] = data[var] - data["cyclical_adjustment"]
    return data

def cyclical_adj_full(data, weekdays = True, var = "Prietok", return_fit = False):
    """One function for the whole workload"""
    data = create_cyclical_features(data)
    fit = fit_cyclical_model(data, weekdays, var)
    data = get_cyclical_adjustment(data, fit, var)

    if return_fit:
        return data, fit
    return data

def cyclical_adj_external(data, fit, var = "Prietok"):
    """Optionality to use external fit"""
    data = create_cyclical_features(data)
    data = get_cyclical_adjustment(data, fit, var)

    return data

#Cooking up modification to smooth out the cyclicity
def apply_cyclical_smoothing(data,model = None,  var = "Prietok", weekdays = True, num = 1):
    """TODO: Write description"""
    if'hour' not in data:
        data = create_cyclical_features(data)
    if model == None:
        model = fit_cyclical_model(data, weekdays, var)
    if 'cyclical_adjustment' not in data:
        data = cyclical_adj_external(data, model, var = "Prietok")
    data["cyclical_adjustment_smoothed"] = data.apply(cyclical_smoothing, fit = model, num = num, axis=1)

def cyclical_smoothing(row, fit, num = 1):
    """Computes smoothed values of cyclical component for a single row"""
    current_val = row["cyclical_adjustment"] 
    if row["minute"] >= 58 - 2*num:
        left_val = current_val
        right_val = fit.predict(row.to_frame().transpose().assign(hour = row["hour"]+1)%24)
        return (smoothing(left_val, right_val, row["minute"], num))
    if row["minute"]<= 0 +2*num:
        right_val = current_val
        left_val = fit.predict(row.to_frame().transpose().assign(hour =  row["hour"]-1)%24)
        return (smoothing(left_val, right_val, row["minute"], num))
    
    return current_val


def smoothing(left_val, right_val, minute, num = 1 ):
    """Computes the weighed seasonality"""
    if minute >= 58 - 2*num:
        left_weight = (num + (58-minute)/2 + 1)
        right_weight = 2*num + 1 - left_weight
    if minute<= 0 +2*num:
        right_weight = (num + (minute)/2 + 1)
        left_weight = 2*num + 1 - left_weight
    return (left_val*left_weight + right_val*right_weight)/(2*num+1)


    


