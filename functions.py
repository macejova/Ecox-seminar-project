"""Python script containing some useful functions"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

##SP - tinker with dummy variables
from statsmodels.formula.api import ols


def hr_func(ts):
    return ts.hour ##Goes

def create_cyclical_features(data):
    data["hour"] = data["date"].apply(hr_func)
    data["weekday"] = data["date"].apply(datetime.weekday)
    data["weekend"] = data["weekday"] >=5 
    data.describe()
    return data

def fit_cyclical_model(data, weekdays = True, var = "Prietok"):
    if weekdays:
        relev = "weekday"
    else:
        relev = "weekend"
    fit = ols(f"{var}~ C(hour) * C({relev})", data = data).fit()
    return(fit)

def get_cyclical_adjustment(data, fit, var = "Prietok"):
    if 'hour' not in data: ##Create cyclical features, if the don't already exist
        data = create_cyclical_features(data)
    data["cyclical_adjustment"] = fit.predict(data) - data[var].mean() 
    data["flow_cyclicaly_adjusted"] = data[var] - data["cyclical_adjustment"]
    return data

def cyclical_adj_full(data, weekdays = True, var = "Prietok", return_fit = False):
    data = create_cyclical_features(data)
    fit = fit_cyclical_model(data, weekdays, var)
    data = get_cyclical_adjustment(data, fit, var)

    if return_fit:
        return data, fit
    return data

def cyclical_adj_external(data, fit, var = "Prietok"):
    data = create_cyclical_features(data)
    data = get_cyclical_adjustment(data, fit, var)

    return data

