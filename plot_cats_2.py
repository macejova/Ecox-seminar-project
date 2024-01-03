import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def plot_categories_2(df, classif_var, unit, categories="all", fig_size=None, corrections=None,
                    show=True, give_fig=False):
    
    if categories == "all":
        cats = list(df[classif_var + "_category"].unique())
        if "OK" in cats:
            cats.remove("OK")
    else:
        cats = [cat for cat in categories if (df[classif_var + "_category"] == cat).any()]
    cols = [classif_var + "_" + cat for cat in cats]
    
    if fig_size is None: 
        fig_size = (10, 6)
    plt.figure(figsize=fig_size)
    plt.plot(df['date'], df[classif_var], label=classif_var, color='blue')  # Line plot for var1
    

    # Plot special categories as scatter plots with different markers
    markers = {"volatile_rain":'o', 'const_value':'s', 'outlier':'^', 'zero_value':'D','volatile':"*", 'prol_down': "v"}  # Define markers for each category (max 6 categories)
    colors = ["green", "brown", "red", "yellow", "orange", "black"]
    colors = {key: col for key, col in zip(markers.keys(), colors)}
    for i, column in enumerate(cols):
        cat = cats[i]
        if cat in ["volatile_rain", "volatile"]:
            plt.plot(df['date'], df[column], label=cat, linestyle='-', marker=markers[cat], color = colors[cat])
        else:
            plt.scatter(df['date'], df[column], label=cat, marker=markers[cat], color = colors[cat])
    
    if corrections is not None:
        plt.plot(df['date'], df[corrections], label=corrections, color='purple')  # Line plot for corrected data
    plt.xlabel('Date and Time')
    plt.ylabel(classif_var)
    plt.title(f'Time Series for {unit}')
    plt.legend()
    if show:
        plt.show()
    if give_fig:
        return plt.gcf()


#fig = plot_categories_2(explorer.TS_objects["MP3"].data.loc[s1[600:1000]], "prutok_computed", "example of Constant Values", 
#               categories=["zero_value", "const_value"], show=False, give_fig=True)
#fig.suptitle('Updated Title', fontsize=16)  # Change the title
#fig.get_axes()[0].set_ylabel('Flow' )  # Change the y-axis label
#fig.savefig('new_figure.png')

    
