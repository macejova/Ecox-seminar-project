
"""Python script containing some useful functions"""

import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

##SP - tinker with dummy variables
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans



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



def load_series(directory, files_list=False):
    """
    Imports all CSV files from a specified directory or list of files as Pandas DataFrames.
    Assumed csv files structure: 
        - two columns, no headers 
        - first column contains datetimes
        - second column contains variable of interest
        - name of the CSV file in the form 'sitename_variablename'
    

    Parameters:
    directory (str or list): If files_list=False (default), it represents the directory path containing CSV files.
                             If files_list=True, it is a list of file paths to individual CSV files.
    files_list (bool): Indicates whether the 'directory' parameter is a list of file paths (True) or a directory path (False).

    Returns:
    dict: A dictionary containing Pandas DataFrames, where keys are filenames and values are DataFrames.
    
    Raises:
    FileNotFoundError: If the specified directory does not exist.
    UserWarning: If a csv file does not satisfy the expected structure and cannot be loaded.
    
    """
    dataframes_dict = {}
    # Check if the directory exists
    if not files_list:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    
        files = os.listdir(directory)
    else:
        files = [path.split("\\")[-1] for path in directory]
        #directories = [path[: -len(file)] for path, file in zip(directory, files)]
        
        
    # Import each file as a separate DataFrame and assign headers
    for i, file_name in enumerate(files):
        if file_name.endswith('.csv'):  
            try:
                # Extract headers from the filename by splitting it by "_"
                headers = file_name.split("_")   # splitting sitename and variablename
                header = "_".join(headers[1:])
                header = header.split(".")[0]     # eliminating .csv suffix
                header = "_".join(header.split(" "))  # replacing " " with "_"
                headers = ["date", header]

                # Read the csv file into a DataFrame, specifying column names as headers
                if files_list:
                    file_path = directory[i]
                else:
                    file_path = os.path.join(directory, file_name)
                df = pd.read_csv(file_path, header=None, names=headers, sep=";")
                df["date"] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S')
    
                # Store the DataFrame in the dictionary with the filename as the key
                dataframes_dict[file_name] = df
            # raise warning if file does not have expected strucutre
            except (pd.errors.ParserError, pd.errors.EmptyDataError, pd.errors.DtypeWarning, ValueError) as e:
                warning_msg = f"The file '{file_name}' was not loaded: {str(e)}"
                warnings.warn(warning_msg, UserWarning)
                continue  
    return dataframes_dict



def merge_site_data(dataframes_dict):
    """
    Merge data from multiple DataFrames based on common date columns for each site. 
    Assumes input in the format of load_series function's output.

    Parameters:
    dataframes_dict (dict): A dictionary containing Pandas DataFrames where keys represent file names in the form 'sitename_etc'.

    Returns:
    dict: A dictionary containing merged Pandas DataFrames for each unique site found in the file names.
          Keys correspond to site names and values are merged DataFrames, joining on the 'date' column.

    Notes:
    This function extracts unique site names from the keys of the input dictionary. It creates a new DataFrame for each site
    and merges data from individual DataFrames (from dataframes_dict) having filenames starting with the site name.
    The merge is performed on the 'date' column using an outer join, ensuring all dates from various DataFrames are included.
    The resulting dictionary contains merged DataFrames for each unique site.
    
    """
    sites = set(file.split("_")[0] for file in dataframes_dict.keys()) # extract site names
    site_dataframes = {}
    
    for site in sites:
        df = pd.DataFrame(columns=['date'])  # create site-dataframe
        for file in dataframes_dict.keys():
            if file.startswith(site):   # move through dataframes corresponding to the current site
                site_df = dataframes_dict[file]
                df = pd.merge(df, site_df, on='date', how='outer')  # add data to the site-dataframe
        site_dataframes[site] = df
    return site_dataframes



def clean_data(datas_dictionary, main_var="prutok_computed"):
    """
    Clean data in Pandas DataFrames by removing rows with missing values in a specified main variable.

    Parameters:
    datas_dictionary (dict): A dictionary containing Pandas DataFrames where keys represent site names.
    main_var (str): The main variable used to clean the data by dropping rows with missing values in this column.
                    Defaults to 'prutok_computed' if not specified.

    Returns:
    dict: A dictionary containing 'cleaned' Pandas DataFrames for each site.
          'Cleaned' means that rows with missing values in the specified main variable are removed.

    Notes:
    This function cleans data by removing rows with missing values in the specified main variable.
    It iterates through each DataFrame in datas_dictionary and drops rows where the main variable has NaN values.
    The resulting dictionary contains cleaned DataFrames for each site without missing values in the main variable.
    
    """
    cleaned_datas = {}
    for site in datas_dictionary:
        df = datas_dictionary[site]
        df = df.dropna(subset=[main_var])  # drop NaNs of the main variable
        cleaned_datas[site] = df
    return cleaned_datas



def join_data(datas_dictionary):
    """
    Concatenate data from multiple Pandas DataFrames corresponding to different sites. Adds 'site' information to the resulting DF.

    Parameters:
    datas_dictionary (dict): A dictionary containing Pandas DataFrames where keys represent site names.

    Returns:
    pandas.DataFrame: A DataFrame resulting from concatenating the DataFrames in datas_dictionary.
                      The 'site' column is added to identify the source of each row.

    Notes:
    This function adds a 'site' column to each DataFrame in datas_dictionary to label the data from different sites.
    Then, it concatenates all DataFrames vertically using `pd.concat()`, resulting in a single DataFrame.
    The 'join' parameter is set to 'outer' to include all columns from different DataFrames.
    The resulting DataFrame contains data from all sites with an added 'site' column indicating the source site.
    
    """
    for site in datas_dictionary:
        datas_dictionary[site]["site"] = site  # add site name column
    result = pd.concat(list(datas_dictionary.values()), join='outer', ignore_index=True)
    return result



def join_series(indicators, tol=5, join=1):
    """
    Connects groups of successive ones/zeros in zero-one indicators if there are fewer than 'tol' zeros/ones separating them.

    Parameters:
    indicators (pandas.Series): A zero-one indicator series where 1 represents a condition and 0 represents its absence.
    tol (int): Maximum number of (1-'join')s allowed between successive 'join's to be connected. Defaults to 5.
    join (int): Value of which groups are to be connected. Must be either 0 or 1. Defaults to 1.

    Returns:
    pandas.Series: A modified indicator series where groups of successive ones/zeros are connected if separated by fewer than 'tol' zeros/ones.

    Notes:
    This function processes a zero-one indicator series and connects groups of successive ones, if join==1, if there are 
    less than 'tol' zeros between them, or vice versa if join == 0. It uses a cumulative sum approach to label successive 
    groups of ones (zeros), checks the count of ones (zeros) in each group, identifies groups to be joined based on 
    the specified 'tol' parameter, and updates the indicator series accordingly.
    
    """
    ones_groups = (indicators.diff() != 0).cumsum()  # identifies groups of successive indicators with the same value
    ones_counts = indicators.groupby(ones_groups).transform('size')   # calculates size of each group
    
    #  small groups of "delete" value are to be swallowed by their neighbouring groups of "join" value 
    groups_to_join = ones_counts < tol      
    delete = 1 - join
    new_indicators = indicators.mask(groups_to_join & (indicators == delete), join)
    return new_indicators



class TS_Class:
    """
    Time Series Class for handling and analyzing time series data.

    Parameters:
    data (pandas.DataFrame): The input time series data containing a 'date' column and the main variable.
    main_var (str): The main variable to analyze within the provided data. Defaults to "Prietok".
    start_date (str): The start date for the analysis. If not specified, defaults to the minimum date in the 'date' column.
    end_date (str): The end date for the analysis. If not specified, defaults to the maximum date in the 'date' column.
    periodicity (str): The desired periodicity for the time series data. Defaults to "2T" (2 minutes).
    check_per (bool): If True, ensures constant periodicity in observations. Defaults to True.

    Attributes:
    start_date (str): The start date of the time series analysis.
    end_date (str): The end date of the time series analysis.
    periodicity (str): The specified periodicity for the time series data.
    data (pandas.DataFrame): The time series data which is analyzed.
    main_var (str): The main variable to analyze within the provided data.
    models (dict): A dictionary to store models for the time series data.
    period_data (pandas.groupby): Grouped data based on the specified period.     *non-public?

    Methods:
    enforce_periodicity(start_date=None, end_date=None, periodicity=None):
        Ensures constant periodicity in observations within the specified start and end dates.
    
    get_y_lim(value, negative=False):                                             *non-public?
        Calculate y-axis limit based on the provided value.
        
    get_period_data(period, start_time=None, which=1, subset=None):               *non-public
        Retrieves data grouped by specified periods.

    """
    def __init__(self, data, main_var ="Prietok", start_date=None, end_date=None, periodicity="2T", check_per=True):
        if start_date is None:
            start_date = data["date"].min()
        self.start_date = start_date
        
        if end_date is None:
            end_date = data["date"].max()
        self.end_date = end_date
        
        self.periodicity = periodicity
        
        self.data = data
        if check_per:       # ensuring we have constant periodicity in observations
            self.enforce_periodicity(start_date, end_date, periodicity)
        
        self.check_per = check_per
        self.main_var = main_var
        self.models = {}
        # automatically add first and second differences of the main variable
        self.data[main_var + "_diff_1"] = data[main_var].diff()
        self.data[main_var + "_diff_2"] = self.data[main_var + "_diff_1"].diff()
        
    
    def enforce_periodicity(self, start_date=None, end_date=None, periodicity=None):
        """
        Ensures constant periodicity in class's data within specified datetimes interval and periodicity.

        Parameters:
        start_date (str): The start date for ensuring constant periodicity. If None, defaults to the class's start date.
        end_date (str): The end date for ensuring constant periodicity. If None, defaults to the class's end date.
        periodicity (str): The desired periodicity for ensuring constant periodicity. If None, defaults to the class's periodicity.

        Notes:
        This method modifies the class's data attribute to enforce constant periodicity.
        It generates a date range based on provided or default start and end dates using the specified or default periodicity
        and merges it with the existing data, deleting observations with non-conforming datetimes and creating NaN observations 
        for missing datetimes in the process. This ensures data is observed at regular intervals within desired datetime range.
        
        """
        if start_date is None: start_date = self.start_date
        if end_date is None: end_date = self.end_date
        if periodicity is None: periodicity = self.periodicity
        # create dataframe with prescribed datetimes 
        date_range = pd.date_range(start=start_date, end=end_date, freq=periodicity)
        df = pd.DataFrame({'date': date_range})
        # left join - creates NaN values for missing datetimes and deletes non-comforming datetimes
        self.data = df.merge(self.data, on='date', how='left')   
    
    def get_period_data(self, period, start_time = None, which = 1, subset = None):
        """
        Retrieves data grouped by specified datetime period and assigns it to period_data or period_data_2 attribute. 
        Denotes each group by its start and end datetimes or, for daily period, its date.
        
        Parameters:
        period (str or int): Length of the period for grouping data. Can be 'daily', 'weekly', or a whole number indicating number of hours.
        start_time (str): The start time for hourly grouping. Needs to be compatible with "%H:%M:%S" format and present in the class's data.
                        If None, defaults to the earliest timestamp in the class's data.
        which (int): Indicates to which attribute the grouped data are assigned. If 1 it uses self.period_data, else it uses
                    self.period_data_2. Defaults to 1.
        subset (list or None): Subset of indices to consider. If None, uses the entire dataset.
        
        Result:
        pandas.core.groupby.DataFrameGroupBy: A grouped DataFrame object based on the specified period. 
                                                Assigned to period_data or period_data_2 attribute.

        Raises:
        NameError: If an unrecognized period is entered. Expected an integer value, 'weekly' or 'daily'.
        
        """
        used_data = self.data.copy() if subset is None else self.data.iloc[subset,].copy()
        if type(period) == int:    # integer number of hours as period length 
            if start_time is None or start_time == "start":
                reference_point = used_data["date"].min()
            else:
                start_time = datetime.strptime(start_time, "%H:%M:%S")
                reference_point = used_data[used_data['date'].dt.time == start_time.time()]["date"].min() # choose the earliest observation with given time
                used_data = used_data[used_data["date"]>= reference_point]    # cut off observations before start_time
            # indicate for each observation in which period it lays and give periods their name based on their starting and end datetimes
            used_data["period_index"] = used_data["date"].apply(lambda x: (x-reference_point).total_seconds() // (period * 3600))
            used_data["period_name"] = used_data["period_index"].apply(
                lambda x:f"[{reference_point + timedelta(hours=period*x)} -- {reference_point + timedelta(hours=period*(x+1))})")
        
        elif period == "daily":
            used_data['period_name'] = used_data['date'].dt.date  # differentiate individual days
            
        elif period == "weekly":
            first = used_data['date'].dt.strftime('%Y-%W').apply(  # get the year and the week number and...
                lambda x: x.split("-")).apply(
                lambda y: datetime.strptime(f'{y[0]}-W{y[1]}-1', "%Y-W%U-%w") )  # get the date of Monday for given week
            used_data['period_name'] = first.apply(  # name each week as its first_day--last_day
                lambda x: f"{x.strftime('%Y-%m-%d')} -- {(x+timedelta(days=6)).strftime('%Y-%m-%d')}")
                       
        else:
            raise NameError('Period which is not recognized have been entered. Integer value, "weekly" or "daily" are expected.')
        
        if which == 1:
            self.period_data = used_data.groupby('period_name')
        else:
            self.period_data_2 = used_data.groupby('period_name')
        
    def get_y_lim(self, value, negative = False):
        """
        Calculate y-axis limit based on the provided value.

        Parameters:
        value (float): The input value for determining the y-axis limit.
        negative (bool): If True, the value is considered negative. Defaults to False.

        Returns:
        float: The calculated y-axis limit based on the provided value.

        Notes:
        This method calculates the y-axis limit based on the input value.
        If the 'negative' flag is True, the value is treated as negative.
        The method uses a predefined set of conditions to determine the appropriate y-axis limit.
        
        """
        if negative:
            value = - value
        if value < 0.05: return value*1.1
        if value < 0.09: return 0.1
        if value < 0.47: return 0.5
        if value < 0.9: return 1
        if value < 2.45: return 2.5
        if value < 4.9: return 5
        if value < 9.5: return 10
        if value < 19: return 20
        return value*1.1
    
    def get_rob_subset(self, variable, rob_quantile, two_sided):
        subset = self.data[variable] < self.data[variable].quantile(rob_quantile)
        if two_sided:
            L = self.data[variable].quantile(1-rob_quantile)
            U = self.data[variable].quantile(rob_quantile)
            subset =  self.data[variable].apply(lambda x: L<x<U)
        return subset
    
    def get_measures(self, variable, include, window, which = 1, 
                     quantile = 0.5, rob_quantile = 0.8, two_sided_rob_q = False, rob_q_mult = 1):   
        suff = "" if which == 1 else "_2"
        if two_sided_rob_q:
            rob_quantile = 0.5 + rob_quantile/2
        if "CMA" in include:
            self.data['Centered_Moving_Average'+suff] = self.data[variable].rolling(window=window, center=True).mean()
        if ("CMA_bounds" in include) or ("CMA_bounds_2sd" in include):
            self.data['Centered_Moving_Average'+suff] = self.data[variable].rolling(window=window, center=True).mean()
            self.data['Centered_Moving_SD'+suff] = self.data[variable].rolling(window=window, center=True).std()
            self.data['CMA_upper'+suff] = self.data['Centered_Moving_Average'+suff] + self.data['Centered_Moving_SD'+suff]
            self.data['CMA_lower'+suff] = self.data['Centered_Moving_Average'+suff] - self.data['Centered_Moving_SD'+suff]
            if "CMA_bounds_2sd" in include:
                self.data['CMA_upper_2sd'+suff] = self.data['Centered_Moving_Average'+suff] + 2*self.data['Centered_Moving_SD'+suff]
                self.data['CMA_lower_2sd'+suff] = self.data['Centered_Moving_Average'+suff] - 2*self.data['Centered_Moving_SD'+suff]
        if "CMA_rob_bounds"  in include:
            self.data['Centered_Moving_Average'+suff] = self.data[variable].rolling(window=window, center=True).mean()
            subset = self.get_rob_subset(variable, rob_quantile, two_sided_rob_q)
            robust_sd = self.data[subset][variable].std()
            self.data['CMA_upper_rob'+suff] = self.data['Centered_Moving_Average'+suff] + rob_q_mult*robust_sd
            self.data['CMA_lower_rob'+suff] = self.data['Centered_Moving_Average'+suff] - rob_q_mult*robust_sd
                
        if "MA" in include:
            self.data['Moving_Average'+suff] = self.data[variable].rolling(window=window, center=False).mean()
        if ("MA_bounds" in include) or ("MA_bounds_2sd" in include):
            self.data['Moving_Average'+suff] = self.data[variable].rolling(window=window, center=False).mean()
            self.data['Moving_SD'+suff] = self.data[variable].rolling(window=window, center=False).std()
            self.data['MA_upper'+suff] = self.data['Moving_Average'+suff] + self.data['Moving_SD'+suff]
            self.data['MA_lower'+suff] = self.data['Moving_Average'+suff] - self.data['Moving_SD'+suff]
            if "MA_bounds_2sd" in include:
                self.data['MA_upper_2sd'+suff] = self.data['Moving_Average'+suff] + 2*self.data['Moving_SD'+suff]
                self.data['MA_lower_2sd'+suff] = self.data['Moving_Average'+suff] - 2*self.data['Moving_SD'+suff]
        if "MA_rob_bounds" in include:
            self.data['Moving_Average'+suff] = self.data[variable].rolling(window=window, center=False).mean()
            subset = self.get_rob_subset(variable, rob_quantile, two_sided_rob_q)
            robust_sd = self.data[subset][variable].std()
            self.data['MA_upper_rob'+suff] = self.data['Moving_Average'+suff] + rob_q_mult*robust_sd
            self.data['MA_lower_rob'+suff] = self.data['Moving_Average'+suff] - rob_q_mult*robust_sd
        
        if "CMSD" in include:
            self.data['Centered_Moving_SD'+suff] = self.data[variable].rolling(window=window, center=True).std()
        if "MSD" in include:
            self.data['Moving_SD'+suff] = self.data[variable].rolling(window=window, center=False).std()
        if "tot_avg" in include:
            self.data['total_avg'+suff] = self.data[variable].mean()
        if "quant" in include:
            self.data['quantile'+suff] = self.data[variable].quantile(quantile)
        if "robust_avg" in include:
            subset = self.get_rob_subset(variable, rob_quantile, two_sided_rob_q)
            self.data['r_avg'+suff] = self.data[subset][variable].mean()
            
    
    def get_ax(self, ax, group_data, unit, variable, include, rain_lims, which, include_rain = True, marker = None):  
        suff = "" if which == 1 else "_2"
        ax.plot(group_data['date'], group_data[variable], label = variable)
        include_c = include.copy()   
        if "CMA" in include_c:
            ax.plot(group_data['date'], group_data['Centered_Moving_Average'+suff], color="red", label = "CMA")
            include_c.remove("CMA")
        
        if "CMA_bounds_2sd" in include_c:
            ax.plot(group_data['date'], group_data['CMA_upper_2sd'+suff], color="orange", label = "CMA_Up")
            ax.plot(group_data['date'], group_data['CMA_lower_2sd'+suff], color="orange", label = "CMA_L")
            include_c.remove("CMA_bounds_2sd")
        if "CMA_bounds" in include_c:
            ax.plot(group_data['date'], group_data['CMA_upper'+suff], color="orange", label = "CMA_Up")
            ax.plot(group_data['date'], group_data['CMA_lower'+suff], color="orange", label = "CMA_L")
            include_c.remove("CMA_bounds")
        
        if "CMA_rob_bounds" in include_c:
            ax.plot(group_data['date'], group_data['CMA_upper_rob'+suff], color="black", label = "CMA_rob_Up")
            ax.plot(group_data['date'], group_data['CMA_lower_rob'+suff], color="black", label = "CMA_rob_L")
            include_c.remove("CMA_rob_bounds")
            
        if "MA" in include_c:
            ax.plot(group_data['date'], group_data['Moving_Average'+suff], color="brown", label = "MA")
            include_c.remove("MA")
        if "MA_bounds" in include_c:
            ax.plot(group_data['date'], group_data['MA_upper'+suff], color="yellow", label = "MA_Up")
            ax.plot(group_data['date'], group_data['MA_lower'+suff], color="yellow", label = "MA_L")
            include_c.remove("MA_bounds")
        if "MA_bounds_2sd" in include_c:
            ax.plot(group_data['date'], group_data['MA_upper_2sd'+suff], color="yellow", label = "MA_Up")
            ax.plot(group_data['date'], group_data['MA_lower_2sd'+suff], color="yellow", label = "MA_L")
            include_c.remove("MA_bounds_2sd")
        if "MA_rob_bounds" in include_c:
            ax.plot(group_data['date'], group_data['MA_upper_rob'+suff], color="black", label = "MA_rob_Up")
            ax.plot(group_data['date'], group_data['MA_lower_rob'+suff], color="black", label = "MA_rob_L")
            include_c.remove("MA_rob_bounds")
            
        if "CMSD" in include_c:
            ax.plot(group_data['date'], group_data['Centered_Moving_SD'+suff], color="green", label = "CMSD")
            include_c.remove("CMSD")
        if "MSD" in include_c:
            ax.plot(group_data['date'], group_data['Moving_SD'+suff], color="black", label = "MSD")
            include_c.remove("MSD")
        if "tot_avg" in include_c:
            ax.plot(group_data['date'], group_data['total_avg'+suff], color="black", label = "tot_avg")
            include_c.remove("tot_avg")
        if "quant" in include_c:
            ax.plot(group_data['date'], group_data['quantile'+suff], color="black", label = "quantile")
            include_c.remove("quant")
        if "robust_avg" in include_c:
            ax.plot(group_data['date'], group_data['r_avg'+suff], color="black", label = "robust_avg")
            include_c.remove("robust_avg")
        
        max_y = group_data[variable].max()
        min_y = group_data[variable].min()
        #print(variable+str(min_y))    #########
        #print(variable+str(max_y))    #########
        #print(variable+" no-nan "+str(max_y))    #########
        for item in include_c:
            if item in group_data.columns:
                ax.plot(group_data['date'], group_data[item], label = item)
                max_y = max(max_y, group_data[item].max())
                min_y = min(min_y, group_data[item].min())
                #print(item+str(min_y))    #########
               # print(item+str(max_y))    #########
                        
        ax.set_title(f'Time Series for {unit}')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel(variable)
        ax.legend(loc="upper left")
        ylim_up = self.get_y_lim(max_y)
        ylim_down = -self.get_y_lim(min_y, True) if min_y < 0 else 0
 #       print("down "+str(ylim_down))            ##############
 #       print("up "+str(ylim_up))
        try:
            ax.set_ylim(ylim_down, ylim_up)
        except:
            print("down "+str(ylim_down))            ##############
            print("up "+str(min_y))
        for line in ax.lines:
            line.set_marker(marker)
            
        if include_rain:
            ax2 = ax.twinx()
            ax2.set_ylim(*rain_lims)
            ax2.plot(group_data["date"], group_data['rain_2m'], label='rain', color='green')
            ax2.invert_yaxis()
            ax2.set_ylabel('rain (Upside-Down)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='upper right')
            return (ax, ax2)
        return (ax, "")
        
    def plot(self, variable = None, period = "all", start_time = None, subset = None, 
             quantile = 0.5, rob_quantile = 0.8, two_sided_rob_q = False, rob_q_mult = 1,
             include = ["CMA", "CMA_bounds"], window = 30, rain_lims = (0,5), fig_size = None, 
            variable_2 = None, start_time_2 = None, include_2 = ["CMA", "CMA_bounds"], window_2 = 30, 
            quantile_2 = 0.5, rob_quantile_2 = 0.8, two_sided_rob_q_2 = False, rob_q_mult_2 = 1, 
             double_plot = False, include_rain = True, marker = None):
        if variable is None:
            variable = self.main_var
        
        self.get_measures(variable, include, window, which = 1, quantile = quantile, rob_quantile = rob_quantile, 
                          two_sided_rob_q = two_sided_rob_q, rob_q_mult = rob_q_mult)
        if start_time_2 is not None:
            if variable_2 is None: variable_2 = variable
            self.get_measures(variable_2, include_2, window_2, which = 2, quantile = quantile_2, 
                              rob_quantile = rob_quantile_2, two_sided_rob_q = two_sided_rob_q_2, rob_q_mult = rob_q_mult_2)
                
        if period == "all":
            self.data["const"] = "whole data"
            used_data = self.data.copy() if subset is None else self.data.iloc[subset,].copy()
            self.period_data = used_data.groupby("const")
        else:
            self.get_period_data(period, start_time, 1, subset)
        
        if start_time_2 is None:
            self.period_data_2 = self.period_data
        else:
            if period == "all":
                self.data["const"] = "whole data"
                used_data = self.data.copy() if subset is None else self.data.iloc[subset,].copy()
                self.period_data_2 = used_data.groupby("const")
            else:
                self.get_period_data(period, start_time_2, 2, subset)
            
        for grouped_1, grouped_2 in zip(self.period_data,self.period_data_2):
            unit_1, group_data_1 = grouped_1
            unit_2, group_data_2 = grouped_2
            
            if double_plot:
                group_data_1.set_index('date', inplace=True)
                group_data_2.set_index('date', inplace=True)
                group_data_1[variable].plot(xlabel = 'Date and Time', ylabel = variable, label = variable)
                group_data_2[variable_2].plot(label = variable_2)
                
                plt.title(f'Time Series for {unit_1}')
                plt.legend(loc="upper left")
                plt.show()
                continue
            
            if start_time_2 is not None:
                if fig_size is None: 
                    fig_size = (19, 6)
                fig, ax = plt.subplots(1, 2, figsize=fig_size)
                ax[0], axb1 = self.get_ax(ax[0], group_data_1, unit_1, variable, include, rain_lims, 1, include_rain, marker)
                ax[1], axb2 = self.get_ax(ax[1], group_data_2, unit_2, variable_2, include_2, rain_lims, 2, include_rain, marker)
            else:
                if fig_size is None: 
                    fig_size = (10, 6)
                fig, ax = plt.subplots(figsize=fig_size)
                ax, axb1 = self.get_ax(ax, group_data_1, unit_1, variable, include, rain_lims, 1, include_rain, marker)
            plt.grid(True)
            plt.show()
            
    def groupby(self, groupby):
        if groupby is None:
            self.data["const"] = "whole data"
            grouped = self.data.groupby("const")
        else:
            if groupby == "daily":
                self.data['period_name'] = self.data['date'].dt.date
                grouped = self.data.groupby("period_name")
            elif groupby == "weekly":
                self.data['period_name'] = self.data['date'].dt.strftime('%Y-%W')
                grouped = self.data.groupby("period_name")
            else:
                grouped = self.data.groupby(groupby)
        return grouped
    
    def get_ETS(self, variable, model_name, groupby = None, show_progress = True,
                error="add", trend="add", seasonal="add", damped_trend=False, seasonal_periods=720):
        """Note: there is 720 observations per day, for 1 day periodicity we need 720 seasonal components, which is too 
        many for practical estimatation - careful about seasonal components."""
        self.data[model_name + "_fitted"] = np.nan
        self.models[model_name] = []
        i = -1
        grouped = self.groupby(groupby)
                        
        length = len(grouped)
        for unit, grouped_data in grouped:
            model = ETSModel(grouped_data[variable], error=error, trend=trend, seasonal=seasonal, 
                                            damped_trend=damped_trend, seasonal_periods=seasonal_periods)
            result = model.fit()
            grouped_data[model_name + "_fitted"] = result.fittedvalues
            self.models[model_name].append((unit, model, results))
            self.data.loc[grouped_data.index,model_name + "_fitted"] = grouped_data[model_name + "_fitted"]
            if show_progress:
                i += 1
                if i%5 == 0:
                    print(f"{i+1} out of {length} ({(i+1)*100/length}%)")
                    
    def get_ARIMA(self, p, d, q, model_name, variable = None,  groupby = None, show_progress = True):
        if variable is None: variable = self.main_var
        self.data[model_name + "_fitted"] = np.nan
        self.models[model_name] = []
        i = -1
        grouped = self.groupby(groupby)           
        length = len(grouped)
        for unit, grouped_data in grouped:
            model = ARIMA(grouped_data[variable], order=(p, d, q))
            results = model.fit()
            grouped_data[model_name + "_fitted"] = results.fittedvalues
            self.models[model_name].append((unit, model, results))
            self.data.loc[grouped_data.index,model_name + "_fitted"] = grouped_data[model_name + "_fitted"]
            if show_progress:
                i += 1
                if i%5 == 0:
                    print(f"{i+1} out of {length} ({(i+1)*100/length}%)")
                    
    def ARIMA_diagnostics(self, results, unit, model_name, show_summary = True):
        print("ARIMA diagnostics for " + model_name + " " + str(unit))
        if show_summary:
            print(results.summary())
        residuals = results.resid
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0, 0].plot(residuals)
        ax[0, 0].set_title('Residuals')
        lags_1 = min(40, len(residuals)-1)
        lags_2 = min(40, int(len(residuals)/2)-2)
        sm.graphics.tsa.plot_acf(residuals, lags=lags_1, ax=ax[0, 1])
        sm.graphics.tsa.plot_pacf(residuals, lags=lags_2, ax=ax[1, 0])
        plt.show()

        # Ljung-Box test for autocorrelation in residuals
        acf, q_val, p_val = sm.tsa.acf(residuals, fft=True, qstat=True)
        p_val_df = pd.DataFrame({'Lag': range(1, len(p_val) + 1), 'P-Value': p_val})

        print(p_val_df)
    
    def ARIMAs_diagnostics(self, model_name, subset = None):
        models = self.models[model_name]
        if subset is not None:
            models = models[subset]
        for unit, mod, res in models:
            self.ARIMA_diagnostics(res, unit, model_name)
            
    def get_time_trend(self, values, give = "slope"):
        """Values should have a data series format with specified index which represents time order."""
        X = sm.add_constant(values.index)
        model = sm.OLS(values, X).fit()
        if give == "slope":
            return model.params[1]
        if give == "constant":
            return model.params[0]
        if give == "model":
            return model
        
    def copy(self):
        data_copy = self.data.copy()
        self_copy = TS_Class(data_copy, self.main_var, 
                 self.start_date, self.end_date, self.periodicity, self.check_per)
        return self_copy
    
    def classify(self, data = None, main_var = None, W_0 = 3
             , c_1 = 2.5, W_1 = 30
             , c_2 = 1, W_2 = 30, p_1 = 0.7
            , W_3 = 5, p_2 = 0.9
            , tol_vol_1 = 5, tol_vol_2 = 5
            , tol_rain_1 = 5, tol_rain_2 = 10):
                
        ret = True
        if data is None: 
            data = self.data
            ret = False
        if main_var is None:
            main_var = self.main_var
            
        data = classify(data, main_var, W_0, c_1, W_1, c_2, W_2, p_1, W_3, p_2
                        ,tol_vol_1, tol_vol_2, tol_rain_1, tol_rain_2)
        if not ret:
            self.data = data
        else:
            return data
    
    def plot_categories(self, categories = "all", main_var = None, only_events = False
                        ,period = "all", start_time = None, subset = None, fig_size = None):
        if main_var is None:
            main_var = self.main_var
        
        # TO DO: if only_events choose only days in which there is event from categories (not "OK")
        
        if period == "all":
            self.data["const"] = "whole data"
            used_data = self.data.copy() if subset is None else self.data.iloc[subset,].copy()
            self.period_data = used_data.groupby("const")
        else:
            self.get_period_data(period, start_time, 1, subset)
        
        for grouped in self.period_data:
            unit, group_data = grouped
            plot_categories(group_data, main_var, unit, categories, fig_size)
    



def classify(data, main_var = "prutok_computed"
             , W_0 = 3
             , c_1 = 2.5, W_1 = 30
             , c_2 = 1, W_2 = 30, p_1 = 0.7
            , W_3 = 5, p_2 = 0.9
            , tol_vol_1 = 5, tol_vol_2 = 5
            , tol_rain_1 = 5, tol_rain_2 = 10):
    data[main_var + "_category"] = "OK"
    
    # priority 5
    const = data[main_var].rolling(window=W_0, center=True).std() == 0
    data.loc[const, main_var + "_category"] = "const_value"
    
    # priority 4
    subset = data[main_var] <= data[main_var].quantile(p_1)
    sd_p_1 = data[subset][main_var].std()
    K = c_2*sd_p_1
    sd_2 = data[main_var].rolling(window=W_2, center=True).std()
    high_vol_orig = sd_2 > K
    high_vol = join_series(high_vol_orig.astype(int), tol = tol_vol_1, join = 1)  # group nearby islands of volatility
    high_vol = join_series(high_vol, tol = tol_vol_2, join = 0)  # delete small islands of volatility
    high_vol = high_vol.astype(bool)
    data.loc[high_vol, main_var + "_category"] = "volatile"
    
    # priority 3
    MA = data[main_var].rolling(window=W_3, center=True).mean()
    rain_threshold = data[main_var].quantile(p_2)
    rainy = MA >= rain_threshold
    high_vol_rain = high_vol_orig & rainy
    high_vol_rain = join_series(high_vol_rain.astype(int), tol = tol_rain_1, join = 1)  # group nearby islands of rain
    high_vol_rain = join_series(high_vol_rain, tol = tol_rain_2, join = 0)  # delete small islands of rain
    high_vol_rain = high_vol_rain.astype(bool)
    data.loc[high_vol_rain, main_var + "_category"] = "volatile_rain"
    
    # priority 2
    zeros = data[main_var] == 0
    data.loc[zeros, main_var + "_category"] = "zero_value"
    
    # priority 1
    first_diff = data[main_var].diff()
    first_diff_plus = first_diff.shift(-1)
    sd_1 = first_diff.rolling(window=W_1, center=True).std()
    T = c_1*sd_1
    outliers = ((first_diff > T) & (first_diff_plus < -T)) | ((first_diff < -T) & (first_diff_plus > T))
    data.loc[outliers, main_var + "_category"] = "outlier"
    
    dummies = pd.get_dummies(data[main_var + "_category"], prefix=main_var)
    for col in dummies.columns:
        data[col] = np.where(dummies[col] == 1, data[main_var], np.nan)
    #data = pd.concat([data, dummies], axis=1)
    return data



def plot_categories(df, main_var, unit, categories = "all", fig_size = None):
    
    if categories == "all":
        cats = list(df[main_var + "_category"].unique())
        if "OK" in cats:
            cats.remove("OK")
    else:
        cats = categories
    cols = [main_var + "_" + cat for cat in cats]
    
    if fig_size is None: 
        fig_size = (10, 6)
    plt.figure(figsize=fig_size)
    plt.plot(df['date'], df[main_var], label=main_var, color='blue')  # Line plot for var1

    # Plot special categories as scatter plots with different markers
    markers = {"volatile_rain":'o', 'const_value':'s', 'outlier':'^', 'zero_value':'D','volatile':"*"}  # Define markers for each category (max 5 categories)
    colors = ["green", "brown", "red", "yellow", "orange"]
    colors = {key: col for key, col in zip(markers.keys(), colors)}
    for i, column in enumerate(cols):
        cat = cats[i]
        if cat in ["volatile_rain", "volatile"]:
            plt.plot(df['date'], df[column], label=cat, linestyle='-', marker=markers[cat], color = colors[cat])
        else:
            plt.scatter(df['date'], df[column], label=cat, marker=markers[cat], color = colors[cat])

    plt.xlabel('Date and Time')
    plt.ylabel(main_var)
    plt.title(f'Time Series for {unit}')
    plt.legend()
    plt.show()



class Data_Explorer:
    
    def __init__(self, datas_dictionary, main_vars = None, 
                 start_dates = None, end_dates = None, periodicities = None, check_pers = None):
        datas_dictionary = {site: datas_dictionary[site] for site in sorted(datas_dictionary)}  # sort dictionary by keys
        self.data_dict = datas_dictionary
        self.sites = list(datas_dictionary.keys())
        self.joined_data = join_data(datas_dictionary)
        
        if main_vars is None: main_vars = {site: "prutok_computed" for site in self.sites}
        if start_dates is None: start_dates = {site: None for site in self.sites}
        if end_dates is None: end_dates = {site: None for site in self.sites}
        if periodicities is None: periodicities = {site: "2T" for site in self.sites}
        if check_pers is None: check_pers = {site: True for site in self.sites}
                
        self.TS_objects = {site: TS_Class(
            datas_dictionary[site], main_var = main_vars[site] 
            , start_date = start_dates[site], end_date = end_dates[site]
            , periodicity = periodicities[site], check_per = check_pers[site]
        ) for site in self.sites}
        
        self.TS_objects["joined"] = TS_Class(self.joined_data, main_var = main_vars[self.sites[0]]
                                            , start_date = start_dates[self.sites[0]], end_date = end_dates[self.sites[0]]
                                            , periodicity = periodicities[self.sites[0]], check_per = False)
        self.main_sites = self.sites.copy()
        self.sites.append("joined")
        self.data_dict["joined"] = self.joined_data
        
        self.main_vars = {site: self.TS_objects[site].main_var for site in self.sites}
        self.start_dates = {site: self.TS_objects[site].start_date for site in self.sites}
        self.end_dates = {site: self.TS_objects[site].end_date for site in self.sites}
        self.periodicities = {site: self.TS_objects[site].periodicity for site in self.sites}





