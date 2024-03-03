'''
Author: Alexander Macejovsky, Stepan Pardubicky
Date: 29.2.2024

This is script for using tools for automated classification and correction of errors in TS.
It is expected that examined TS will be provided in dfs0 file with 1 column and observation date-times as row indeces.


Parameters:
The script requires indicators of whether to print debug messages and whether to only print parameters, and parameters 
denoting names of input and output dfs0 files. 
Additionally, parameters indicating characteristics of concerned TS and customized parameters for classification and 
correction of errors can be provided. Use parameters only option and see relevant documentation for more information 
on the latter.
Note: Leave parameters for characteristics of concerned TS and for classification and correction of errors blank if you
        want to use their default values.
   
Output:
The output dataset contains, in addition to the original values of the examined TS, these new columns (main_var stands for 
the name of the examined TS):
    - main_var_corrected (numeric): corrected values of TS
    - main_var_OK (boolean): 0-1 indicator of whether the original observation was classified as correct
    - main_var_outlier (boolean): -||- as outlier
    - main_var_prol_down (boolean): -||- as prolonged drop
    - main_var_zero_value (boolean): -||- as zero value
    - main_var_volatile_rain (boolean): -||- as volatile rain
    - main_var_volatile (boolean): -||- as volatile
    - main_var_const_value (boolean): -||- as constant value
    
 ---- TO DO ---- 
* establish agreed upon conventions for naming of output variables
* verify whether volatility should be smoothed or not by default (latter as of now)
'''

############ Parameters:
TS_chars_pars = ["start_date", "end_date", "periodicity", "check_per"] #date, date, nT, bool
classif_pars = ["W_0", "tol_const","c_1", "W_1","c_2", "W_2", "p_1","W_3", "p_2","tol_vol_1", "tol_vol_2",
                "tol_rain_1", "tol_rain_2","volatile_diffs","num_back", "num_fol","fol_tresh", "W_4", "c_3"] # volatile_diffs=bool, other num
corr_pars = ["outliers_window", "volatility_window", "corr_vol"]  # corr_vol=boolean

parameterNames = TS_chars_pars + classif_pars + corr_pars
TS_chars_pars_dict, classif_pars_dict, corr_pars_dict  = {}, {}, {}  # for creation of **kwargs
boolean_pars = ["check_per", "volatile_diffs", "corr_vol"]
############ End of parameters

import sys, subprocess
from DHI_functions import *   ##

# read arguments
args = sys.argv

debug = int(args[1])
onlyParameters = int(args[2])

# return parameter names (when GND button "Get parameters" clicked)
if (onlyParameters == 1):
    for p in parameterNames:
        print(p)
    sys.exit()

# when executing script
if onlyParameters==0:
    try:
        import mikeio
    except ImportError as e:
        if debug:
            print("Package mikeio not installed. Installing...")
        subprocess.check_call("pip install mikeio")
        import mikeio

    # read arguments
    inputFileName = args[3]
    outputFileName = args[4]
    params = args[5:]

    if debug:
        print("Python engine:")
        print(sys.executable)
        print("Running script:")
        print(args[0])


## read from inputs non-default parameters
for i, p in enumerate(params):
    if str(p).strip() == "":    # assuming that leaving parameters blank in GF leads to empty string -> use default value of parameter
        params[i] = "default"

# assign parameters to correct parameter dictionary
n1 = len(TS_chars_pars)
n2 = len(TS_chars_pars) + len(classif_pars_dict)
TS_chars_pars_dict = {parameterNames[i]: p for i, p in enumerate(params) if (i<n1 and p!="default")}
classif_pars_dict = {parameterNames[i]: p for i, p in enumerate(params) if (n1<=i and i<n2 and p!="default")}
corr_pars_dict = {parameterNames[i]: p for i, p in enumerate(params) if (n2<=i and p!="default")}

# change 0-1 to boolean type if parameter is boolean
for key, val in TS_chars_pars_dict.items():
    if key in boolean_pars:
        TS_chars_pars_dict[key] = bool(val)
for key, val in classif_pars_dict.items():
    if key in boolean_pars:
        classif_pars_dict[key] = bool(val)
for key, val in corr_pars_dict.items():
    if key in boolean_pars:
        corr_pars_dict[key] = bool(val)

non_default = list(TS_chars_pars_dict.keys()) + list(classif_pars_dict.keys()) + list(corr_pars_dict.keys())


if debug == 1:
    if len(parameterNames) == len(params):
        print("\nParameters used:")
        for i in range(0,len(parameterNames)):
            print(parameterNames[i] + " = " + params[i])
        
        print("Parameters with non-default values:")     ##
        for p in non_default:
            print(p)
    else:
        raise ValueError("Length of parameters and parameter names differs!")

############ BODY (edit script as needed):           TO DO
    
# read input dfs0
inputDfs = mikeio.Dfs0(inputFileName)

# read input item info
inputItemInfo = inputDfs.items[0]
if debug:
    print("\nInput item info")
    print(inputItemInfo)

# read type and units of input TS
inputType = inputDfs.items[0].type
inputUnits = inputDfs.items[0].unit

# print header
if debug:
    print("\nInput TS type: ")
    print(inputType)    
    print("\nInput TS unit: ")
    print(inputUnits)

# extract data to dataframe
inputDataframe = inputDfs.to_dataframe()

if debug:
    print("\nInput TS preview:")
    print(inputDataframe.head())

## prepare correct format of dataframe
main_var = inputDataframe.columns[0]    # assuming examined TS is the first column of the dataframe
inputDataframe.reset_index(inplace=True)
inputDataframe.rename(columns={'index': 'date'}, inplace=True) # assuming observation datetimes are the index of the dataframe

# Set the 'Date' column as a datetime type
inputDataframe['date'] = pd.to_datetime(inputDataframe['date'])

## perform classification and correction 
TS_chars_pars_dict["main_var"] = main_var
TS_object = TS_Class(inputDataframe, **TS_chars_pars_dict)

TS_object.classify(**classif_pars_dict)

data =  correct_data(TS_object.data, TS_object.main_var, **corr_pars_dict)

## create output dataframe
outputDataframe = data[[main_var, "date"]]

outputDataframe[main_var + "_corrected"] = data[main_var + "_corrected_"]

# Item info
items_info = [inputItemInfo, inputItemInfo] # for original and corrected TS

dummies = pd.get_dummies(data[main_var + "_category"], prefix="")  # columns of dummies named as "_CategoryName"
for category in dummies:
    outputDataframe[main_var + category] = dummies[category].astype(int)  # 0-1 indicators
    items_info.append(mikeio.ItemInfo(main_var + category))

outputDataframe.set_index('date', inplace=True)  # move date back to index

# print header
if debug:
    print("\nOutput TS preview:")
    print(outputDataframe.head())

# output item info:
outputItemInfo = inputItemInfo
if debug:
    print("\nOutput item info:")
    print(outputItemInfo)


############ End of BODY

#write output dfs0
##outputDfs = mikeio.Dfs0()
##outputDfs.from_dataframe(outputDataframe, outputFileName, items=[outputItemInfo])

outputDataframe.to_dfs0(outputFileName, items=items_info)

print ("Succesfully finished")
