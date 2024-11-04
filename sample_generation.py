# Importing frameworks/packages that are required for the model to run

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
import torch
from torch import nn, optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from pylab import rcParams
import math
import time
from tqdm import tqdm
import keras.models
import keras.layers
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

#File paths
train_path = "../CMAPSS_GRP_2/dataset/train_FD001.txt"
test_path = "../CMAPSS_GRP_2/dataset/test_FD001.txt"
rul_path = "../CMAPSS_GRP_2/dataset/RUL_FD001.txt"

# Define column names
columns = [
    "unit_ID",
    "cycles",
    "setting_1",
    "setting_2",
    "setting_3",
    "T2",
    "T24",
    "T30",
    "T50",
    "P2",
    "P15",
    "P30",
    "Nf",
    "Nc",
    "epr",
    "Ps30",
    "phi",
    "NRf",
    "NRc",
    "BPR",
    "farB",
    "htBleed",
    "Nf_dmd",
    "PCNfR_dmd",
    "W31",
    "W32",
]

truth_columns = [
    "unit_ID",
    "RUL",
]


def load_and_clean_data(train_path, test_path, rul_path, columns, truth_columns):
    
    # Load datasets
    train_df = pd.read_csv(train_path, sep=" ", header=None)
    test_df = pd.read_csv(test_path, sep=" ", header=None)
    rul_df = pd.read_csv(rul_path, sep=" ", header=None)

    # Drop NaN columns and rename columns,there exists 2 Nan columns after the 26th column
    train_df.dropna(axis=1, inplace=True)
    test_df.dropna(axis=1, inplace=True)
    rul_df.dropna(axis=1, inplace=True)
    rul_df.insert(0, "unit_ID", range(1, len(rul_df) + 1))

    train_df.columns = columns
    test_df.columns = columns
    rul_df.columns = truth_columns

    return train_df, test_df, rul_df

# 1. Load and clean data
train_df, test_df, rul_df = load_and_clean_data(
    train_path, test_path, rul_path, columns, truth_columns)


# 2. Feature Selection: Remove redundant columns based on observations from sensor measurement plots
columns_to_remove = ["T2", "P2", "P15", "epr", "farB", "Nf_dmd", "PCNfR_dmd"]
train_df.drop(columns=columns_to_remove, inplace=True)
test_df.drop(columns=columns_to_remove, inplace=True)

# 3. Parameters for sliding windows,shifts,early RUL and empty lists to store generated sample data from using shifting window
window_length = 30
shift = 1
early_rul = 125           
processed_train_data = []
processed_train_targets = []
num_test_windows = 5     
processed_test_data = []
num_test_windows_list = []

train_data_first_column = train_df["unit_ID"]
test_data_first_column = test_df["unit_ID"]



# 4. Perform Min Max scalar with a desired range between 0 and 1.

# Initialize the MinMaxScaler with a desired range
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the data
train_df = scaler.fit_transform(train_df.drop(columns=['unit_ID']))
test_df = scaler.transform(test_df.drop(columns=['unit_ID']))

train_df = pd.DataFrame(data = np.c_[train_data_first_column, train_df])
test_df = pd.DataFrame(data = np.c_[test_data_first_column, test_df])

num_train_machines = len(train_df[0].unique())
num_test_machines = len(test_df[0].unique())


# 5. Using sliding window to generate a new test data from the existing test data.

def process_targets(data_length, early_rul = None):
    if early_rul == None:
        return np.arange(data_length-1, -1, -1)
    else:
        early_rul_duration = data_length - early_rul
        if early_rul_duration <= 0:
            
            # Return decreasing RUL based on length of train data if RUL is lesser than specified early RUL (of 125)
            return np.arange(data_length-1, -1, -1)
        else:
            # Return a RUL of constant value (of 125) until it reaches the point when RUL starts to drop below 125.
            return np.append(early_rul*np.ones(shape = (early_rul_duration,)), np.arange(early_rul-1, -1, -1))
        
    
def process_input_data_with_targets(input_data, target_data = None, window_length = 1, shift = 1):
    num_batches = int(np.floor((len(input_data) - window_length)/shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats = num_batches * window_length * num_features).reshape(num_batches, window_length,
                                                                                                  num_features)
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch,:,:] = input_data[(0+shift*batch):(0+shift*batch+window_length),:]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats = num_batches)
        for batch in range(num_batches):
            output_data[batch,:,:] = input_data[(0+shift*batch):(0+shift*batch+window_length),:]
            output_targets[batch] = target_data[(shift*batch + (window_length-1))]
        return output_data, output_targets


for i in np.arange(1, num_train_machines + 1):
    
    # Temporarily getting train data that belongs to a certain engine number and dropping its engine ID column
    temp_train_data = train_df[train_df[0] == i].drop(columns = [0]).values
    
    # Determine whether it is possible to extract training data with the specified window length.
    if (len(temp_train_data) < window_length):
        print("Train engine {} doesn't have enough data for window_length of {}".format(i, window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")
    
    # Generating a RUL target column for every entry in the temporarily train data
    temp_train_targets = process_targets(data_length = temp_train_data.shape[0], early_rul = early_rul)
    
    # Generating a new sample using sliding window for every engine.
    data_for_a_machine, targets_for_a_machine = process_input_data_with_targets(temp_train_data, temp_train_targets, 
                                                                                window_length= window_length, shift = shift)
    # Appending the aggregated train data for every engine to a list.
    processed_train_data.append(data_for_a_machine)
    
    # Appending the aggregated RUL target data for every engine to a list.
    processed_train_targets.append(targets_for_a_machine)

# Compiling the newly generated data (for train and RUL) from sliding window into a dataframe.
processed_train_data = np.concatenate(processed_train_data)
processed_train_targets = np.concatenate(processed_train_targets)


# 6. Split the processed training data into 2, training and validation sets
processed_train_data, processed_val_data, processed_train_targets, processed_val_targets = train_test_split(processed_train_data,
                                                                                                            processed_train_targets,
                                                                                                            test_size = 0.2,
                                                                                                            random_state = 83)
print("Processed train data shape: ", processed_train_data.shape)
print("Processed validation data shape: ", processed_val_data.shape)
print("Processed train targets shape: ", processed_train_targets.shape)
print("Processed validation targets shape: ", processed_val_targets.shape)





