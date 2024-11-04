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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

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



def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows = 1):
  
    max_num_test_batches = int(np.floor((len(test_data_for_an_engine) - window_length)/shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches -1)* shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data = None,
                                                                          window_length= window_length, shift = shift)
        return batched_test_data_for_an_engine, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data = None,
                                                                          window_length= window_length, shift = shift)
        return batched_test_data_for_an_engine, num_test_windows


for i in np.arange(1, num_test_machines + 1):
    temp_test_data = test_df[test_df[0] == i].drop(columns = [0]).values
    
    # Determine whether it is possible to extract test data with the specified window length.
    if (len(temp_test_data) < window_length):
        print("Test engine {} doesn't have enough data for window_length of {}".format(i, window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")
    
    # Prepare test data
    test_data_for_an_engine, num_windows = process_test_data(temp_test_data, window_length=window_length, shift = shift,
                                                             num_test_windows = num_test_windows)
    
    processed_test_data.append(test_data_for_an_engine)
    num_test_windows_list.append(num_windows)

processed_test_data = np.concatenate(processed_test_data)
true_rul = rul_df["RUL"].values

# Shuffle training data
index = np.random.permutation(len(processed_train_targets))
processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]

print("Processed training data shape: ", processed_train_data.shape)
print("Processed training ruls shape: ", processed_train_targets.shape)
print("Processed test data shape: ", processed_test_data.shape)
print("True RUL shape: ", true_rul.shape)



# def create_compiled_model():
#     # Define the LSTM model
#     model = Sequential()
#     model.add(
#         Bidirectional(
#             LSTM(
#                 50,
#                 return_sequences=True,
#                 input_shape = (window_length, 18),
#                 kernel_regularizer=regularizers.l2(0.01),  # Apply L2 regularization
#             )
#         )
#     )
#     model.add(Dropout(0.2))
#     model.add(Bidirectional(LSTM(50, kernel_regularizer=regularizers.l2(0.01))))  # Second LSTM layer with L2
#     model.add(Dropout(0.2))
#     model.add(Dense(1))  # Use linear activation for regression

#     # Compile the model
#     model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    
#     return model

# def scheduler(epoch):
#     return 0.001 if epoch < 5 else 0.0001

# # Define callbacks
# early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
# callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)

# model = create_compiled_model()

# # Train the model
# history = model.fit(processed_train_data, processed_train_targets, epochs = 10,
#                     validation_data = (processed_val_data, processed_val_targets),
#                     callbacks = callback,
#                     batch_size = 128, verbose = 2)



def create_compiled_model():
    model = Sequential([
        # CNN layers for spatial feature extraction
        layers.Conv1D(64, kernel_size=3, activation="relu", input_shape=(window_length, 18)),
        layers.Conv1D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=2),  # Downsampling to reduce the sequence length
        layers.Dropout(0.2),

        # LSTM layers for temporal pattern recognition
        layers.LSTM(128, return_sequences=True, activation="tanh"),
        layers.Dropout(0.2),
        layers.LSTM(64, activation="tanh", return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, activation="tanh"),
        layers.Dropout(0.2),

        # Fully connected layers
        layers.Dense(96, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        
        # Output layer
        layers.Dense(1)
    ])

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    
    return model


def scheduler(epoch):
    if epoch < 5:
        return 0.001
    else:
        return 0.0001
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)
model = create_compiled_model()
history = model.fit(processed_train_data, processed_train_targets, epochs = 10,
                    validation_data = (processed_val_data, processed_val_targets),
                    callbacks = callback,
                    batch_size = 128, verbose = 2)


rul_pred = model.predict(processed_test_data).reshape(-1)
preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows)) 
                             for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)]
RMSE = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine))
print("RMSE: ", RMSE)

plt.plot(true_rul, label = "True RUL", color = "orange")
plt.plot(mean_pred_for_each_engine, label = "Pred RUL", color = "blue")
plt.legend()
plt.show()













