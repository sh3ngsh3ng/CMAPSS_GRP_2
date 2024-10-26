import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
column_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 27)]
train_df = pd.read_csv('./dataset/train_FD001.txt', sep='\s+', header=None, names=column_names)
test_df = pd.read_csv('./dataset/test_FD001.txt', sep='\s+', header=None, names=column_names)
true_rul = pd.read_csv('./dataset/RUL_FD001.txt', header=None)
train_df = train_df.dropna(axis=1, how="all")
test_df = test_df.dropna(axis=1, how="all")

rng = np.random.RandomState(42)

from sklearn.model_selection import train_test_split
engine_ids = train_df['engine_id'].unique()
train_engine_ids, test_engine_ids = train_test_split(engine_ids, test_size=0.2, random_state=42)


train_data = train_df[train_df['engine_id'].isin(train_engine_ids)]
test_data = train_df[train_df['engine_id'].isin(test_engine_ids)]

columns_to_drop = ["setting3", "sensor1", "sensor5", "sensor10", "sensor16", "sensor19"]
train_df_dropped = train_df.drop(columns=columns_to_drop)

# Normalization
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Separate the columns to normalize and the columns to skip
columns_to_skip = train_df_dropped.columns[:2]
columns_to_normalize = train_df_dropped.columns[2:]

# Normalize only the selected columns
normalized_data = scaler.fit_transform(train_df_dropped[columns_to_normalize])

# Combine the normalized and unnormalized columns
train_df_normalized = pd.DataFrame(train_df_dropped[columns_to_skip].values, columns=columns_to_skip)
train_df_normalized = pd.concat([train_df_normalized, pd.DataFrame(normalized_data, columns=columns_to_normalize)], axis=1)

train_df_normalized['RUL'] = train_df_normalized.groupby('engine_id')['cycle'].transform(lambda x: x.max() - x)

# PWRUL
# Set the early RUL threshold
early_rul_threshold = 120

# Define the piecewise linear degradation function
def piecewise_rul(cycle, max_cycle):
    remaining_life = max_cycle - cycle
    if remaining_life > early_rul_threshold:
        return early_rul_threshold  # slower degradation in the early phase
    else:
        return remaining_life  # direct linear degradation after threshold
    
train_df_normalized["PWRUL"] = train_df_normalized.apply(lambda row: piecewise_rul(row['cycle'], row['cycle'] + row['RUL']), axis=1)


from tsfresh.utilities.dataframe_functions import roll_time_series
df_rolled = roll_time_series(train_df_normalized, column_id="engine_id", column_sort="cycle", max_timeshift=30, min_timeshift=5)

from tsfresh import extract_features
features = extract_features(df_rolled, column_id="engine_id", column_sort="cycle")
