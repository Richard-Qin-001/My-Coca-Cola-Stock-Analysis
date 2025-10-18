import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

import config

class StockDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length, :]
        X.append(seq)
        label = data[i + seq_length, 0] 
        y.append(label)
    
    return np.array(X), np.array(y).reshape(-1, 1)

def load_and_preprocess_data(data_file, start_date):
    data_path = os.path.join(config.DATA_ROOT, data_file)
    try:
        df_raw = pd.read_csv(data_path)
    except FileExistsError:
        print(f"Error: Data File {data_file} not found.")
        return None, None, None, None, None, None
    

    df_raw['Date'] = pd.to_datetime(df_raw['Date'], utc=True).dt.tz_convert(None).dt.date
    df_raw.set_index('Date', inplace=True)

    df = df_raw[['Close', 'Volume']].copy()

    start_date = '2015-01-01'
    df_filtered = df[df.index >= pd.to_datetime(start_date).date()]

    data_matrix = df_filtered.values

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(data_matrix)

    X_np, y_np = create_sequences(scaled_data, config.SEQUENCE_LENGTH)

    train_size = int(len(X_np) * config.TRAIN_RATIO)

    # Use List Slice
    X_train_np = X_np[:train_size]
    y_train_np = y_np[:train_size]

    X_test_np = X_np[train_size:]
    y_test_np = y_np[train_size:]

    print(f"\nThe size of Training Set: {len(X_train_np)}")
    print(f"The size of Testing Set: {len(X_test_np)}")

    # Transform
    X_train_tensor = torch.from_numpy(X_train_np).float()
    y_train_tensor = torch.from_numpy(y_train_np).float()
    X_test_tensor = torch.from_numpy(X_test_np).float()
    y_test_tensor = torch.from_numpy(y_test_np).float()

    
    train_dataset = StockDataset(X_train_tensor, y_train_tensor)
    test_dataset = StockDataset(X_test_tensor, y_test_tensor)

    input_size = scaled_data.shape[1]
    
    return train_dataset, test_dataset, scaler, scaled_data, input_size, y_test_np