import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import io
from sklearn.preprocessing import MinMaxScaler

# Load Data
df_raw = pd.read_csv(os.path.join(DATA_ROOT, 'Coca_Cola_historical_data.csv'))
print(df_raw.info())

df_raw['Date'] = pd.to_datetime(df_raw['Date'], utc=True).dt.tz_convert(None).dt.date
df_raw.set_index('Date', inplace=True)

df = df_raw[['Close', 'Volume']].copy()

start_date = '2015-01-01'
df_filtered = df[df.index >= pd.to_datetime(start_date).date()]

data_matrix = df_filtered.values

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(data_matrix)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Sequence Length
SEQUENCE_LENGTH = 60

def create_sequences(data, seq_length):
    """
    将时间序列数据转换为 LSTM 所需的序列和标签。
    
    Args:
        data (np.ndarray): 归一化后的数据 (N, Features_Count)。
        seq_length (int): 序列长度 L。
        
    Returns:
        tuple: (X_sequences, y_targets) 的 NumPy 数组。
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length, :]
        X.append(seq)
        label = data[i + seq_length, 0] 
        y.append(label)
    
    return np.array(X), np.array(y).reshape(-1, 1)

X_np, y_np = create_sequences(scaled_data, SEQUENCE_LENGTH)

TRAIN_RATIO = 0.8
train_size = int(len(X_np) * TRAIN_RATIO)

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

class StockDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, *args, **kwargs):
        super(StockLSTM, self).__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout)
        # FC Layer
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # Create zero Tensor
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])

        return out
    
def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    print(f"Start Training on {device}")
    print(f"Epochs: {num_epochs}")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            optimizer.zero_grad() 
            loss.backward()       # Calculate Grad
            optimizer.step()      # Update Weight

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

def evaluate_predict(model, test_loader, scaler, y_test_np, device):
    model.eval() # Evaluate Mode
    test_predictions = []

    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            
            outputs = model(batch_x)
            
            test_predictions.extend(outputs.cpu().numpy())

    dummy_test_matrix = np.zeros((len(test_predictions), scaled_data.shape[1]))
    dummy_test_matrix[:, 0] = np.array(test_predictions).flatten()

    real_predictions_matrix = scaler.inverse_transform(dummy_test_matrix)
    real_predictions = real_predictions_matrix[:, 0]

    dummy_true_matrix = np.zeros((len(y_test_np), scaled_data.shape[1]))
    dummy_true_matrix[:, 0] = y_test_np.flatten()
    real_targets_matrix = scaler.inverse_transform(dummy_true_matrix)
    real_targets = real_targets_matrix[:, 0]

    mse = mean_squared_error(real_targets, real_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_targets, real_predictions)

    print(f"\nModel Prediction: ")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(real_targets, label = "Actual Price", color = 'blue')
    plt.plot(real_predictions, label = "Predicted Price", color = 'red', linestyle = '--')
    plt.title("Coca Cola Stock Predict")
    plt.xlabel("Days")
    plt.ylabel("Price(USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return real_targets, real_predictions



train_dataset = StockDataset(X_train_tensor, y_train_tensor)
test_dataset = StockDataset(X_test_tensor, y_test_tensor)

if __name__ == "__main__":
    BATCH_SIZE = 32
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers = 4
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers = 4
        )
    print(f"\nBatch Size: {BATCH_SIZE}")
    print("DataLoader Creation Complete...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    INPUT_SIZE = scaled_data.shape[1]  # 2 (Close, Volume)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1
    DROPOUT = 0.2
    model = StockLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    # Mean Squared Error
    criterion = nn.MSELoss()
    LEARNING_RATE = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Input Size: {INPUT_SIZE}")
    print(f"Hidden Size: {HIDDEN_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")

    NUM_EPOCHS = 10
    train_model(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)
    print("Training Complete...")

    real_targets, real_predictions = evaluate_predict(model, test_loader, scaler, y_test_np, device)
