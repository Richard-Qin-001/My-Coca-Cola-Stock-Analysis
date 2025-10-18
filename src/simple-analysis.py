import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

df = pd.read_csv(os.path.join(DATA_ROOT, 'Coca_Cola_historical_data.csv'))

# Display the first few rows of the dataframe
print('Data loaded successfully:')
print(df.head())

# Check the DataFrame's info
print('\nDataFrame Info:')
print(df.info())

df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)

df.sort_values('Date', inplace=True)

df.dropna(inplace=True)

df.set_index('Date', inplace=True)

data = df[['Close', 'Volume']].copy()

print("\nProcessed Data Head:")
print(data.head())

data['MA_20'] = data['Close'].rolling(window=20).mean()

data.dropna(inplace=True)

print('-'*30)
print(f"\n{data.shape}")
print(data.tail())

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Sequencing & Splitting
LOOKBACK = 60
TARGET_COLUMN_INDEX = 0

def create_sequences(data_array, lookback):
    X = []
    Y = []
    for i in range(lookback, len(data_array)):
        X.append(data_array[i-lookback:i, :]) 
        Y.append(data_array[i, TARGET_COLUMN_INDEX])
        
    return np.array(X), np.array(Y)

X, Y = create_sequences(scaled_data, LOOKBACK)

print(f"\nSequenced:")
print(f"X (Input Sequence) Shape: {X.shape} -> (Samples, 60 days, Eigenvalues)")
print(f"Y (Output Target) Shape: {Y.shape} -> (Samples)")

TRAIN_RATIO = 0.8
train_size = int(len(X) * TRAIN_RATIO)

X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

print(f"Train Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")

# Model Building and Training

lookback = X_train.shape[1] 
feature_count = X_train.shape[2]

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(lookback, feature_count)),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(
    optimizer='adam', 
    loss='mean_squared_error'
)

print("\n Model Summary:")
model.summary()

EPOCHS = 25  # 25 times training
BATCH_SIZE = 32

print("\nStart Training...")
history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, Y_test), # Test Set
    shuffle=False # Time sequence needed
)

print("Training Complete!")

predicted_scaled = model.predict(X_test)

temp_array = np.zeros((len(predicted_scaled), feature_count))
temp_array[:, TARGET_COLUMN_INDEX] = predicted_scaled.flatten()

predicted_prices = scaler.inverse_transform(temp_array)[:, TARGET_COLUMN_INDEX]

temp_array_real = np.zeros((len(Y_test), feature_count))
temp_array_real[:, TARGET_COLUMN_INDEX] = Y_test.flatten()
real_prices = scaler.inverse_transform(temp_array_real)[:, TARGET_COLUMN_INDEX]

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))

print(f"\n Model Evaluation:")
print(f"Root Mean Square Error (RMSE) of the test set: ${rmse:.4f}")

test_dates = data.index[-len(real_prices):]

plt.figure(figsize=(14, 6))
plt.plot(test_dates, real_prices, color='blue', label='Actual closing price')
plt.plot(test_dates, predicted_prices, color='red', label='LSTM predict closing price')
plt.title('Coca Cola Stock Predict')
plt.xlabel('Date')
plt.ylabel('Price(Dollar)')
plt.legend()
plt.grid(True)
plt.show()