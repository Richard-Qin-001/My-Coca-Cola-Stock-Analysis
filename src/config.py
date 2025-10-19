import os

DATA_FILE = 'Coca_Cola_historical_data.csv'
START_DATE = '2015-01-01'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
MODEL_ROOT = os.path.join(PROJECT_ROOT, 'model')
CHECKPOINT_PATH = os.path.join(MODEL_ROOT, 'ko_lstm_checkpoint.pth')

SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
BATCH_SIZE = 32
FINAL_FEATURES = ['Close', 'Volume', 'RSI', 'SMA_10', 'SMA_50', 'MACD', 'MACD_Signal']

HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 1
DROPOUT = 0.2

LEARNING_RATE = 0.001
NUM_EPOCHS = 30

# ReduceLROnPlateau
LR_PATIENCE = 5
LR_FACTOR = 0.5
LR_MIN = 1e-6