import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

os.environ['KAGGLE_CACHE_DIR'] = DATA_ROOT

print(DATA_ROOT)

import kagglehub

# Download latest version
path = kagglehub.dataset_download("isaaclopgu/coca-cola-stock-daily-updated")

print("Path to dataset files:", path)