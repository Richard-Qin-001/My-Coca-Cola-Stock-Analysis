import os

import config

try:
    os.remove(os.path.join(config.MODEL_ROOT, 'ko_lstm_checkpoint.pth'))

except:
    exit(0)
