import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

import config
from data_prep import load_and_preprocess_data
from model import StockLSTM
from train import train_model, evaluate_predict

def main():
    print("Data Preparation...")
    train_dataset, test_dataset, scaler, scaled_data, input_size, y_test_np = load_and_preprocess_data(
        config.DATA_FILE, config.START_DATE
    )
    
    if train_dataset is None:
        return
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers = 4 # 保持 num_workers = 4 的设置
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers = 4
    )
    print(f"Batch Size: {config.BATCH_SIZE}")
    print("DataLoader Creation Complete.")

    print("\nModel Initialization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = StockLSTM(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Input Size: {input_size}")
    print(f"Hidden Size: {config.HIDDEN_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")

    start_epoch = 0
    checkpoint_path = config.CHECKPOINT_PATH

    try:
        
        if os.path.exists(checkpoint_path):
            print(f"\nFind Checkpoint. Loading from Checkpoint {checkpoint_path} ")
    
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint['epoch'] 
            
            print(f"Checkpoint Loaded Successful.Starting Training from Epoch {start_epoch + 1} ")
        else:
            print("\nCheckpoint not found, Starting Training from Epoch 1")
            
    except Exception as e:
        print(f"Load Checkpoint error ({e}), Starting Training from Epoch 1")
        start_epoch = 0

    print("\nTraining and Evaluation...")
    train_model(model, train_loader, criterion, optimizer, device, config.NUM_EPOCHS, start_epoch, checkpoint_path)
    print("Training Complete.")

    evaluate_predict(model, test_loader, scaler, y_test_np, device, scaled_data)

if __name__ == "__main__":
    main()