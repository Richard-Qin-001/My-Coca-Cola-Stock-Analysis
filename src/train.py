import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_val_loss += loss.item()
            
    model.train()
    return total_val_loss / len(val_loader)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, start_epoch=0, save_path=None, scheduler=None):
    model.train()
    print(f"Start Training on {device}, Epochs: {num_epochs}, Starting from Epoch {start_epoch + 1}")
    best_val_loss = float('inf')

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

        val_loss = validate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            print(f" -> Loss from {best_val_loss:.6f} to {val_loss:.6f}. Saving as Checkpoint.")
            best_val_loss = val_loss

            if save_path:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, save_path.replace('.pth', '_best.pth'))
                print(f" -> Checkpoint saved to {save_path}")
def evaluate_predict(model, test_loader, scaler, y_test_np, device, scaled_data):
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
