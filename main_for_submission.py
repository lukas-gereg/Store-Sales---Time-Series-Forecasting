import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error

from models.store_sales_gru_model import StoreSalesGRUModel
from Data.sales_dataset import SalesDataset

def main():
    # Load and preprocess training dataset
    train_data_path = './MyData/final_train_dataset.csv'
    df = pd.read_csv(train_data_path)

    # Preprocess the dataset
    df['date'] = pd.to_datetime(df['date'])
    df['days_from_start'] = (df['date'] - df['date'].min()).dt.days
    df = df.drop(columns=['date'])

    # Define sequence length and split into train/validation sets
    sequence_length = 30
    val_size = 0.1  # 10% of the data for validation
    val_split_size = int(len(df) * val_size)

    train_df = df.iloc[:-val_split_size]
    val_df = df.iloc[-val_split_size:]

    # Create datasets and data loaders
    train_dataset = SalesDataset(train_df, sequence_length=sequence_length)
    val_dataset = SalesDataset(val_df, sequence_length=sequence_length)

    batch_size = 5000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # Define network parameters
    model_properties = {
        'input_size': train_df.drop(columns=['sales']).shape[1],  # Number of input features
        'hidden_size': 64,
        'num_layers': 4,
        'output_size': 1,  # Predicting one "sales" value
        'dropout': 0.2
    }

    # Initialize the model
    model = StoreSalesGRUModel(model_properties)

    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Early stopping parameters
    num_epochs = 50
    patience = 15  # Stop training if validation loss doesn't improve for 15 consecutive epochs
    best_val_loss = float('inf')
    trigger_times = 0

    # Training loop with validation and early stopping
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    # Load test dataset (final_test_dataset.csv)
    test_features_path = './MyData/final_test_dataset.csv'
    test_ids_path = './MyData/test.csv'

    test_features = pd.read_csv(test_features_path)
    test_ids = pd.read_csv(test_ids_path)

    # Preprocess test dataset
    test_features['date'] = pd.to_datetime(test_features['date'])
    test_features['days_from_start'] = (test_features['date'] - test_features['date'].min()).dt.days
    test_features = test_features.drop(columns=['date'])

    test_dataset = SalesDataset(test_features, sequence_length=sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Predict sales for the test dataset
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            predictions.extend(outputs.cpu().numpy())

    # Ensure predictions length matches the expected submission format
    predictions = np.array(predictions, dtype=np.float64)

    # Adjust predictions to match the test IDs
    submission = pd.DataFrame({
        'id': test_ids['id'].iloc[sequence_length:].reset_index(drop=True),  # Align IDs
        'sales': predictions
    })

    # Fill missing rows with zeros for IDs without predictions
    missing_rows = len(test_ids) - len(submission)
    if missing_rows > 0:
        missing_data = pd.DataFrame({
            'id': test_ids['id'].iloc[:sequence_length],
            'sales': [0] * sequence_length
        })
        submission = pd.concat([missing_data, submission], ignore_index=True)

    # Save to submission.csv
    submission.to_csv('submission.csv', index=False)
    print(f"Submission file created with {len(submission)} rows: submission.csv")

if __name__ == '__main__':
    main()
