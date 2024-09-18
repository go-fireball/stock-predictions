import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from stock_predictions.stock_lstm import StockLSTM

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MVA.10",
    "MVA.20",
    "MVA.30",
    "MVA.60",
    "MVA.90",
    "MVA.120",
    "MVA.180",
    "Std.2",
    "Std.5",
    "Std.30",
    "Return.daily",
    "Actual_Volatility.30",
    "RSI"
]


def load(ticker, start_date):
    data = pd.read_csv('../data/derived/' + ticker + '_1d.csv', index_col=0, parse_dates=True)
    data['Expected'] = data['Close'].rolling(window=120, min_periods=1).max().shift(-119)  # Look ahead 120 days
    data = data.dropna(subset=['Expected'])
    data = data[data.index >= pd.to_datetime(start_date)]
    return data


def preprocess_data(df, scalar):
    if scalar is None:
        min_max_scalar = RobustScaler()
    else:
        min_max_scalar = scalar

    data = min_max_scalar.fit_transform(df[features + ['Expected']])
    seq_length = 60
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, :-1])
        y.append(data[i, -1])

    X = np.array(X)
    y = np.array(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), min_max_scalar


# Inverse-transform y_test to the original scale based on the Expected column
def inverse_transform_y(y, min_max_scaler):
    # Create a dummy array with the same number of columns as the features plus the 'Expected' column
    dummy_data = np.zeros((len(y), len(features) + 1))  # Assuming 'Expected' is the last column

    # Place scaled y values into the 'Expected' column (the last column)
    dummy_data[:, -1] = y

    # Inverse transform the dummy data to get the original Expected (target) values
    y_original = min_max_scaler.inverse_transform(dummy_data)[:, -1]  # Extract the original 'Expected' values
    return y_original


def train_model_with_early_stopping(
        model, X_train, y_train, X_val,
        y_val, epochs=100, patience=10, batch_size=64,
        lr=1e-4):
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam([
    #     {'params': model.lstm.parameters(), 'lr': 1e-4},  # Use Adam for LSTM layers
    #     {'params': model.fc.parameters(), 'lr': 1e-3}    # Use a larger learning rate for the fully connected layers
    # ])

    optimizer = torch.optim.RMSprop([
        {'params': model.lstm.parameters(), 'lr': 1e-5, "weight_decay": 1e-4},
        {'params': model.linear.parameters(), 'lr': 1e-4, "weight_decay": 1e-3}
    ])

    best_loss = np.inf
    patience_counter = 0

    # Create DataLoader for mini-batch processing
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Loop over mini-batches in the training dataset
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.view(-1, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate batch losses

        avg_loss = epoch_loss / len(train_loader)  # Average loss per epoch

        # Validation phase (no mini-batches needed, process entire validation set)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.view(-1, 1))

        print(f'Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss.item():.4f}')

        # Early stopping condition
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0  # Reset the counter if validation loss improves
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return model


def train_model(model, X, y, epochs=100, batch_size=64, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            model.train()
            outputs = model(X_batch)

            # Ensure both output and y_batch have the same size (batch_size, 1)
            loss = criterion(outputs, y_batch.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model


def predict(model, X, min_max_scalar):
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()

    # Create a placeholder array with the same structure as the original data for inverse transform
    # Only the 4th column (index 3, the Close price) contains the prediction data, other columns are 0s
    dummy_data = np.zeros((predictions.shape[0], len(features) + 1))
    dummy_data[:, -1] = predictions[:, 0]

    actual_predictions = min_max_scalar.inverse_transform(dummy_data)[:, -1]  # Extract the actual Close prices

    return actual_predictions


def get_model():
    return StockLSTM(input_size, hidden_layer_size=100, output_size=1, num_layers=3)


if __name__ == "__main__":
    input_size = len(features)
    model = get_model()
    scalar = None
    initial_lr = 1e-5
    decay_factor = 0.5
    for ticker in ["MSFT", "AAPL", "AMZN", "KLAC", "GOOG", "SMH", "QQQ", "AMAT"]:
        print(f"Training using: {ticker}")
        df = load(ticker, '2022-01-01')
        X, y, scalar = preprocess_data(df, scalar)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        train_model_with_early_stopping(
            model, X_train=X_train, y_train=y_train,
            X_val=X_test, y_val=y_test,
            batch_size=8, epochs=1500, lr=initial_lr)
        # initial_lr *= decay_factor
        joblib.dump(scalar, '../models/{0}_scalar.pkl'.format(ticker))
        torch.save(model.state_dict(), '../models/{0}_model.pth'.format(ticker))

        predictions = predict(model, X_test, scalar)
        y_test_original = inverse_transform_y(y_test.numpy(), scalar)
        mse = mean_squared_error(y_test_original, predictions)
        rmse = np.sqrt(mse)
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Predicting using: {ticker}")

        X, y, scalar = preprocess_data(df, scalar)
        predictions = predict(model, X[-120:], scalar)
        print(predictions)
        print(max(predictions))

