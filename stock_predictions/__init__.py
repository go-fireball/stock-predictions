import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from stock_predictions.stock_lstm import StockLSTM


def load(ticker, start_date):
    data = pd.read_csv('../data/raw/' + ticker + '_1d.csv', index_col=0, parse_dates=True)
    data['MA30'] = data['Close'].rolling(window=30).mean()
    data['MA60'] = data['Close'].rolling(window=60).mean()
    data['MA90'] = data['Close'].rolling(window=90).mean()
    data['MA120'] = data['Close'].rolling(window=120).mean()
    data = data[data.index >= pd.to_datetime(start_date)]
    return data


def preprocess_data(df):
    features = ['Open', 'High', 'Low', 'Close', 'MA30', 'MA60', 'MA90', 'MA120']

    minMaxScaler = MinMaxScaler(feature_range=(0, 1))
    data = minMaxScaler.fit_transform(df[features])

    seq_length = 60
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i][3])

    X = np.array(X)
    y = np.array(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), minMaxScaler


def train_model(model, X, y, epochs=100, batch_size=64, lr=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model


def predict(model, X, minMaxScaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()

    # Create a placeholder array with the same structure as the original data for inverse transform
    # Only the 4th column (index 3, the Close price) contains the prediction data, other columns are 0s
    dummy_data = np.zeros((predictions.shape[0], 8))  # Assuming 8 features including Open, High, Low, etc.
    dummy_data[:, 3] = predictions[:, 0]  # Populate only the Close price (index 3)

    # Inverse transform the dummy data to get back to original scale
    actual_predictions = minMaxScaler.inverse_transform(dummy_data)[:, 3]  # Extract the actual Close prices

    return actual_predictions


if __name__ == "__main__":
    df = load("MSFT", '2010-01-01')
    X, y, scalar = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    input_size = X.shape[2]
    stockLSTM = StockLSTM(input_size, hidden_layer_size=50, output_size=1, num_layers=5)
    model = train_model(stockLSTM, X_train, y_train, epochs=100, lr=1e-4)
    torch.save(model.state_dict(), 'model.pth')
    joblib.dump(scalar, 'scalar.pkl')

    stockLSTM = StockLSTM(input_size, hidden_layer_size=50, output_size=1, num_layers=5)
    stockLSTM.load_state_dict(torch.load('model.pth', weights_only=False))
    stockLSTM.eval()
    prediction_scalar = joblib.load('scalar.pkl')
    predictions = predict(stockLSTM, X_test, prediction_scalar)

    mse = mean_squared_error(y_test.numpy(), predictions)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")

    # Predict future prices based on the last 30 sequences
    predictions = predict(stockLSTM, X[-30:], prediction_scalar)
    print(predictions)
    # Predict future prices based on the last 30 sequences
    predictions = predict(model, X[-30:], scalar)
    print(predictions)
