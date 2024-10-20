import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pathlib import Path

class NeuralNetModel(nn.Module):
    def __init__(self):
        super(NeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def load_and_aggregate_data(file_path: Path | str):
    daily_data = pd.read_csv(file_path, parse_dates=['# Date'])
    monthly_data = daily_data.resample('M', on='# Date').sum()
    
    X_train = np.array(monthly_data.index.month).reshape(-1, 1)
    y_train = monthly_data['Receipt_Count'].to_numpy().reshape(-1, 1)
    
    return X_train, y_train

def train_model(data_path: str, model_save_path: str):
    X_train, y_train = load_and_aggregate_data(data_path)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    model = NeuralNetModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5000
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), model_save_path)
    return model, scaler_X, scaler_y
