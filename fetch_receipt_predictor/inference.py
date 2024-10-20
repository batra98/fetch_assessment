import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_model(model_path: str, model_class, device='cpu'):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_receipts(model, scaler_X: MinMaxScaler, scaler_y: MinMaxScaler, months: list):
    X_test = np.array([[i] for i in months])
    X_test_scaled = scaler_X.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor)
    
    predictions = scaler_y.inverse_transform(predictions_scaled.numpy())
    return predictions
