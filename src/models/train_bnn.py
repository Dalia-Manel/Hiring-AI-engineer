import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.models.BnnModel import BayesianModel
from src.data.data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)

MUANA_DATA_PATH = "data/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "data/international-airline-passengers.csv"

# ------------------------------------------
# mauna_loa_atmospheric_co2 Dataset
# ------------------------------------------

# Prepare data
X1, y1, X1_normalized = load_mauna_loa_atmospheric_co2(MUANA_DATA_PATH)

# Split the data into training and test sets
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1_normalized, y1, test_size=0.2, random_state=42
)

# Convert NumPy arrays to PyTorch tensors
X1_train_tensor = torch.from_numpy(X1_train).float()
y1_train_tensor = torch.from_numpy(y1_train).float()
X1_test_tensor = torch.from_numpy(X1_test).float()

# Define the Bayesian neural network model
input_size = X1_train.shape[1]
hidden_size1 = 20
hidden_size2 = 10
output_size = 1
model = BayesianModel(input_size, hidden_size1, hidden_size2, output_size)

# Define loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Normalize the data
scaler = StandardScaler()
X1_train_scaled = scaler.fit_transform(X1_train)
X1_test_scaled = scaler.transform(X1_test)

X1_train_tensor = torch.tensor(X1_train_scaled, dtype=torch.float32)
X1_test_tensor = torch.tensor(X1_test_scaled, dtype=torch.float32)

# Convert y1_test to a Torch tensor
y1_test_tensor = torch.tensor(y1_test, dtype=torch.float32)

# Evaluation function for regression
def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression metrics.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: A dictionary containing MSE, RMSE, and MAE.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

# Training loop
num_epochs = 1000
train_losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X1_train_tensor)
    loss = loss_function(outputs, y1_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

# Plot training losses
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

# Plot Ground Truth vs Predictions

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    predictions_1 = model(X1_test_tensor)

# Convert predictions to NumPy array for plotting and evaluation
predictions_np_1 = predictions_1.numpy()

# Evaluate regression metrics
regression_metrics = evaluate_regression(y1_test_tensor.numpy(), predictions_np_1)
print("Regression Metrics:", regression_metrics)

# Export model
torch.save(model, '{}/best_model_{}.pth.tar'.format(model_save, 1))

plt.figure(figsize=(10, 6))
plt.plot(X1_test, y1_test, "b.", markersize=10, label="Ground Truth")
plt.plot(X1_test, predictions_1, "r.", markersize=10, label="Predictions")
plt.xlabel("Input Features (X_test)")
plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
plt.title("True Values vs Predictions")
plt.legend()
plt.show()

# ------------------------------------------
# international-airline-passengers Dataset
# ------------------------------------------

# Prepare data
X2, y2, X2_normalized = load_international_airline_passengers(AIRLINE_DATA_PATH)

# Split the data into training and test sets
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_normalized, y2, test_size=0.2, random_state=42
)

# Convert NumPy arrays to PyTorch tensors
X2_train_tensor = torch.from_numpy(X2_train).float()
y2_train_tensor = torch.from_numpy(y2_train).float()
X2_test_tensor = torch.from_numpy(X2_test).float()

# Define the Bayesian neural network model
input_size = X2_train.shape[1]
hidden_size1 = 20
hidden_size2 = 10
output_size = 1
model = BayesianModel(input_size, hidden_size1, hidden_size2, output_size)

# Define loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Normalize the data
scaler = StandardScaler()
X2_train_scaled = scaler.fit_transform(X2_train)
X2_test_scaled = scaler.transform(X2_test)

X2_train_tensor = torch.tensor(X2_train_scaled, dtype=torch.float32)
X2_test_tensor = torch.tensor(X2_test_scaled, dtype=torch.float32)

# Convert y2_test to a Torch tensor
y2_test_tensor = torch.tensor(y2_test, dtype=torch.float32)

# Evaluation function for regression
def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression metrics.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: A dictionary containing MSE, RMSE, and MAE.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

# Training loop
num_epochs = 1000
train_losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X2_train_tensor)
    loss = loss_function(outputs, y2_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

# Plot training losses
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

# Plot Ground Truth vs Predictions

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    predictions_2 = model(X2_test_tensor)

# Convert predictions to NumPy array for plotting and evaluation
predictions_np_2 = predictions_2.numpy()

# Evaluate regression metrics
regression_metrics = evaluate_regression(y2_test_tensor.numpy(), predictions_np_2)
print("Regression Metrics:", regression_metrics)

# Export model
torch.save(model, '{}/best_model2_{}.pth.tar'.format(model_save, 2))

plt.figure(figsize=(10, 6))
plt.plot(X2_test, y2_test, "b.", markersize=10, label="Ground Truth")
plt.plot(X2_test, predictions_np_2, "r.", markersize=10, label="Predictions")
plt.xlabel("Input Features (X_test)")
plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
plt.title("True Values vs Predictions")
plt.legend()
plt.show()
