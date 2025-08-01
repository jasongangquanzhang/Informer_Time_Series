import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
# from models.model import (
#     Informer,
# )  # Assuming Informer model is already defined and available
# from models.RNN import CustomReLUNet, CustomReLURNN
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_process import ArmaProcess
import argparse
import csv
import sys
from pmdarima import auto_arima
from filelock import Timeout, FileLock
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import timesfm


# Set up random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def piecewise_function(x, theta_1, theta_2):
    if x > 0:
        return theta_1 * x
    else:
        return theta_2 * x


def generatedata_ld(T, func_type, covis=False):
    Z1 = np.random.uniform(-1, 1, T)
    Z2 = np.random.uniform(-1, 1, T)
    Z3 = np.random.uniform(-1, 1, T)
    Z4 = np.random.uniform(-1, 1, T)
    Z5 = np.random.uniform(-1, 1, T)
    Z6 = np.random.uniform(-1, 1, T)
    Z7 = np.random.uniform(-1, 1, T)
    if func_type == "abs":
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        theta_0 = 0.5
        theta_1 = 0.5
        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = theta_0 * abs(X[t - 1]) + U[t] + theta_1 * U[t - 1]
            X.append(X_t)
            EX_t = theta_0 * abs(X[t - 1]) + theta_1 * U[t - 1]
            EX.append(EX_t)

    elif func_type == "arma":
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        theta_0 = 0.5
        theta_1 = 0.5
        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = theta_0 * X[t - 1] + U[t] + theta_1 * U[t - 1]
            X.append(X_t)
            EX_t = theta_0 * X[t - 1] + theta_1 * U[t - 1]
            EX.append(EX_t)

    elif func_type == "ar":
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        theta_0 = 0.5
        theta_1 = 0.5
        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = theta_0 * X[t - 1] + U[t]
            X.append(X_t)
            EX_t = theta_0 * X[t - 1]
            EX.append(EX_t)

    elif func_type == "ma":
        U = np.random.uniform(
            -1, 1, T
        )  # Generate white noise from a normal distribution
        X = [U[0]]  # Initialize the series with the first noise term
        theta_0 = 0.5
        theta_1 = 0.5
        EX = [0]
        for t in range(1, T):
            X_t = U[t] + theta_1 * U[t - 1]
            X.append(X_t)
            EX_t = theta_1 * U[t - 1]  # E[X_t] given past noise
            EX.append(EX_t)

    elif func_type == "lin-non1":
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        theta_0 = 0.5
        theta_1 = 0.5
        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = theta_0 * X[t - 1] + U[t] + theta_1 * U[t - 1] ** 2
            X.append(X_t)
            EX_t = theta_0 * X[t - 1] + theta_1 * U[t - 1] ** 2
            EX.append(EX_t)

    elif func_type == "lin-non":
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        theta_0 = 0.1
        theta_1 = 0.1
        theta_2 = 0.8

        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = (
                theta_0 * X[t - 1]
                + U[t]
                + piecewise_function(U[t - 1], theta_1, theta_2)
            )
            X.append(X_t)
            EX_t = theta_0 * X[t - 1] + piecewise_function(U[t - 1], theta_1, theta_2)
            EX.append(EX_t)

    elif func_type == "non-non_X":
        U = np.random.randn(T)
        # U = np.random.uniform(-1, 1, T) # 生成长度为T的均匀分布随机数
        beta_0 = -0.2
        beta_1 = 0.2
        theta_0 = -0.2
        theta_1 = 0.2
        theta_2 = -0.2
        theta_3 = 0.8
        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = (
                piecewise_function(X[t - 1], theta_0, theta_1)
                + U[t]
                + piecewise_function(U[t - 1], theta_2, theta_3)
                + beta_0 * Z1[t]
                + beta_0 * Z2[t]
                + beta_0 * Z3[t]
                + beta_1 * Z4[t]
                + beta_1 * Z5[t]
                + beta_1 * Z6[t]
                + beta_1 * Z7[t]
            )
            X.append(X_t)
            EX_t = (
                piecewise_function(X[t - 1], theta_0, theta_1)
                + piecewise_function(U[t - 1], theta_2, theta_3)
                + beta_0 * Z1[t]
                + beta_0 * Z2[t]
                + beta_0 * Z3[t]
                + beta_1 * Z4[t]
                + beta_1 * Z5[t]
                + beta_1 * Z6[t]
                + beta_1 * Z7[t]
            )
            EX.append(EX_t)

    elif func_type == "exp":
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        theta_1 = 0.5
        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = np.exp(-5 * X[t - 1] ** 2) + U[t] + theta_1 * U[t - 1]
            X.append(X_t)
            EX_t = np.exp(-5 * X[t - 1] ** 2) + theta_1 * U[t - 1]
            EX.append(EX_t)

    elif func_type == "FAR":
        U = np.random.randn(T)
        # U = np.random.uniform(-1, 1, T) # 生成长度为T的均匀分布随机数
        a1 = a2 = 0
        b1 = 1
        b2 = 1
        phi1 = 1
        phi2 = 1
        theta_1 = 0.5
        theta_2 = 0.2
        X = [U[0]]  # 初始化序列
        X.append(U[1])
        EX = [0, 0]
        for t in range(2, T):
            X_t = (
                (a1 + b1 * np.exp(-phi1 * X[t - 1] ** 2)) * X[t - 1]
                + (a2 + b2 * np.exp(-phi2 * X[t - 1] ** 2)) * X[t - 2]
                + U[t]
                + theta_1 * U[t - 1]
                + theta_2 * U[t - 2]
            )
            X.append(X_t)
            EX_t = (
                (a1 + b1 * np.exp(-phi1 * X[t - 1] ** 2)) * X[t - 1]
                + (a2 + b2 * np.exp(-phi2 * X[t - 1] ** 2)) * X[t - 2]
                + theta_1 * U[t - 1]
                + theta_2 * U[t - 2]
            )
            EX.append(EX_t)

    elif func_type == "bilinear":
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        theta_0 = 0.5
        theta_1 = 0.5
        theta_2 = 0.5
        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = (
                theta_0 * X[t - 1]
                + U[t]
                + theta_1 * U[t - 1]
                + theta_2 * X[t - 1] * U[t - 1]
            )
            X.append(X_t)
            EX_t = (
                theta_0 * X[t - 1] + theta_1 * U[t - 1] + theta_2 * X[t - 1] * U[t - 1]
            )
            EX.append(EX_t)

    elif func_type == "STAR":
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        phi = 0.1
        theta_1 = 0.5
        theta_2 = 0.5
        gamma = 20
        c = 1

        X = [U[0]]  # 初始化序列
        EX = [0]
        # G = [0]
        for t in range(1, T):
            # 计算平滑转换函数值
            G_t = 1 / (1 + np.exp(-gamma * (X[t - 1] - c)))
            # G.append(G_t)
            X_t = phi * X[t - 1] + theta_1 * X[t - 1] * G_t + U[t] + theta_2 * U[t - 1]
            X.append(X_t)
            EX_t = phi * X[t - 1] + theta_1 * X[t - 1] * G_t + +theta_2 * U[t - 1]
            EX.append(EX_t)

    if covis:
        X = np.array(X).reshape(-1, 1)  # 将 X 转换为列向量
        Z1 = Z1.reshape(-1, 1)  # 将 Z1 转换为列向量
        Z2 = Z2.reshape(-1, 1)  # 将 Z2 转换为列向量
        Z3 = Z3.reshape(-1, 1)  # 将 Z3 转换为列向量
        Z4 = Z4.reshape(-1, 1)  # 将 Z4 转换为列向量
        Z5 = Z5.reshape(-1, 1)  # 将 Z5 转换为列向量
        Z6 = Z6.reshape(-1, 1)  # 将 Z6 转换为列向量
        Z7 = Z7.reshape(-1, 1)  # 将 Z7 转换为列向量
        X = np.concatenate([X, Z1, Z2, Z3, Z4, Z5, Z6, Z7], axis=1)

    return np.array(X), np.array(EX)


def generate_arma_time_series(ar_params, ma_params, n_samples):
    """
    Generate an ARMA time series.

    Parameters:
        ar_params (list): Coefficients of AR terms. Include 1 as the first coefficient.
        ma_params (list): Coefficients of MA terms. Include 1 as the first coefficient.
        n_samples (int): Number of time series samples to generate.
    Returns:
        np.ndarray: Generated ARMA time series.
    """

    # Define the ARMA process
    arma_process = ArmaProcess(ar_params, ma_params)
    time_series = arma_process.generate_sample(nsample=n_samples)
    return time_series


# AR
def rolling_auto_ar(
    data, pred_len, information_criterion="bic", seasonal=False, max_order=(10, 2, 10)
):
    """
    Perform rolling forecast using auto ARIMA to determine the best order.
    The model updates only with the latest observed value.

    Parameters:
        data (array-like): Univariate time series data.
        pred_len (int): Number of data points to predict iteratively.
        seasonal (bool): Whether to fit a seasonal ARIMA model.
        max_order (tuple): Maximum (p, d, q) values for order search.

    Returns:
        forecasts (list): List of predicted values.
        mse (float): Mean Squared Error of the predictions.
    """
    # Ensure data is a NumPy array
    data = np.asarray(data)
    train_len = len(data) - pred_len
    # Calculate training length
    train_split = int(train_len * 0.8)

    # Split data into training and test sets
    train = data[:train_split]
    valid = data[train_split:train_len]
    forecasts = []

    # Use auto_arima to determine the best ARIMA order
    try:
        arima_model = auto_arima(
            train,
            seasonal=seasonal,
            start_q=0,
            max_p=max_order[0],
            d=0,  # Set differencing order to 0
            max_q=max_order[2],  # Set max MA order to 0
            information_criterion=information_criterion,
            stepwise=True,
            suppress_warnings=True,
        )
        print(f"Selected Order: {arima_model.order}")
    except Exception as e:
        print(f"ARIMA model fitting failed: {e}")
        sys.exit(-1)
    for i in range(len(valid)):
        arima_model.update([valid[i]])  # Update model step-by-step

    # Perform rolling forecast
    for i in range(pred_len):
        # Forecast the next value
        forecast = arima_model.predict(n_periods=1)[0]
        forecasts.append(forecast)

        # Update the model with the latest observed value
        new_data = [data[train_len + i]]  # Only the current observed value
        arima_model.update(new_data)

    return forecasts, arima_model.order


###### ARMA Benchmark ######
def rolling_auto_arima(
    data, pred_len, information_criterion="bic", seasonal=False, max_order=(20, 2, 20)
):
    """
    Perform rolling forecast using auto ARIMA to determine the best order.
    The model updates only with the latest observed value.

    Parameters:
        data (array-like): Univariate time series data.
        pred_len (int): Number of data points to predict iteratively.
        seasonal (bool): Whether to fit a seasonal ARIMA model.
        max_order (tuple): Maximum (p, d, q) values for order search.

    Returns:
        forecasts (list): List of predicted values.
        mse (float): Mean Squared Error of the predictions.
    """
    # Ensure data is a NumPy array
    data = np.asarray(data)
    train_len = len(data) - pred_len
    # Calculate training length
    train_split = int(train_len * 0.8)

    # Split data into training and test sets
    train = data[:train_split]
    valid = data[train_split:train_len]
    forecasts = []

    # Use auto_arima to determine the best ARIMA order
    try:
        arima_model = auto_arima(
            train,
            seasonal=seasonal,
            start_q=0,
            max_p=max_order[0],
            d=0,  # Set differencing order to 0
            max_q=max_order[2],  # Set max MA order to 0
            information_criterion=information_criterion,
            stepwise=True,
            suppress_warnings=True,
        )
        train_preds = arima_model.predict_in_sample()
        valid_preds = arima_model.predict(n_periods=len(valid))

        # Compute RMSE and MAE
        # Compute training and validation MSE
        train_mse = mean_squared_error(train, train_preds)
        valid_mse = mean_squared_error(valid, valid_preds)
        print(f"Selected Order: {arima_model.order}")
    except Exception as e:
        print(f"ARIMA model fitting failed: {e}")
        sys.exit(-1)
    for i in range(len(valid)):
        arima_model.update([valid[i]])  # Update model step-by-step

    # Perform rolling forecast
    for i in range(pred_len):
        # Forecast the next value
        forecast = arima_model.predict(n_periods=1)[0]
        forecasts.append(forecast)

        # Update the model with the latest observed value
        new_data = [data[train_len + i]]  # Only the current observed value
        arima_model.update(new_data)

    return forecasts, arima_model.order, train_mse, valid_mse

######## TimesFM ########
def TimesFM_forecast(data):
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=500,
            num_layers=50,
            use_positional_embedding=True,
            context_len=2048,
            point_forecast_mode='median'
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            path="pretrained_models/torch_model.ckpt",
            # huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
    )
    print("Using TimesFM for forecasting...")
    
    start = data_length - target_len
    forecast, _ = tfm.forecast(
        inputs=[data[: -target_len]],
        freq=[0],                
        window_size=None,            
        forecast_context_len=100,
        return_forecast_on_context=False,
        normalize=True,
    )
    print(f"Forecasting at time: {forecast}")

    return np.array(forecast[0]).tolist()  # final shape (500,)


###### Informer Module ######
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path="checkpoint.pth"):
        """
        Early stopping to stop the training when the validation loss doesn't improve after a given patience.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save model when validation loss decreases.
        """
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# Define custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len, target_len):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        enc_input = self.data[idx : idx + self.seq_len]
        dec_input = self.data[
            idx + self.seq_len - self.label_len : idx + self.seq_len + self.pred_len
        ]
        target = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return (
            torch.tensor(enc_input, dtype=torch.float32),
            torch.tensor(dec_input, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


# def iterative_prediction_with_update(
#     model, test_data, seq_len, label_len, pred_len, target_len, device
# ):
#     """
#     Perform iterative prediction and update the model with each new prediction and true value.
#     """
#     test_dataset = TimeSeriesDataset(
#         test_data, seq_len, label_len, pred_len, target_len=1
#     )
#     test_loader = DataLoader(test_dataset, batch_size=target_len, shuffle=False)
#     predictions = []
#     # Make a prediction
#     model.eval()
#     with torch.no_grad():
#         for enc_input, dec_input, target in test_loader:
#             enc_input, dec_input, target = (
#                 enc_input.unsqueeze(-1).to(device),
#                 dec_input.unsqueeze(-1).to(device),
#                 target.unsqueeze(-1).to(device),
#             )

#             y_pred = None
#             enc_in = enc_input
#             dec_in = dec_input

#             # Set the last point of decoder input to 0 (masking the last point as before)
#             dec_in[:, -1, :] = 0  # Mask the last bit of decoder input

#             # Make the prediction using the model
#             pred = model(enc_in, enc_in, dec_in, dec_in)

#     return pred.squeeze(-1).cpu().numpy()[:, 0].tolist()


# # Train the Informer model
# def train(
#     model,
#     train_loader,
#     val_loader,
#     criterion,
#     optimizer,
#     epochs,
#     device,
#     checkpoint_path="checkpoint.pth",

# ):
#     """
#     Train the model and return the final validation loss.
#     """
#     early_stopping = EarlyStopping(patience=8, verbose=True, path=checkpoint_path)
#     best_val_loss = float("inf")  # Track the best validation loss
#     train_lst = []
#     val_lst = []
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0
#         # Training loop
#         for enc_input, dec_input, target in train_loader:
#             enc_input, dec_input, target = (
#                 enc_input.unsqueeze(-1).to(device),
#                 dec_input.unsqueeze(-1).to(device),
#                 target.unsqueeze(-1).to(device),
#             )
#             optimizer.zero_grad()

#             enc_in = enc_input
#             dec_in = dec_input

#             # Set the last point of decoder input to 0 (masking the last point as before)
#             dec_in[:, -1, :] = 0  # Mask the last bit of decoder input

#             # Make the prediction using the model
#             y_pred = model(enc_in, enc_in, dec_in, dec_in)


#             loss = criterion(y_pred, target)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         train_loss /= len(train_loader)

#         # Validation loop
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for enc_input, dec_input, target in val_loader:
#                 enc_input, dec_input, target = (
#                     enc_input.unsqueeze(-1).to(device),
#                     dec_input.unsqueeze(-1).to(device),
#                     target.unsqueeze(-1).to(device),
#                 )

#                 enc_in = enc_input
#                 dec_in = dec_input


#                 # Set the last point of decoder input to 0 (masking the last point as before)
#                 dec_in[:, -1, :] = 0  # Mask the last bit of decoder input

#                 # Make the prediction using the model
#                 y_pred = model(enc_in, enc_in, dec_in, dec_in)

#                 loss = criterion(y_pred, target)
#                 val_loss += loss.item()
#             val_loss /= len(val_loader)

#         train_lst.append(train_loss)
#         val_lst.append(val_loss)
#         print(
#             f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
#         )

#         # Save the best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_state = deepcopy(model.state_dict())

#         early_stopping(val_loss, model=model)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break

#     # Load the best model
#     if best_model_state is not None:
#         model.load_state_dict(best_model_state)
#     return best_val_loss, train_lst, val_lst


# def informer_predict(informer_len_combinations, data):
#     """
#     Perform grid search over seq_len and label_len combinations to choose the best one based on validation loss.
#     """

#     best_val_loss = float("inf")
#     best_combination = None
#     best_model = None

#     # Iterate over all seq_len and label_len combinations
#     for seq_len, label_len in informer_len_combinations:
#         train_len = len(data) - seq_len - target_len
#         train_split = int(train_len * 0.8)
#         train_data = data[:train_split]
#         val_data = data[train_split:train_len]
#         # Prepare datasets and loaders
#         train_dataset = TimeSeriesDataset(
#             train_data, seq_len, label_len, pred_len, target_len=1
#         )
#         val_dataset = TimeSeriesDataset(
#             val_data, seq_len, label_len, pred_len, target_len=1
#         )
#         train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
#         val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#         for lr in lr_lst:

#             # Model setup
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             print(f"Using device: {device}")
#             model = Informer(
#                 enc_in=1,
#                 dec_in=1,
#                 c_out=1,
#                 seq_len=seq_len,
#                 label_len=label_len,
#                 out_len=pred_len,
#                 factor=5,
#                 d_model=d_model,
#                 n_heads=8,
#                 e_layers=2,
#                 d_layers=1,
#                 d_ff=d_ff,
#                 dropout=dropout,
#                 attn="prob",
#                 embed="fixed",
#                 freq="h",
#                 activation="gelu",
#                 output_attention=False,
#                 distil=True,
#                 mix=True,
#                 device=device,
#             ).to(device)

#             # Training setup
#             criterion = torch.nn.MSELoss()
#             optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
#             # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#             val_loss, train_lst, val_lst = train(
#                 model,
#                 train_loader,
#                 val_loader,
#                 criterion,
#                 optimizer,
#                 epochs=500,
#                 device=device,
#                 checkpoint_path=checkpoint_path,
#             )

#             # Update the best combination
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_combination = (seq_len, label_len)
#                 best_model = model  # Save model parameters
#                 best_lr = lr
#                 best_train_lst = train_lst
#                 best_val_lst = val_lst

#     print(
#         f"Best Combination: seq_len: {best_combination[0]}, label_len: {best_combination[1]}, Val Loss: {best_val_loss:.4f}"
#     )

#     # Find minimum losses
#     min_val_loss = min(best_val_lst)
#     min_train_loss = min(best_train_lst)

#     # Find epochs where minimum losses occur
#     min_val_epoch = best_val_lst.index(min_val_loss) + 1
#     min_train_epoch = best_train_lst.index(min_train_loss) + 1

#     # Plot training and validation loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(
#         range(1, len(best_val_lst) + 1),
#         best_val_lst,
#         marker="o",
#         label="Validation Loss",
#         color="r",
#     )
#     plt.plot(
#         range(1, len(best_train_lst) + 1),
#         best_train_lst,
#         marker="s",
#         label="Training Loss",
#         color="b",
#     )

#     # Mark minimum points
#     plt.scatter(
#         min_val_epoch,
#         min_val_loss,
#         color="red",
#         s=100,
#         zorder=3,
#         label=f"Min Val Loss: {min_val_loss:.4f}",
#     )
#     plt.scatter(
#         min_train_epoch,
#         min_train_loss,
#         color="blue",
#         s=100,
#         zorder=3,
#         label=f"Min Train Loss: {min_train_loss:.4f}",
#     )

#     # Add text annotations to show the exact loss values
#     plt.text(
#         min_val_epoch,
#         min_val_loss,
#         f"{min_val_loss:.4f}",
#         fontsize=12,
#         verticalalignment="bottom",
#         horizontalalignment="right",
#         color="red",
#     )
#     plt.text(
#         min_train_epoch,
#         min_train_loss,
#         f"{min_train_loss:.4f}",
#         fontsize=12,
#         verticalalignment="top",
#         horizontalalignment="right",
#         color="blue",
#     )

#     # Labels and title
#     plt.title("Training and Validation Loss Over Epochs")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.grid()

#     # Save the plot
#     plt.savefig(f"{plot_dir}/validation_loss_plot_{seed}.png")
#     plt.show()

#     # Perform iterative prediction using the best model
#     informer_predictions = iterative_prediction_with_update(
#         best_model,
#         data[len(data) - best_combination[0] - target_len :],
#         best_combination[0],
#         best_combination[1],
#         pred_len,
#         target_len,
#         device,
#     )

#     return informer_predictions, best_combination, best_lr

# Main function
def main():
    print("Setting seed for reproducibility...")
    set_seed(seed)
    result = {}
    result["seed"] = seed
    print("Generating synthetic data...")
    # Generate synthetic ARMA time series data
    data, EX = generatedata_ld(data_length, func_type=func_type)
    # data = generate_arma_time_series(ar, ma, data_length)
    # std = data.std()
    test_value = data[-target_len:].tolist()
    true_value = EX[-target_len:].tolist()
    # result['STD'] = std
    result["Test"] = test_value
    result["True"] = true_value
    print("Generated data length:", len(data))
    ###### ARMA Module ######
    (
        result["ARMA"],
        result["ARMA_Order"],
        result["ARMA_Train_loss"],
        result["ARMA_Valid_loss"],
    ) = rolling_auto_arima(data=data, pred_len=target_len)

    result["AR"], result["AR_Order"] = rolling_auto_ar(
        data=data, pred_len=target_len, max_order=(20, 2, 0)
    )

    # informer_pred, informer_para, informer_lr = informer_predict(
    #     informer_len_combinations=informer_len, data=data
    # )
    # result["Informer"] = informer_pred
    # result["Informer_para"] = informer_para
    # result["Informer_lr"] = informer_lr
    
    ####### TimesFM ######
    result["TimesFM"] = TimesFM_forecast(data)
    return result


if __name__ == "__main__":
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    parser = argparse.ArgumentParser(
        description="Train and evaluate Informer on synthetic time series with ARMA benchmark."
    )  #
    parser.add_argument(
        "integer", metavar="N", type=int, help="an integer for the accumulator"
    )
    arg = parser.parse_args()
    seed = int(arg.integer)
    func_type = "arma"
    # Generate data
    data_length = 1500
    target_len = 500
    # Parameters for ARMA(2,1) process
    # ar = [1, -0.5, 0.25]  # AR coefficients
    # ma = [1, 0.4]  # MA coefficients
    # informer setting
    pred_len = 1
    d_model = 64  # 512
    d_ff = 512  # 2048
    dropout = 0.2
    
    
    #############Pretrained Model Settings#############
    
    # mercury
    # informer_len = [(10, 5), (20, 10), (50, 20)]  
    # midway
    informer_len = [(10, 2), (20, 4), (50, 10)]
    lr_lst = [1e-4, 1e-3, 1e-2]  
    
    num = 10
    plot_dir = f"pretrained_val_plots_{num}"
    os.makedirs(plot_dir, exist_ok=True)

    output_file = f"pretrained_csv_results/result_{num}.csv"

    checkpoint_dir = "checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_seed_{seed}.pth")
    result = main()

    with FileLock(output_file + ".lock"):
        if os.path.exists(output_file):
            # Append results to the existing file
            with open(output_file, "a", newline="") as csvfile:
                writer = None
                if writer is None:  # Initialize writer with fieldnames if needed
                    writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                writer.writerow(result)
        else:
            # Create a new file and write the header with results
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", newline="") as csvfile:
                writer = None
                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                    writer.writeheader()  # Write header only once for a new file
                writer.writerow(result)
