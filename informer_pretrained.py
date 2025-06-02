import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from transformers import (
    InformerConfig,
    InformerForPrediction,
    Trainer,
    TrainingArguments,
)  # Assuming Informer model is already defined and available
from models.RNN import CustomReLUNet, CustomReLURNN
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_process import ArmaProcess
import argparse
import csv
import sys
from pmdarima import auto_arima
from filelock import Timeout, FileLock
from copy import deepcopy
import matplotlib.pyplot as plt


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


class UnivariateTimeSeriesDataset(Dataset):
    def __init__(self, series, context_length=20):
        self.series = series
        self.context_length = context_length
        self.prediction_length = 1  # fixed to 1

    def __len__(self):
        return len(self.series) - self.context_length - self.prediction_length

    def __getitem__(self, idx):
        past_values = self.series[idx : idx + self.context_length]
        future_values = self.series[
            idx + self.context_length : idx + self.context_length + 1
        ]  # 1 target

        return {
            "past_values": torch.tensor(past_values, dtype=torch.float).unsqueeze(-1),
            "past_observed_mask": torch.ones(self.context_length, 1),
            "future_values": torch.tensor(future_values, dtype=torch.float).unsqueeze(
                -1
            ),
            "past_time_features": torch.zeros(self.context_length, 4),
            "future_time_features": torch.zeros(1, 4),
            "static_categorical_features": torch.zeros(1, dtype=torch.long),
            "static_real_features": torch.zeros(1),
        }


def collate_fn(batch):
    return {key: torch.stack([x[key] for x in batch]) for key in batch[0]}


def load_model():
    config = InformerConfig.from_pretrained(
        "huggingface/informer-tourism-monthly"  # override to match your target shape
    )
    config.prediction_length = 1
    model = InformerForPrediction.from_pretrained(
        "huggingface/informer-tourism-monthly",
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model


def iterative_prediction_with_update(
    model, test_data, context_length, prediction_length, device
):
    """
    Perform iterative prediction using the pretrained Hugging Face Informer model.
    Output shape is (B, 1, 1).
    """
    model.eval()
    predictions = []

    # Iterate one step at a time for each future prediction
    for i in range(len(test_data) - context_length - prediction_length + 1):
        context_window = test_data[i : i + context_length]

        # Prepare input tensors
        inputs = {
            "past_values": torch.tensor(context_window, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(device),  # (1, context_length, 1)
            "past_observed_mask": torch.ones(1, context_length, 1).to(device),
            "past_time_features": torch.zeros(1, context_length, 4).to(device),
            "future_time_features": torch.zeros(1, prediction_length, 4).to(device),
            "static_categorical_features": torch.zeros(1, 1, dtype=torch.long).to(
                device
            ),
            "static_real_features": torch.zeros(1, 1).to(device),
        }

        with torch.no_grad():
            outputs = model(**inputs)

        # Get prediction: (1, prediction_length, 1), take first step only
        pred = outputs.predictions[:, 0, :].unsqueeze(1)  # shape (1, 1, 1)
        predictions.append(pred.cpu())

    return torch.cat(predictions, dim=0).numpy().tolist()  # Final shape: (N, 1, 1)


def fine_tune(model, dataset):
    args = TrainingArguments(
        output_dir="./informer-finetuned",
        per_device_train_batch_size=16,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=None,
    )

    trainer.train()
    trainer.save_model("./informer-finetuned")


# --------------------------
# STEP 5: Prediction
# --------------------------


def predict(model, past_values, context_length=168, prediction_length=24):
    model.eval()
    with torch.no_grad():
        inputs = {
            "past_values": torch.tensor(
                past_values[-context_length:], dtype=torch.float
            )
            .unsqueeze(0)
            .unsqueeze(-1),
            "past_observed_mask": torch.ones(1, context_length, 1),
            "past_time_features": torch.zeros(1, context_length, 4),
            "future_time_features": torch.zeros(1, prediction_length, 4),
            "static_categorical_features": torch.zeros(1, 1, dtype=torch.long),
            "static_real_features": torch.zeros(1, 1),
        }
        outputs = model(**inputs)
        return outputs.predictions.squeeze().numpy()


def informer_predict(informer_len_combinations, data):
    """
    Perform grid search over seq_len and label_len combinations to choose the best one based on validation loss.
    Using a pretrained Informer model from Hugging Face.
    """
    best_val_loss = float("inf")
    best_combination = None
    best_model = None

    for seq_len, label_len in informer_len_combinations:
        train_len = len(data) - seq_len - target_len
        train_split = int(train_len * 0.8)
        train_data = data[:train_split]
        val_data = data[train_split:train_len]

        # Dataset using your defined UnivariateTimeSeriesDataset
        train_dataset = UnivariateTimeSeriesDataset(train_data, context_length=seq_len)
        val_dataset = UnivariateTimeSeriesDataset(val_data, context_length=seq_len)

        for lr in lr_lst:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model().to(device)

            # Fine-tune the model
            fine_tune(model, train_dataset)

            # Validation
            model.eval()
            val_loader = DataLoader(
                val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
            )
            criterion = torch.nn.MSELoss()
            val_loss_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = criterion(outputs.predictions, batch["future_values"])
                    val_loss_total += loss.item()
            val_loss = val_loss_total / len(val_loader)

            # Track the best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_combination = (seq_len, label_len)
                best_model = model
                best_lr = lr

    print(
        f"Best Combination: seq_len: {best_combination[0]}, label_len: {best_combination[1]}, Val Loss: {best_val_loss:.4f}"
    )

    # Final prediction using best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    informer_predictions = iterative_prediction_with_update(
        best_model,
        data[len(data) - best_combination[0] - target_len :],
        seq_len=best_combination[0],
        label_len=best_combination[1],
        pred_len=1,
        target_len=1,
        device=device,
    )

    return informer_predictions, best_combination, best_lr


################################### RNN ##################################
# 定义滞后函数，生成滞后一阶和二阶的数据
def create_lagged_features(series, lag):
    X, y = [], []
    lags = list(range(1, lag + 1))
    for i in range(max(lags), len(series)):
        features = [series[i - lag] for lag in lags]  # 获取滞后的元素
        X.append(features)
        y.append(series[i])  # 当前时间步的元素作为目标值
    return np.array(X), np.array(y)


class RNNDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length

    def __len__(self):
        return self.X.shape[0] - self.seq_length

    def __getitem__(self, idx):
        X1 = torch.tensor(self.X[idx : idx + self.seq_length + 1].squeeze(-1))
        y1 = torch.tensor(self.y[idx : idx + self.seq_length + 1])
        return X1, y1


def train_and_evaluate(
    T1,
    T2,
    input_size,
    output_size,
    relu_layer_sizes_before,
    hidden_layers,
    relu_layer_sizes_after,
    lr,
    train_loader,
    val_loader,
    test_loader,
    model_type="RNN",
):
    """
    Train and evaluate either an RNN or DNN model based on the model_type input.

    Parameters:
    - model_type: "RNN" for CustomReLURNN, "DNN" for CustomReLUNet
    """

    # Select the model based on model_type
    if model_type == "RNN":
        model = CustomReLURNN(
            input_size,
            hidden_layers,
            output_size,
            relu_layer_sizes_before,
            relu_layer_sizes_after,
        )
    elif model_type == "DNN":
        model = CustomReLUNet(
            input_size,
            hidden_layers,
            output_size,
            relu_layer_sizes_before,
            relu_layer_sizes_after,
        )
    else:
        raise ValueError("Invalid model_type. Choose 'RNN' or 'DNN'.")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    early_stopping = EarlyStopping(patience=100, delta=0.0001)

    # Training and validation loop
    num_epochs = 2000
    train_losses = []
    val_losses = []

    # Track the best model
    best_val_loss = float("inf")
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            # print(X_batch.shape)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                val_outputs = model(X_val)
                val_outputs = val_outputs.reshape(-1)
                # print(val_outputs.shape)
                y1_val = y_val.reshape(-1)
                loss = criterion(val_outputs[T1:], y1_val[T1:])
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())

        early_stopping(val_loss, model=model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            test_outputs = model(X_test)
            test_outputs = test_outputs.reshape(-1)
            y1_test = y_test.reshape(-1)
            loss = criterion(test_outputs[(T1 + T2) :], y1_test[(T1 + T2) :])
            test_loss += loss.item()

    test_loss /= len(test_loader)
    if model_type == "RNN":
        W = model.get_W()
    else:
        W = None
    pred = np.array(test_outputs[(T1 + T2) :])
    print(f"Test Loss: {test_loss:.4f}")

    return best_val_loss, train_losses, val_losses, pred, test_loss, W


def find_best_model(
    T1,
    T2,
    input_size,
    output_size,
    lr_list,
    layer_list,
    train_loader,
    val_loader,
    test_loader,
    model_type="RNN",
):
    """
    Find the best model configuration by looping over different layer configurations and learning rates.
    Select either an RNN or DNN model based on the model_type parameter.

    Parameters:
    - T: Sequence length
    - input_size: Size of the input features
    - output_size: Size of the output
    - lr_list: List of learning rates to try
    - layer_list: List of configurations of relu_layer_sizes_before, hidden_layers, and relu_layer_sizes_after
    - train_loader, val_loader, test_loader: Data loaders for training, validation, and test sets
    - EX_T: Expected values (for comparison in test set)
    - model_type: Either "RNN" or "DNN" to choose the model architecture

    Returns:
    - best_config: Best layer configuration
    - best_lr: Best learning rate
    - best_model_results: Results (losses and predictions) of the best model
    - best_W: Weights of the best model
    """

    best_config = None
    best_lr = None
    best_val_loss = float("inf")
    best_model_results = None
    best_W = None

    # Loop over all layer configurations
    for relu_layer_sizes_before, hidden_layers, relu_layer_sizes_after in layer_list:
        print(
            f"Training with layer config: {relu_layer_sizes_before}, {hidden_layers}, {relu_layer_sizes_after}"
        )

        # Initialize variables to track the best learning rate for this specific layer configuration
        best_lr_for_config = None
        best_val_loss_for_config = float("inf")
        best_model_results_for_config = None
        best_W_for_config = None

        # Loop over learning rates for the current layer configuration
        for lr in lr_list:
            print(f"Training with learning rate: {lr}")

            # Call train_and_evaluate function to train the model and evaluate
            val_loss, train_losses, val_losses, pred, test_loss, W = train_and_evaluate(
                T1,
                T2,
                input_size,
                output_size,
                relu_layer_sizes_before,
                hidden_layers,
                relu_layer_sizes_after,
                lr,
                train_loader,
                val_loader,
                test_loader,
                model_type,
            )

            print(
                f"Learning Rate: {lr}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}"
            )

            # Update the best learning rate for this configuration if validation loss improves
            if val_loss < best_val_loss_for_config:
                best_val_loss_for_config = val_loss
                best_lr_for_config = lr
                best_model_results_for_config = (
                    val_loss,
                    train_losses,
                    val_losses,
                    pred,
                    test_loss,
                )
                best_W_for_config = W

        # After going through all learning rates for the current layer configuration,
        # check if this configuration (with its best learning rate) is the best overall
        if best_val_loss_for_config < best_val_loss:
            best_val_loss = best_val_loss_for_config
            best_config = (
                relu_layer_sizes_before,
                hidden_layers,
                relu_layer_sizes_after,
            )
            best_lr = best_lr_for_config
            best_model_results = best_model_results_for_config
            best_W = best_W_for_config

    # Return the best model configuration, learning rate, model results, and weights
    return best_config, best_lr, best_model_results, best_W


def rnn_forecast(train_data, val_data, test_data):
    # RNN
    input_size_list = [1]
    output_size = 1
    lr_list = [0.01, 0.001]
    layer_list = [
        ([4], 4, [4]),
        ([8], 4, [8]),
        ([8], 8, [8]),
        ([4, 8], 8, [8, 4]),
        ([8, 16], 8, [+16, 8]),
    ]

    # lr_list = [0.01]
    # layer_list = [([4], 4, [4])]
    Xtrain = np.vectorize(float)(train_data)
    Xval = np.vectorize(float)(val_data)
    Xtest = np.vectorize(float)(test_data)
    series = np.concatenate((Xtrain, Xval, Xtest))
    series = series[..., np.newaxis].astype(np.float32)

    # ensembling
    DNN_ensemble = []
    RNN_ensemble = []
    ensemble = 1
    for rep in range(ensemble):  #########################
        best_val_loss_RNN = float("inf")
        best_val_loss_DNN = float("inf")
        for input_size in input_size_list:
            # print(input_size)
            X, y = create_lagged_features(series, lag=input_size)

            T1 = len(Xtrain) - input_size
            T2 = len(Xval)
            train_X, train_y = X[:T1], y[:T1]
            val_X, val_y = X[: (T1 + T2)], y[: (T1 + T2)]
            test_X, test_y = X, y

            train_dataset = RNNDataset(train_X, train_y, train_X.shape[0] - 1)
            val_dataset = RNNDataset(val_X, val_y, val_X.shape[0] - 1)
            test_dataset = RNNDataset(test_X, test_y, test_X.shape[0] - 1)

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            # # DNN
            # best_config, best_lr, best_model_results, best_W = find_best_model(
            #     T1,
            #     T2,
            #     input_size,
            #     output_size,
            #     lr_list,
            #     layer_list,
            #     train_loader,
            #     val_loader,
            #     test_loader,
            #     "DNN",
            # )

            # best_val_loss_for_config_DNN = best_model_results[0]
            # if best_val_loss_for_config_DNN < best_val_loss_DNN:
            #     best_val_loss_DNN = best_val_loss_for_config_DNN
            #     best_input_size_DNN = input_size
            #     DNN_pred = best_model_results[3]

            # RNN
            best_config, best_lr, best_model_results, best_W = find_best_model(
                T1,
                T2,
                input_size,
                output_size,
                lr_list,
                layer_list,
                train_loader,
                val_loader,
                test_loader,
                "RNN",
            )

            best_val_loss_for_config_RNN = best_model_results[0]
            if best_val_loss_for_config_RNN < best_val_loss_RNN:
                best_val_loss_RNN = best_val_loss_for_config_RNN
                best_input_size_RNN = input_size
                RNN_pred = best_model_results[3]

        # DNN_ensemble.append(DNN_pred)
        RNN_ensemble.append(RNN_pred)

    RNN_ensemble1 = np.zeros((ensemble, len(test_X) - (T1 + T2)))
    # DNN_ensemble1 = np.zeros((ensemble, len(test_X) - (T1 + T2)))

    for i in range(ensemble):
        # DNN_ensemble1[i, :] = DNN_ensemble[i]
        RNN_ensemble1[i, :] = RNN_ensemble[i]

    # DNN_pred = np.mean(DNN_ensemble1, axis=0)
    RNN_pred = np.mean(RNN_ensemble1, axis=0)
    return RNN_pred


# Main function
def main():
    set_seed(seed)
    result = {}
    result["seed"] = seed

    # Generate synthetic ARMA time series data
    data, EX = generatedata_ld(data_length, func_type=func_type)
    # data = generate_arma_time_series(ar, ma, data_length)
    # std = data.std()
    test_value = data[-target_len:].tolist()
    true_value = EX[-target_len:].tolist()
    # result['STD'] = std
    result["Test"] = test_value
    result["True"] = true_value

    ###### ARMA Module ######
    # (
    #     result["ARMA"],
    #     result["ARMA_Order"],
    #     result["ARMA_Train_loss"],
    #     result["ARMA_Valid_loss"],
    # ) = rolling_auto_arima(data=data, pred_len=target_len)

    # result["AR"], result["AR_Order"] = rolling_auto_ar(
    #     data=data, pred_len=target_len, max_order=(20, 2, 0)
    # )

    ###### Informer Module ######
    informer_pred, informer_para, informer_lr = informer_predict(
        informer_len_combinations=informer_len, data=data
    )
    result["Informer"] = informer_pred
    result["Informer_para"] = informer_para
    result["Informer_lr"] = informer_lr
    ###### RNN Module ######
    # train_len = data_length - target_len
    # train_split = int(train_len * 0.8)
    # train_data = data[:train_split]
    # val_data = data[train_split:train_len]
    # test_data = data[train_len:]
    # rnn_predictions = rnn_forecast(
    #     train_data=train_data, val_data=val_data, test_data=test_data
    # )
    # result["RNN"] = rnn_predictions.tolist()

    return result


if __name__ == "__main__":
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
    data_length = 1100
    target_len = 100
    # Parameters for ARMA(2,1) process
    # ar = [1, -0.5, 0.25]  # AR coefficients
    # ma = [1, 0.4]  # MA coefficients
    # informer setting
    pred_len = 1
    d_model = 32  # 512
    d_ff = 512  # 2048
    dropout = 0.2
    # mercury
    # informer_len = [(10, 5), (20, 10), (50, 20)]
    # midway
    informer_len = [(10, 2), (20, 4), (50, 10)]
    lr_lst = [1e-4, 1e-3, 1e-2]
    num = 1
    plot_dir = f"pretrained_val_plots_{num}"
    os.makedirs(plot_dir, exist_ok=True)

    output_file = f"csv_results_pretrained/result_{num}.csv"

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
