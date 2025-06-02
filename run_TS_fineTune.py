import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from models.model import (
    Informer,
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
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
import matplotlib.pyplot as plt

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule, MoiraiFinetune
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule


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

def fine_tune_predict(data):
    date_range = pd.date_range("2000-01-01", periods=data_length, freq="D")
    df = pd.DataFrame({"value": data}, index=date_range)

    # 3. Convert into GluonTS dataset
    ds = PandasDataset({"value": df["value"]})

    MODEL = "moirai"  # model name: choose from {'moirai', 'moirai-moe'}
    SIZE = "base"  # model size: choose from {'small', 'base', 'large'}
    PDT = pred_len  # prediction length: any positive integer
    CTX = 20  # context length: any positive integer
    PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
    BSZ = 32  # batch size: any positive integer
    TEST = target_len  # test set length: any positive integer
    train, test_template = split(ds, offset=-TEST)


    if MODEL == "moirai":
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
            prediction_length=PDT,
            context_length=CTX,
            patch_size=PSZ,
            num_samples=1,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
            prediction_length=PDT,
            context_length=CTX,
            patch_size=16,
            num_samples=1,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )

    test_data = test_template.generate_instances(
        prediction_length=PDT, windows=TEST, distance=PDT,
    )
    predictor = model.create_predictor(batch_size=BSZ)
    forecasts = predictor.predict(test_data.input)

    input_it = iter(test_data.input)
    label_it = iter(test_data.label)
    forecast_it = iter(forecasts)

    forecast_result = []
    for i in range(TEST):
        print(i)
        inp = next(input_it)
        label = next(label_it)
        forecast = next(forecast_it)
        forecast_result.append(forecast.samples[0][0])
    return forecast_result



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

    # ###### ARMA Module ######
    # (
    #     result["ARMA"],
    #     result["ARMA_Order"],
    #     result["ARMA_Train_loss"],
    #     result["ARMA_Valid_loss"],
    # ) = rolling_auto_arima(data=data, pred_len=target_len)

    # result["AR"], result["AR_Order"] = rolling_auto_ar(
    #     data=data, pred_len=target_len, max_order=(20, 2, 0)
    # )

    ####### Fine-tune Moirai ######
    result["Moirai"] = fine_tune_predict(data)
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
    num = 1
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
