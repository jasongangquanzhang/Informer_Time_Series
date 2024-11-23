# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 20:26:45 2024

@author: 12489
"""

# %%
import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import matplotlib.dates as mdates
import pmdarima as pm
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "integer", metavar="N", type=int, help="an integer for the accumulator"
)
arg = parser.parse_args()
an = int(arg.integer)
randomseed = an


# %%
# ARMA(1,1) 模型拟合和预测函数
def fit_ARMA(Xtrain, Xval, Xtest):
    best_val_loss = float("inf")
    best_p = None
    best_q = None
    best_model = None

    # 对 p 和 q 进行遍历，选择最佳 ARIMA(p, 0, q) 模型
    for p in [0, 1, 2, 3]:
        for q in [0, 1, 2, 3]:
            # 使用 Xtrain 数据拟合 ARIMA 模型
            model = sm.tsa.ARIMA(Xtrain, order=(p, 0, q))
            fitted_model = model.fit()

            # 用于存储验证集预测
            Xval_pred = []

            # 逐步使用验证集数据进行预测并更新模型
            for ii in range(len(Xval)):
                forecast = fitted_model.forecast(steps=1)[0]  # 预测 Xval 的第 ii 步
                Xval_pred.append(forecast)  # 保存预测值
                fitted_model = fitted_model.append(
                    [Xval[ii]], refit=False
                )  # 更新模型状态

            # 计算验证集的 MSE
            val_loss = np.mean((Xval - np.array(Xval_pred)) ** 2)

            # 如果验证损失比当前最优小，则更新最优参数
            if val_loss < best_val_loss:
                best_p = p
                best_q = q
                best_val_loss = val_loss
                best_model = fitted_model

    # 在测试集上进行预测，使用最优模型
    Xtest_pred = []
    for ii in range(len(Xtest)):
        forecast = best_model.forecast(steps=1)[0]  # 预测 Xtest 的第 ii 步
        Xtest_pred.append(forecast)  # 保存预测值
        best_model = best_model.append([Xtest[ii]], refit=False)  # 更新模型状态

    return best_p, best_q, Xtest_pred


# %%
# 定义RNN类，前置多层ReLU=>RNN=>后置多层ReLU


# 简单RNN单元类定义
class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(
            hidden_size, hidden_size
        )  # 定义线性层，将输入和隐藏状态拼接后的向量映射到隐藏状态
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.make_weights_upper_triangular()  # 在初始化时调用

    def make_weights_upper_triangular(self):
        # 获取线性层的权重
        with torch.no_grad():
            # 创建一个与 i2h 权重矩阵相同大小的上三角矩阵
            upper_triangular_mask = torch.triu(torch.ones_like(self.i2h.weight))
            # 强制将 h2h 的下三角部分设为 0
            self.i2h.weight.data *= upper_triangular_mask

    def forward(self, input, hidden):
        self.make_weights_upper_triangular()  # 确保在前向传播前矩阵为上三角
        hidden_mapped = self.i2h(hidden)  # 对 hidden 进行线性映射
        combined = hidden_mapped + input  # 将映射结果与输入相加
        hidden = self.relu(combined)  # 通过 ReLU 激活函数
        return hidden  # 返回 hidden 和 i2h 的权重矩阵

    def get_W(self):
        return self.i2h.weight.data

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)  # 返回全零的隐藏状态张量


# 自定义的多层ReLU和RNN模型类定义
class CustomReLURNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, hidden_layers, output_layers
    ):
        super(CustomReLURNN, self).__init__()
        self.hidden_size = hidden_size

        # 定义前置的多层ReLU层
        self.input_relu_layers = []
        layer_input_size = input_size
        hidden_layers1 = hidden_layers + [hidden_size]
        # print(hidden_layers1)
        for layer_hidden_size in hidden_layers1:
            linear_layer = nn.Linear(layer_input_size, layer_hidden_size)  # 线性层
            self.input_relu_layers.append(linear_layer)
            self.input_relu_layers.append(nn.ReLU())  # ReLU激活函数
            self.init_weights(linear_layer)  # 初始化权重
            layer_input_size = layer_hidden_size

        self.head = nn.Sequential(
            *self.input_relu_layers
        )  # 使用Sequential模块将前置ReLU层连接起来

        # 定义RNN单元
        self.rnn_cell = SimpleRNNCell(layer_input_size, hidden_size)

        # 定义后置的多层ReLU层
        self.output_relu_layers = []
        layer_input_size = hidden_size
        for layer_output_size in output_layers:
            linear_layer = nn.Linear(layer_input_size, layer_output_size)  # 线性层
            self.output_relu_layers.append(linear_layer)
            self.output_relu_layers.append(nn.ReLU())  # ReLU激活函数
            self.init_weights(linear_layer)  # 初始化权重
            layer_input_size = layer_output_size

        self.tail = nn.Sequential(
            *self.output_relu_layers
        )  # 使用Sequential模块将后置ReLU层连接起来

        # 最后的全连接层
        self.fc = nn.Linear(layer_input_size, output_size)
        self.init_weights(self.fc)  # 初始化权重

    # 初始化权重函数
    def init_weights(self, layer):
        stdev = (2.0 / layer.in_features) ** 0.5
        layer.weight.data.normal_(0, stdev)  # 使用正态分布初始化权重

    def get_W(self):
        return self.rnn_cell.get_W()

    # 前向传播函数
    def forward(self, input):
        batch_size, seq_len, _ = input.size()  # 获取输入张量的尺寸

        hidden = self.rnn_cell.init_hidden(batch_size)  # 初始化隐藏状态
        outputs = []
        for t in range(seq_len):
            x = input[:, t, :]  # 获取输入序列中的第t个时间步的数据
            x = self.head(x)  # 通过前置的多层ReLU层
            hidden = self.rnn_cell(x, hidden)
            output = self.tail(hidden)  # 通过后置的多层ReLU层
            output = self.fc(output)  # 通过最后的全连接层
            outputs.append(output.unsqueeze(1))  # 将输出添加到输出列表中

        outputs = torch.cat(outputs, dim=1)  # 将所有时间步的输出连接成一个张量
        return outputs


class CustomReLUNet(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, hidden_layers, output_layers
    ):
        super(CustomReLUNet, self).__init__()
        self.hidden_size = hidden_size

        # Define the input fully connected layers with ReLU activations (before "hidden" layers)
        self.input_relu_layers = []
        layer_input_size = input_size
        hidden_layers1 = hidden_layers + [
            hidden_size
        ]  # Append hidden size as final hidden layer

        for layer_hidden_size in hidden_layers1:
            linear_layer = nn.Linear(
                layer_input_size, layer_hidden_size
            )  # Linear layer
            self.input_relu_layers.append(linear_layer)
            self.input_relu_layers.append(nn.ReLU())  # ReLU activation
            self.init_weights(linear_layer)  # Initialize weights
            layer_input_size = layer_hidden_size

        self.head = nn.Sequential(
            *self.input_relu_layers
        )  # Connect layers using Sequential

        # Define the output fully connected layers with ReLU activations (after "hidden" layers)
        self.output_relu_layers = []
        layer_input_size = hidden_size
        for layer_output_size in output_layers:
            linear_layer = nn.Linear(
                layer_input_size, layer_output_size
            )  # Linear layer
            self.output_relu_layers.append(linear_layer)
            self.output_relu_layers.append(nn.ReLU())  # ReLU activation
            self.init_weights(linear_layer)  # Initialize weights
            layer_input_size = layer_output_size

        self.tail = nn.Sequential(
            *self.output_relu_layers
        )  # Connect layers using Sequential

        # Final fully connected layer to map to the output size
        self.fc = nn.Linear(layer_input_size, output_size)
        self.init_weights(self.fc)  # Initialize weights

    # Initialize weights function
    def init_weights(self, layer):
        stdev = (2.0 / layer.in_features) ** 0.5
        layer.weight.data.normal_(
            0, stdev
        )  # Initialize weights with normal distribution

    # Forward pass function
    def forward(self, input):
        batch_size, seq_len, _ = input.size()  # Get input tensor dimensions

        outputs = []  # Collect outputs for each time step

        for t in range(seq_len):
            x = input[:, t, :]  # Get the input for the t-th time step
            x = self.head(x)  # Pass through input fully connected layers with ReLU
            x = nn.ReLU()(x)  # Apply ReLU activation to the middle layer output
            output = self.tail(
                x
            )  # Pass through output fully connected layers with ReLU
            output = self.fc(
                output
            )  # Final fully connected layer to map to output size
            outputs.append(output.unsqueeze(1))  # Collect output for this time step

        outputs = torch.cat(outputs, dim=1)  # Concatenate outputs across all time steps
        return outputs  # Return the final output tensor


# %%
# EarlyStopping 类
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# %%


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

    early_stopping = EarlyStopping(patience=100, min_delta=0.0001)

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

        early_stopping(val_loss)
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


# %%
# 定义滞后函数，生成滞后一阶和二阶的数据
def create_lagged_features(series, lag):
    X, y = [], []
    lags = list(range(1, lag + 1))
    for i in range(max(lags), len(series)):
        features = [series[i - lag] for lag in lags]  # 获取滞后的元素
        X.append(features)
        y.append(series[i])  # 当前时间步的元素作为目标值
    return np.array(X), np.array(y)


# %%
# 自定义数据集类
class TimeSeriesDataset(Dataset):
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


# %%
# 取每周五的值作为周汇率并求return
def data_return(data, order):
    data = data.to_numpy()
    pos_s = np.where(data[:, 0] == "2000-01-07")
    pos_e = np.where(data[:, 0] == "2023-12-29")

    data = data[pos_s[0][0] : pos_e[0][0] + 1, :]

    for i in range(1, data.shape[0]):  # 从第二行开始遍历
        if data[i, 1] == ".":
            data[i, 1] = data[i - 1, 1]  # 将'.'替换为上一行的值

    data[:, 1] = data[:, 1].astype(float)
    time_point = np.arange(0, len(data))
    time_point1 = time_point[::5]

    data = data[time_point1, :]
    if order == 1:
        data[:, 1] = 1 / data[:, 1]

    # return
    second_column = data.copy()[:, 1]
    second_column[1:] = ((second_column[1:] / second_column[:-1]) - 1) * 100
    data_re = data.copy()[1:,]
    data_re[:, 1] = second_column[1:]

    return data_re


# %%
def return_plot1(data):
    start_year = 2000
    start_date = f"{start_year}-01-14"  # 从1月1日开始

    # 生成与每周数据对应的日期
    dates = pd.date_range(start=start_date, periods=len(data), freq="W-FRI")

    values = data[:, 1]

    plt.figure(figsize=(14, 3))
    # 绘制折线图，并调整点和线的粗细
    plt.plot(dates, values, marker="o", linestyle="-", markersize=1, linewidth=0.5)

    # 设置横坐标显示为年份，并确保每年只显示一次
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())

    # 限制x轴范围只显示数据的年份范围
    plt.xlim(dates[0], dates[-1])

    # 自动调整日期标签的显示角度
    plt.gcf().autofmt_xdate()
    # 显示图形
    plt.show()


# %%
def pred(Xtrain, Xval, Xtest):
    hm_pred = np.mean(np.concatenate((Xtrain, Xval)))
    best_p, best_q, arma_pred = fit_ARMA(Xtrain, Xval, Xtest)

    # RNN
    input_size_list = [1, 2, 3]
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

            train_dataset = TimeSeriesDataset(train_X, train_y, train_X.shape[0] - 1)
            val_dataset = TimeSeriesDataset(val_X, val_y, val_X.shape[0] - 1)
            test_dataset = TimeSeriesDataset(test_X, test_y, test_X.shape[0] - 1)

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            # DNN
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
                "DNN",
            )

            best_val_loss_for_config_DNN = best_model_results[0]
            if best_val_loss_for_config_DNN < best_val_loss_DNN:
                best_val_loss_DNN = best_val_loss_for_config_DNN
                best_input_size_DNN = input_size
                DNN_pred = best_model_results[3]

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

        DNN_ensemble.append(DNN_pred)
        RNN_ensemble.append(RNN_pred)

    RNN_ensemble1 = np.zeros((ensemble, len(test_X)-(T1+T2)))
    DNN_ensemble1 = np.zeros((ensemble, len(test_X)-(T1+T2)))

    for i in range(ensemble):
        DNN_ensemble1[i,:] = DNN_ensemble[i]
        RNN_ensemble1[i,:] = RNN_ensemble[i]

    DNN_pred = np.mean(DNN_ensemble1, axis=0)
    RNN_pred = np.mean(RNN_ensemble1, axis=0)


    return (hm_pred, arma_pred, DNN_pred, RNN_pred)


# %%
def main(data, m1, m2, an, name):
    T = len(data)
    series = np.vectorize(float)(data[:, 1])

    result = [an]
    # expanding window
    Xtrain = series[: (m2 + 4 * (an - 1))]
    Xval = series[(m2 + 4 * (an - 1)) : (m1 + 4 * (an - 1))]
    Xtest = series[(m1 + 4 * (an - 1)) : min(m1 + 4 * an, T)]
    hm_pred, arma_pred, DNN_pred, RNN_pred = pred(Xtrain, Xval, Xtest)
    result.append(hm_pred)
    result.extend(arma_pred)
    result.extend(DNN_pred)
    result.extend(RNN_pred)
    # result=[an, hm_pred, arma_pred, list(DNN_pred), list(RNN_pred)]

    filepath = f"/home/zshen10/ZGQ/RNNResearch/real_data/exchange_" + name + ".txt"
    # filepath = f"C:/Users/12489/Desktop/"+name+"_ld"+".txt"

    directory = os.path.dirname(filepath)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, "a") as f:
        f.write(" ".join(map(str, result)) + "\n")


# %%
# 指定文件夹路径
folder_path = "/home/zshen10/ZGQ/RNNResearch/real_data/"

# 指定文件名
file_name1 = "DEXUSEU.csv"
file_path1 = os.path.join(folder_path, file_name1)
file_name2 = "DEXUSUK.csv"
file_path2 = os.path.join(folder_path, file_name2)
file_name3 = "DEXSZUS.csv"
file_path3 = os.path.join(folder_path, file_name3)
file_name4 = "DEXJPUS.csv"
file_path4 = os.path.join(folder_path, file_name4)

# 读取CSV文件
USEU = pd.read_csv(file_path1)
USUK = pd.read_csv(file_path2)
SZUS = pd.read_csv(file_path3)
JPUS = pd.read_csv(file_path4)

# 顺序变为兑美元汇率
EUUS_re = data_return(USEU, order=1)
UKUS_re = data_return(USUK, order=1)
SZUS_re = data_return(SZUS, order=0)
JPUS_re = data_return(JPUS, order=0)

plt1 = return_plot1(EUUS_re)


# 训练集由7年开始每次增加4周，训练集保持3年数据不变
m1 = 520
m2 = 364
n = math.ceil((len(EUUS_re) - m1) / 4)


main(EUUS_re, m1, m2, an, name="EUUS")
# main(UKUS_re, m1, m2, an, name='UKUS')
# main(SZUS_re, m1, m2, an, name='SZUS')
# main(JPUS_re, m1, m2, an, name='JPUS')
