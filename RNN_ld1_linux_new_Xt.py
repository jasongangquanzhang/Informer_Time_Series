# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:01:56 2024

@author: 12489
"""

#%%
import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import pmdarima as pm
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from copy import deepcopy
import time
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)
randomseed=an

#%%
def piecewise_function(x, theta_1, theta_2):
    if x > 0:
        return theta_1 * x
    else:
        return theta_2 * x


# %%
# 数据生成函数
def generatedata_ld(T, func_type, covis=False):
    Z1 = np.random.uniform(-1, 1, T)
    Z2 = np.random.uniform(-1, 1, T)
    Z3 = np.random.uniform(-1, 1, T)
    Z4 = np.random.uniform(-1, 1, T)
    Z5 = np.random.uniform(-1, 1, T)
    Z6 = np.random.uniform(-1, 1, T)
    Z7 = np.random.uniform(-1, 1, T)
    if func_type == 'abs':
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

    elif func_type == 'arma':
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

    elif func_type == 'lin-non1':
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

    elif func_type == 'lin-non':
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        theta_0 = 0.1
        theta_1 = 0.1
        theta_2 = 0.8

        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = theta_0 * X[t - 1] + U[t] + piecewise_function(U[t - 1], theta_1, theta_2)
            X.append(X_t)
            EX_t = theta_0 * X[t - 1] + piecewise_function(U[t - 1], theta_1, theta_2)
            EX.append(EX_t)

    elif func_type == 'non-non_X':
        U = np.random.randn(T)
        # U = np.random.uniform(-1, 1, T) # 生成长度为T的均匀分布随机数
        beta_0 =  - 0.2
        beta_1 = 0.2
        theta_0 = - 0.2
        theta_1 = 0.2
        theta_2 = -0.2
        theta_3 = 0.8
        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = (piecewise_function(X[t - 1], theta_0, theta_1) + U[t] + piecewise_function(U[t - 1], theta_2,theta_3) 
                    + beta_0 * Z1[t] + beta_0 * Z2[t] + beta_0 * Z3[t]+ beta_1 * Z4[t] + beta_1 * Z5[t] + beta_1 * Z6[t] + beta_1 * Z7[t])
            X.append(X_t)
            EX_t = (piecewise_function(X[t - 1], theta_0, theta_1) + piecewise_function(U[t - 1], theta_2,theta_3) 
                    + beta_0 * Z1[t] + beta_0 * Z2[t] + beta_0 * Z3[t]+ beta_1 * Z4[t] + beta_1 * Z5[t] + beta_1 * Z6[t] + beta_1 * Z7[t])
            EX.append(EX_t)

    elif func_type == 'exp':
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        theta_1 = 0.5
        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = np.exp(-5 * X[t - 1] ** 2) + U[t] + theta_1 * U[t - 1]
            X.append(X_t)
            EX_t = np.exp(-5 * X[t - 1] ** 2) + theta_1 * U[t - 1]
            EX.append(EX_t)

    elif func_type == 'FAR':
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
            X_t = ((a1 + b1 * np.exp(-phi1 * X[t - 1] ** 2)) * X[t - 1] +
                   (a2 + b2 * np.exp(-phi2 * X[t - 1] ** 2)) * X[t - 2] +
                   U[t] + theta_1 * U[t - 1] + theta_2 * U[t - 2])
            X.append(X_t)
            EX_t = ((a1 + b1 * np.exp(-phi1 * X[t - 1] ** 2)) * X[t - 1] +
                    (a2 + b2 * np.exp(-phi2 * X[t - 1] ** 2)) * X[t - 2] +
                    theta_1 * U[t - 1] + theta_2 * U[t - 2])
            EX.append(EX_t)
        
    elif func_type == 'bilinear':
        U = np.random.uniform(-1, 1, T)  # 生成长度为T的均匀分布随机数
        theta_0 = 0.5
        theta_1 = 0.5
        theta_2 = 0.5
        X = [U[0]]  # 初始化序列
        EX = [0]
        for t in range(1, T):
            X_t = theta_0 * X[t - 1] + U[t] + theta_1 * U[t - 1] + theta_2 * X[t - 1] * U[t - 1]
            X.append(X_t)
            EX_t = theta_0 * X[t - 1] + theta_1 * U[t - 1] + theta_2 * X[t - 1] * U[t - 1]
            EX.append(EX_t)
    
    elif func_type == 'STAR':
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
            G_t = 1 / (1 + np.exp(-gamma * (X[t-1] - c)))
            # G.append(G_t)
            X_t = (phi * X[t-1] + theta_1 * X[t-1] * G_t  + 
                    U[t] + theta_2 * U[t-1])
            X.append(X_t)
            EX_t = (phi * X[t-1] + theta_1 * X[t-1] * G_t  + 
                     + theta_2 * U[t-1])
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


# %%
# ARMA(1,1) 模型拟合和预测函数
def fit_ARMA(Xtrain, Xval, Xtest):
    best_val_loss = float('inf')
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
                fitted_model = fitted_model.append([Xval[ii]], refit=False)  # 更新模型状态

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
        self.i2h = nn.Linear(hidden_size, hidden_size)  # 定义线性层，将输入和隐藏状态拼接后的向量映射到隐藏状态
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

#不限制W
class SimpleRNNCell1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNNCell1, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(hidden_size, hidden_size)  
        self.relu = nn.ReLU()  # ReLU 激活函数

    def forward(self, input, hidden):
        hidden_mapped = self.i2h(hidden)  # 对 hidden 进行线性映射
        combined = hidden_mapped + input  # 将映射结果与输入相加
        hidden = self.relu(combined)  # 通过 ReLU 激活函数
        return hidden  # 返回隐藏状态

    def get_W(self):
        return self.i2h.weight.data

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)  # 返回全零的隐藏状态张量

# 自定义的多层ReLU和RNN模型类定义
class CustomReLURNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers, output_layers):
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

        self.head = nn.Sequential(*self.input_relu_layers)  # 使用Sequential模块将前置ReLU层连接起来

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

        self.tail = nn.Sequential(*self.output_relu_layers)  # 使用Sequential模块将后置ReLU层连接起来

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
        # print(input.size())
        hidden = self.rnn_cell.init_hidden(batch_size)  # 初始化隐藏状态
        outputs = []
        for t in range(seq_len):
            x = input[:, t, :]  # 获取输入序列中的第t个时间步的数据
            x = self.head(x)  # 通过前置的多层ReLU层
            # print(x.size())
            hidden = self.rnn_cell(x, hidden)
            output = self.tail(hidden)  # 通过后置的多层ReLU层
            output = self.fc(output)  # 通过最后的全连接层
            # print(output.shape)
            outputs.append(output.unsqueeze(1))  # 将输出添加到输出列表中

        outputs = torch.cat(outputs, dim=1)  # 将所有时间步的输出连接成一个张量
        # if covis:
        #     outputs = outputs.squeeze(-1)
        return outputs

#用 SimpleRNNCell1   
class CustomReLURNN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers, output_layers):
        super(CustomReLURNN1, self).__init__()
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

        self.head = nn.Sequential(*self.input_relu_layers)  # 使用Sequential模块将前置ReLU层连接起来

        # 定义RNN单元
        self.rnn_cell = SimpleRNNCell1(layer_input_size, hidden_size)

        # 定义后置的多层ReLU层
        self.output_relu_layers = []
        layer_input_size = hidden_size
        for layer_output_size in output_layers:
            linear_layer = nn.Linear(layer_input_size, layer_output_size)  # 线性层
            self.output_relu_layers.append(linear_layer)
            self.output_relu_layers.append(nn.ReLU())  # ReLU激活函数
            self.init_weights(linear_layer)  # 初始化权重
            layer_input_size = layer_output_size

        self.tail = nn.Sequential(*self.output_relu_layers)  # 使用Sequential模块将后置ReLU层连接起来

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
        # print(input.size())
        hidden = self.rnn_cell.init_hidden(batch_size)  # 初始化隐藏状态
        outputs = []
        for t in range(seq_len):
            x = input[:, t, :]  # 获取输入序列中的第t个时间步的数据
            x = self.head(x)  # 通过前置的多层ReLU层
            # print(x.size())
            hidden = self.rnn_cell(x, hidden)
            output = self.tail(hidden)  # 通过后置的多层ReLU层
            output = self.fc(output)  # 通过最后的全连接层
            # print(output.shape)
            outputs.append(output.unsqueeze(1))  # 将输出添加到输出列表中

        outputs = torch.cat(outputs, dim=1)  # 将所有时间步的输出连接成一个张量
        # if covis:
        #     outputs = outputs.squeeze(-1)
        return outputs

# DNN
class CustomReLUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers, output_layers):
        super(CustomReLUNet, self).__init__()
        self.hidden_size = hidden_size

        # Define the input fully connected layers with ReLU activations (before "hidden" layers)
        self.input_relu_layers = []
        layer_input_size = input_size
        hidden_layers1 = hidden_layers + [hidden_size]  # Append hidden size as final hidden layer

        for layer_hidden_size in hidden_layers1:
            linear_layer = nn.Linear(layer_input_size, layer_hidden_size)  # Linear layer
            self.input_relu_layers.append(linear_layer)
            self.input_relu_layers.append(nn.ReLU())  # ReLU activation
            self.init_weights(linear_layer)  # Initialize weights
            layer_input_size = layer_hidden_size

        self.head = nn.Sequential(*self.input_relu_layers)  # Connect layers using Sequential

        # Define the output fully connected layers with ReLU activations (after "hidden" layers)
        self.output_relu_layers = []
        layer_input_size = hidden_size
        for layer_output_size in output_layers:
            linear_layer = nn.Linear(layer_input_size, layer_output_size)  # Linear layer
            self.output_relu_layers.append(linear_layer)
            self.output_relu_layers.append(nn.ReLU())  # ReLU activation
            self.init_weights(linear_layer)  # Initialize weights
            layer_input_size = layer_output_size

        self.tail = nn.Sequential(*self.output_relu_layers)  # Connect layers using Sequential

        # Final fully connected layer to map to the output size
        self.fc = nn.Linear(layer_input_size, output_size)
        self.init_weights(self.fc)  # Initialize weights

    # Initialize weights function
    def init_weights(self, layer):
        stdev = (2.0 / layer.in_features) ** 0.5
        layer.weight.data.normal_(0, stdev)  # Initialize weights with normal distribution

    # Forward pass function
    def forward(self, input):
        batch_size, seq_len, _ = input.size()  # Get input tensor dimensions

        outputs = []  # Collect outputs for each time step

        for t in range(seq_len):
            x = input[:, t, :]  # Get the input for the t-th time step
            x = self.head(x)  # Pass through input fully connected layers with ReLU
            x = nn.ReLU()(x)  # Apply ReLU activation to the middle layer output
            output = self.tail(x)  # Pass through output fully connected layers with ReLU
            output = self.fc(output)  # Final fully connected layer to map to output size
            outputs.append(output.unsqueeze(1))  # Collect output for this time step

        outputs = torch.cat(outputs, dim=1)  # Concatenate outputs across all time steps
        # if covis:
        #     outputs = outputs.squeeze(-1)
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

def train_and_evaluate(T, input_size, output_size, relu_layer_sizes_before, hidden_layers, relu_layer_sizes_after, lr,
                       train_loader, val_loader, test_loader, EX, model_type="RNN", constraint = True):
    """
    Train and evaluate either an RNN or DNN model based on the model_type input.

    Parameters:
    - model_type: "RNN" for CustomReLURNN, "DNN" for CustomReLUNet
    """

    # Select the model based on model_type
    if model_type == "RNN":
        if constraint:
            model = CustomReLURNN(input_size, hidden_layers, output_size, relu_layer_sizes_before, relu_layer_sizes_after)
        else:
            model = CustomReLURNN1(input_size, hidden_layers, output_size, relu_layer_sizes_before, relu_layer_sizes_after)
    elif model_type == "DNN":
        model = CustomReLUNet(input_size, hidden_layers, output_size, relu_layer_sizes_before, relu_layer_sizes_after)
    else:
        raise ValueError("Invalid model_type. Choose 'RNN' or 'DNN'.")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    early_stopping = EarlyStopping(patience=100, min_delta=0.0001)

    # Training and validation loop
    num_epochs = 1000
    train_losses = []
    val_losses = []

    # Track the best model
    best_val_loss = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            # print(X_batch.shape)
            optimizer.zero_grad()
            outputs = model(X_batch)
            # print(outputs.shape)
            # print(y_batch.shape)
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
                loss = criterion(val_outputs[-int(0.2 * T):], y1_val[-int(0.2 * T):])
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

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
    test_trueloss = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            test_outputs = model(X_test)

            test_outputs = test_outputs.reshape(-1)
            # print(test_outputs.shape)
            y1_test = y_test.reshape(-1)
            # print(y1_test.shape)
            loss = criterion(test_outputs[-100:], y1_test[-100:])
            test_loss += loss.item()
            loss1 = criterion(test_outputs[-100:], torch.tensor(EX)[-100:])
            test_trueloss += loss1.item()
    test_loss /= len(test_loader)
    test_trueloss /= len(test_loader)
    if model_type == "RNN":
        W = model.get_W()
    else:
        W = None
    pred = np.array(test_outputs[-100:])
    print(f'Test Loss: {test_loss:.4f}, True Test Loss: {test_trueloss:.4f}')

    return best_val_loss, train_losses, val_losses, pred, test_loss, test_trueloss, W


def find_best_model(T, input_size, output_size, lr_list, layer_list, train_loader, val_loader, test_loader, EX,
                    model_type="RNN", constraint = True):
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
    best_val_loss = float('inf')
    best_model_results = None
    best_W = None

    # Loop over all layer configurations
    for relu_layer_sizes_before, hidden_layers, relu_layer_sizes_after in layer_list:
        print(f'Training with layer config: {relu_layer_sizes_before}, {hidden_layers}, {relu_layer_sizes_after}')

        # Initialize variables to track the best learning rate for this specific layer configuration
        best_lr_for_config = None
        best_val_loss_for_config = float('inf')
        best_model_results_for_config = None
        best_W_for_config = None

        # Loop over learning rates for the current layer configuration
        for lr in lr_list:
            print(f'Training with learning rate: {lr}')

            # Call train_and_evaluate function to train the model and evaluate
            val_loss, train_losses, val_losses, pred, test_loss, test_trueloss, W = train_and_evaluate(
                T, input_size, output_size, relu_layer_sizes_before, hidden_layers, relu_layer_sizes_after, lr,
                train_loader, val_loader, test_loader, EX, model_type, constraint
            )

            print(
                f'Learning Rate: {lr}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, True Test Loss: {test_trueloss:.4f}')

            # Update the best learning rate for this configuration if validation loss improves
            if val_loss < best_val_loss_for_config:
                best_val_loss_for_config = val_loss
                best_lr_for_config = lr
                best_model_results_for_config = (val_loss, train_losses, val_losses, pred, test_loss, test_trueloss)
                best_W_for_config = W

        # After going through all learning rates for the current layer configuration,
        # check if this configuration (with its best learning rate) is the best overall
        if best_val_loss_for_config < best_val_loss:
            best_val_loss = best_val_loss_for_config
            best_config = (relu_layer_sizes_before, hidden_layers, relu_layer_sizes_after)
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
        y.append(series[i, 0])  # 当前时间步的元素作为目标值
    return np.array(X), np.array(y)

#%%
# 定义滞后函数，生成滞后一阶和二阶的数据
def create_lagged_features1(series, lag):
    X, y = [], []
    lags = list(range(1, lag+1))
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
        X1 = torch.tensor(self.X[idx:idx + self.seq_length + 1].squeeze(-2), dtype=torch.float32)
        y1 = torch.tensor(self.y[idx:idx + self.seq_length + 1], dtype=torch.float32)
        return X1, y1
    
# 自定义数据集类
class TimeSeriesDataset1(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        
    def __len__(self):
        return self.X.shape[0] - self.seq_length

    def __getitem__(self, idx):
        X1 = torch.tensor(self.X[idx:idx+self.seq_length+1].squeeze(-1))
        y1 = torch.tensor(self.y[idx:idx+self.seq_length+1])
        return X1, y1


# %% 100 replications simulation
def main(T, func_type, randomseed, covis=False):
    # 模型参数
    if func_type == 'FAR':
        p = q = 2
    else:
        p = q = 1

    if covis:
        input_size = 7 + p  # 输入特征的维度
    else:
        input_size = p  # 输入特征的维度
    output_size = 1  # 输出特征的维度
    lr_list = [0.001,0.01,0.1]

    layer_list = [([], 4, []), ([4], 4, [4]), ([8], 4, [8])]
    result = []

    np.random.seed(randomseed)
    torch.manual_seed(randomseed)

    series, EX = generatedata_ld(T + 100, func_type=func_type)

    Xtrain = series[:int(0.8 * T)]
    Xval = series[int(0.8 * T):T]
    Xtest = series[T:]
    best_p, best_q, Xtest_pred = fit_ARMA(Xtrain, Xval, Xtest)
    Xtest_true = EX[T:]
    arma_mse_true = np.mean((Xtest_pred - Xtest_true) ** 2)
    arma_mse = np.mean((Xtest_pred - Xtest) ** 2)

    series, EX = generatedata_ld(T + 100, func_type=func_type, covis = covis)
    if covis:
        X, y = create_lagged_features(series, lag=p)
        y = torch.tensor(y).unsqueeze(1)
        train_X, train_y = X[:int(0.8 * T)], y[:int(0.8 * T)]
        val_X, val_y = X[:(T - 1)], y[:(T - 1)]
        test_X, test_y = X, y
        train_dataset = TimeSeriesDataset(train_X, train_y, train_X.shape[0] - 1)
        val_dataset = TimeSeriesDataset(val_X, val_y, val_X.shape[0] - 1)
        test_dataset = TimeSeriesDataset(test_X, test_y, test_X.shape[0] - 1)
    else:
        series = series[..., np.newaxis].astype(np.float32)
        X, y = create_lagged_features1(series, lag=p)
        train_X, train_y = X[:int(0.8 * T)], y[:int(0.8 * T)]
        val_X, val_y = X[:(T - 1)], y[:(T - 1)]
        test_X, test_y = X, y
        train_dataset = TimeSeriesDataset1(train_X, train_y, train_X.shape[0] - 1)
        val_dataset = TimeSeriesDataset1(val_X, val_y, val_X.shape[0] - 1)
        test_dataset = TimeSeriesDataset1(test_X, test_y, test_X.shape[0] - 1)


    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    start_time = time.time()
    best_config, best_lr, best_model_results, best_W = find_best_model(T, input_size, output_size, lr_list, layer_list,
                                                                       train_loader, val_loader, test_loader, EX, "RNN")
    end_time = time.time()
    t0 = end_time - start_time 
    pred_rnn = best_model_results[3]
    mse_rnn = np.mean((pred_rnn - series[-100:, 0]) ** 2)
    mse_rnn_true = np.mean((pred_rnn - EX[-100:]) ** 2)
    norm_W = torch.max(torch.abs(torch.diag(best_W))).item()
    ss_res = np.sum((pred_rnn - series[-100:, 0]) ** 2)
    ss_tot = np.sum((series[-100:, 0] - np.mean(series[-100:, 0])) ** 2)  # 总平方和
    Rsquare_rnn = 1 - (ss_res / ss_tot)
    
    start_time = time.time()
    best_config0, best_lr0, best_model_results0, best_W0 = find_best_model(T, input_size, output_size, lr_list, layer_list,
                                                                       train_loader, val_loader, test_loader, EX, "RNN", constraint = False)
    end_time = time.time()
    t1 = end_time - start_time 
    pred_rnn1 = best_model_results0[3]
    mse_rnn1 = np.mean((pred_rnn1 - series[-100:, 0]) ** 2)
    mse_rnn_true1 = np.mean((pred_rnn1 - EX[-100:]) ** 2)

    best_config1, best_lr1, best_model_results1, best_W1 = find_best_model(T, input_size, output_size, lr_list, layer_list,
                                                                       train_loader, val_loader, test_loader, EX, "DNN")

    pred_dnn = best_model_results1[3]
    mse_dnn = np.mean((pred_dnn - series[-100:, 0]) ** 2)
    mse_dnn_true = np.mean((pred_dnn - EX[-100:]) ** 2)

    Y_pred = series[:T].mean()
    mse_hm = np.mean((Y_pred - series[-100:, 0]) ** 2)
    mse_hm_true = np.mean((Y_pred - EX[-100:]) ** 2)

    result.append([layer_list.index(best_config), best_lr, mse_rnn,
                   mse_rnn_true, mse_dnn, mse_dnn_true, arma_mse,
                   arma_mse_true, mse_hm, mse_hm_true, norm_W, Rsquare_rnn])

    filepath = f"./RNN/low_d/RNN{T}_" + func_type + "_ld" + ".txt"

    directory = os.path.dirname(filepath)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'a') as f:
        f.write(' '.join(map(str, result)) + '\n')
        
    result1 = np.array([t0, mse_rnn_true, t1, mse_rnn_true1])

    filepath1 = f"./RNN/low_d/RNN{T}_" + func_type + "_Wmse" + ".txt"
    
    # filepath = f"C:/Users/12489/Desktop/RNN{T}_" + func_type + "_Wconstraint" + ".txt"

    directory1 = os.path.dirname(filepath1)

    if not os.path.exists(directory1):
        os.makedirs(directory1)

    with open(filepath1, "a") as file:
        # 将列表转换为字符串，并以逗号分隔每个元素
        line = ",".join(map(str, result1))
        # 写入文件，每次写一行
        file.write(line + "\n")


# %%
# for T in [400]:
#     # main(T,'FAR',an)
#     main(T, 'STAR', an+100, covis = False)

# for T in [200]:
#     main(T, 'non-non_X', an)

# for T in [200, 400]:
#     main(T, 'non-non_X', an,covis=True)

# for T in [600, 800, 1000]:
#     main(T, 'STAR', an, covis = False)
#     main(T, 'bilinear', an, covis = False)
#     main(T, 'non-non_X', an,covis=True)
#     # main(T,'non-non',an)
#     main(T,'FAR',an, covis = False)
#     main(T,'lin-non1',an, covis = False)
#     main(T,'arma',an, covis = False)
    
for T in [200, 400, 600, 800, 1000]:
    main(T, 'STAR', an, covis = False)
    main(T, 'bilinear', an, covis = False)
    main(T, 'non-non_X', an,covis=True)
    # main(T,'non-non',an)
    main(T,'FAR',an, covis = False)
    main(T,'lin-non1',an, covis = False)
    main(T,'arma',an, covis = False)
    # main(T,'abs',an)
