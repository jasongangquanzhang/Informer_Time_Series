import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


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
