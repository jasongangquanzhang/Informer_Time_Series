import os
import torch
import numpy as np
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


# Set up random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


###### ARMA Benchmark ######
def rolling_auto_arima(
    data, pred_len, information_criterion="bic", seasonal=False, max_order=(5, 2, 5)
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

    # Calculate training length
    train_len = len(data) - pred_len

    # Split data into training and test sets
    train = data[:train_len]
    forecasts = []

    # Use auto_arima to determine the best ARIMA order
    try:
        arima_model = auto_arima(
            train,
            seasonal=seasonal,
            max_p=max_order[0],
            max_d=max_order[1],
            max_q=max_order[2],
            information_criterion=information_criterion,
            stepwise=True,
            suppress_warnings=True,
        )
        print(f"Selected Order: {arima_model.order}")
    except Exception as e:
        print(f"ARIMA model fitting failed: {e}")
        sys.exit(-1)

    # Perform rolling forecast
    for i in range(pred_len):
        # Forecast the next value
        forecast = arima_model.predict(n_periods=1)[0]
        forecasts.append(forecast)

        # Update the model with the latest observed value
        new_data = [data[train_len + i]]  # Only the current observed value
        arima_model.update(new_data)

    return forecasts


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
    def __init__(self, data, seq_len, label_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.label_len - self.pred_len + 1

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


def iterative_prediction_with_update(
    model, test_data, seq_len, label_len, pred_len, target_len, device
):
    """
    Perform iterative prediction and update the model with each new prediction and true value.
    """
    predictions = []
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001
    )  # Use the same optimizer as during training
    criterion = torch.nn.MSELoss()  # Define the loss function

    for step in range(target_len):
        # Initialize encoder and decoder inputs
        enc_in = (
            torch.tensor(test_data[step : step + seq_len], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(device)
        )
        dec_in = (
            torch.tensor(
                test_data[step + seq_len - label_len : step + seq_len + pred_len],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(device)
        )

        # Set the last point of decoder input to 0

        dec_in[:, -1, :] = 0

        # Make a prediction
        model.eval()
        with torch.no_grad():
            pred = (
                model(enc_in, enc_in, dec_in, dec_in).squeeze(-1).cpu().numpy()[0, -1]
            )

        # Append prediction and true value to the results
        predictions.append(pred)

        # # Update the model using the prediction and true value
        # model.train()
        # enc_input_update = enc_in.detach().clone()
        # dec_input_update = dec_in.detach().clone()
        # true_value = torch.tensor(
        #     [[test_data[seq_len + step]]], dtype=torch.float32
        # ).to(device)

        # optimizer.zero_grad()
        # updated_pred = model(
        #     enc_input_update, enc_input_update, dec_input_update, dec_input_update
        # ).squeeze(-1)
        # loss = criterion(updated_pred.squeeze(-1), true_value.squeeze(-1))
        # loss.backward()
        # optimizer.step()
    return predictions


# Train the Informer model
def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    checkpoint_path="checkpoint.pth",
):
    """
    Train the model and return the final validation loss.
    """
    early_stopping = EarlyStopping(patience=5, verbose=True, path=checkpoint_path)
    best_val_loss = float("inf")  # Track the best validation loss

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Training loop
        for enc_input, dec_input, target in train_loader:
            enc_input, dec_input, target = (
                enc_input.unsqueeze(-1).to(device),
                dec_input.unsqueeze(-1).to(device),
                target.to(device),
            )
            dec_input[:, -1, :] = 0  # Mask the last bit of decoder input
            optimizer.zero_grad()
            y_pred = model(enc_input, enc_input, dec_input, dec_input)
            loss = criterion(y_pred.squeeze(-1), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for enc_input, dec_input, target in val_loader:
                enc_input, dec_input, target = (
                    enc_input.unsqueeze(-1).to(device),
                    dec_input.unsqueeze(-1).to(device),
                    target.to(device),
                )
                y_pred = model(enc_input, enc_input, dec_input, dec_input)
                loss = criterion(y_pred.squeeze(-1), target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Check early stopping
        early_stopping(val_loss, model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update best validation loss

        if early_stopping.early_stop:
            print("Early stopping triggered. Loading the best model...")
            model.load_state_dict(torch.load(checkpoint_path))
            break

    return best_val_loss


# def informer_predict(informer_len, data):

#     # Split training data into train and validation sets
#     train_len = data_length - target_len
#     train_split = int(train_len * 0.8)
#     train_data = data[:train_split]
#     val_data = data[train_split:train_len]
#     for seq_len, label_len in informer_len:
#         test_data = data[train_len - seq_len :]
#         for lr in lr_lst:

#             # Split training data into train and validation sets

#             # Prepare datasets and loaders
#             train_dataset = TimeSeriesDataset(train_data, seq_len, label_len, pred_len)
#             val_dataset = TimeSeriesDataset(val_data, seq_len, label_len, pred_len)
#             train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#             val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#             # Prepare datasets and loaders
#             train_dataset = TimeSeriesDataset(train_data, seq_len, label_len, pred_len)
#             val_dataset = TimeSeriesDataset(val_data, seq_len, label_len, pred_len)
#             train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#             val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
#                 d_model=512,
#                 n_heads=8,
#                 e_layers=2,
#                 d_layers=1,
#                 d_ff=2048,
#                 dropout=0.05,
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
#             optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#             train(
#                 model, train_loader, val_loader, criterion, optimizer, epochs=10, device=device
#             )

#         ###### Iterative Prediction ######
#         informer_predictions = iterative_prediction_with_update(
#             model, test_data, seq_len, label_len, pred_len, target_len, device
#         )

#     return informer_predictions


def informer_predict(informer_len_combinations, data):
    """
    Perform grid search over seq_len and label_len combinations to choose the best one based on validation loss.
    """
    train_len = len(data) - target_len
    train_split = int(train_len * 0.8)
    train_data = data[:train_split]
    val_data = data[train_split:train_len]

    best_val_loss = float("inf")
    best_combination = None
    best_model = None

    # Iterate over all seq_len and label_len combinations
    for seq_len, label_len in informer_len_combinations:
        for lr in lr_lst:
            # Prepare datasets and loaders
            train_dataset = TimeSeriesDataset(train_data, seq_len, label_len, pred_len)
            val_dataset = TimeSeriesDataset(val_data, seq_len, label_len, pred_len)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Model setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            model = Informer(
                enc_in=1,
                dec_in=1,
                c_out=1,
                seq_len=seq_len,
                label_len=label_len,
                out_len=pred_len,
                factor=5,
                d_model=512,
                n_heads=8,
                e_layers=2,
                d_layers=1,
                d_ff=2048,
                dropout=0.05,
                attn="prob",
                embed="fixed",
                freq="h",
                activation="gelu",
                output_attention=False,
                distil=True,
                mix=True,
                device=device,
            ).to(device)

            # Training setup
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            val_loss = train(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                epochs=10,
                device=device,
                checkpoint_path=checkpoint_path
            )

            # Update the best combination
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_combination = (seq_len, label_len)
                best_model = model.state_dict()  # Save model parameters

    print(
        f"Best Combination: seq_len: {best_combination[0]}, label_len: {best_combination[1]}, Val Loss: {best_val_loss:.4f}"
    )

    # Load the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Informer(
        enc_in=1,
        dec_in=1,
        c_out=1,
        seq_len=best_combination[0],
        label_len=best_combination[1],
        out_len=pred_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.05,
        attn="prob",
        embed="fixed",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
        device=device,
    ).to(device)
    model.load_state_dict(best_model)

    # Perform iterative prediction using the best model
    informer_predictions = iterative_prediction_with_update(
        model,
        data[train_len - seq_len :],
        best_combination[0],
        best_combination[1],
        pred_len,
        target_len,
        device,
    )

    return informer_predictions, best_combination


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
    data = generate_arma_time_series(ar, ma, data_length)
    # std = data.std()
    true_value = data[-target_len:].tolist()
    # result['STD'] = std
    result["True"] = true_value

    ###### ARMA Module ######
    arma_predictions = rolling_auto_arima(data=data, pred_len=target_len)
    result["ARMA"] = arma_predictions

    ###### Informer Module ######
    informer_pred, informer_para = informer_predict(
        informer_len_combinations=informer_len, data=data
    )
    result["Informer"] = informer_pred
    result["Informer_para"] = informer_para
    ###### RNN Module ######
    train_len = data_length - target_len
    train_split = int(train_len * 0.8)
    train_data = data[:train_split]
    val_data = data[train_split:train_len]
    test_data = data[train_len:]
    rnn_predictions = rnn_forecast(
        train_data=train_data, val_data=val_data, test_data=test_data
    )
    result["RNN"] = rnn_predictions.tolist()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate Informer on synthetic time series with ARMA benchmark."
    )
    parser.add_argument(
        "integer", metavar="N", type=int, help="an integer for the accumulator"
    )
    arg = parser.parse_args()
    seed = int(arg.integer)
    # Generate data
    data_length = 1000
    target_len = 10
    # Parameters for ARMA(2,1) process
    ar = [1, -0.5, 0.25]  # AR coefficients
    ma = [1, 0.4]  # MA coefficients
    # informer setting
    pred_len = 1
    informer_len = [(10, 5), (20, 10), (50, 20), (100, 50)]
    lr_lst = [0.01, 0.001, 0.0001]
    
    # informer_len = [(50, 10)]
    # lr_lst = [0.0001]   
    # 6 cancel iterate update model
    output_file = "csv_results/result_8.csv" 

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
