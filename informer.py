
import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from models.model import (
    Informer,
)  # Assuming Informer model is already defined and available
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_process import ArmaProcess
import argparse
import csv
import sys
from pmdarima import auto_arima
from filelock import Timeout, FileLock


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


def iterative_prediction_with_update(model, test_data, seq_len, label_len, pred_len,target_len, device):
    """
    Perform iterative prediction and update the model with each new prediction and true value.
    """
    predictions = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Use the same optimizer as during training
    criterion = torch.nn.MSELoss()  # Define the loss function


    for step in range(target_len):
        # Initialize encoder and decoder inputs
        enc_in = torch.tensor(test_data[step:step+seq_len], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        dec_in = torch.tensor(test_data[step+seq_len - label_len:step+seq_len + pred_len], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        # Set the last point of decoder input to 0
        
        dec_in[:, -1, :] = 0

        # Make a prediction
        model.eval()
        with torch.no_grad():
            pred = model(enc_in, enc_in, dec_in, dec_in).squeeze(-1).cpu().numpy()[0, -1]

        # Append prediction and true value to the results
        predictions.append(pred)

        # Update the model using the prediction and true value
        model.train()
        enc_input_update = enc_in.detach().clone()
        dec_input_update = dec_in.detach().clone()
        true_value = torch.tensor([[test_data[seq_len + step]]], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        updated_pred = model(enc_input_update, enc_input_update, dec_input_update, dec_input_update).squeeze(-1)
        loss = criterion(updated_pred.squeeze(-1), true_value.squeeze(-1))
        loss.backward()
        optimizer.step()
    return predictions

# Train the Informer model
def train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    early_stopping = EarlyStopping(patience=5, verbose=True, path=checkpoint_path)

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
        if early_stopping.early_stop:
            print("Early stopping triggered. Loading the best model...")
            model.load_state_dict(torch.load(checkpoint_path))
            break



# Main function
def main():
    set_seed(seed)
    result = {}
    result["seed"] = seed

    # Generate synthetic ARMA time series data
    data = generate_arma_time_series(ar, ma, data_length)
    true_value = data[-target_len:].tolist()
    result["True"] = true_value

    ###### ARMA Module ######
    arma_predictions = rolling_auto_arima(data=data, pred_len=target_len)
    result["ARMA"] = arma_predictions

    ###### Informer Module ######
    train_len = data_length - target_len
    train_split = int(train_len * 0.8)
    train_data_split = data[:train_split]
    val_data_split = data[train_split:train_len]
    test_data = data[train_len - seq_len :]
    # Split training data into train and validation sets

    # Prepare datasets and loaders
    train_dataset = TimeSeriesDataset(train_data_split, seq_len, label_len, pred_len)
    val_dataset = TimeSeriesDataset(val_data_split, seq_len, label_len, pred_len)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(model, train_loader, val_loader, criterion, optimizer, epochs=10, device=device)

    ###### Iterative Prediction ######
    informer_predictions = iterative_prediction_with_update(
        model, test_data, seq_len, label_len, pred_len, target_len, device
    )
    result["Informer"] = informer_predictions

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
    seq_len, label_len, pred_len = 50, 10, 1

    output_file = "csv_results/result_1.csv"
    
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
