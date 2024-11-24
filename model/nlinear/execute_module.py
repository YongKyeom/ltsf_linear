import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Optional

from utils.metrics import metric


class NLinearModel(nn.Module):
    def __init__(
        self,
        window_size: int = 96,
        forecast_size: int = 96,
        individual: bool = False,
        feature_size: int = 7,
        logger: logging.Logger = None,
    ):
        """
        NLinear Model initializer.

        Args:
            window_size (int): Input sequence length.
            forecast_size (int): Output forecast length.
            individual (bool): If True, separate linear layers for each feature.
            feature_size (int): Number of features in input data.
        """
        super(NLinearModel, self).__init__()
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.individual = individual
        self.channels = feature_size
        self.logger = logger

        if self.individual:
            self.Linear = nn.ModuleList([nn.Linear(self.window_size, self.forecast_size) for _ in range(self.channels)])
        else:
            self.Linear = nn.Linear(self.window_size, self.forecast_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the NLinear model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, feature_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, forecast_size, feature_size).
        """
        x = x.float()
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last  # Distribution shift removal
        if self.individual:
            output = torch.zeros([x.size(0), self.forecast_size, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last  # Adding back the removed shift
        return x

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        device: torch.device,
        epochs: int = 50,
        lr: float = 0.001,
        patience: int = 10,
        best_model_path = "./result/best_model__nlinear.pth"
    ):
        """
        Train the NLinear model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            device (torch.device): Device to run the model on.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                x, y = x.float().to(device), y.float().to(device)
                optimizer.zero_grad()
                y_pred = self(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            if test_loader:
                self.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.float().to(device), y.float().to(device)
                        y_pred = self(x)
                        test_loss += criterion(y_pred, y).item()
                avg_test_loss = test_loss / len(test_loader)

            # Validation loss calculation
            if val_loader:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.float().to(device), y.float().to(device)
                        y_pred = self(x)
                        val_loss += criterion(y_pred, y).item()
                avg_val_loss = val_loss / len(val_loader)
                if self.logger is not None:
                    self.logger.info(
                        f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
                    )
                
                # Apply learning rate decay
                scheduler.step(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if self.logger is not None:
                        self.logger.info("Save best model")
                    torch.save(self.state_dict(), best_model_path)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if self.logger is not None:
                        self.logger.info("Early stopping triggered.")
                    break
            else:
                if self.logger is not None:
                    self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
                    if (epoch + 1) == epochs:
                        torch.save(self.state_dict(), best_model_path)
        
        self.load_state_dict(torch.load(best_model_path))

    def predict(self, data_loader: DataLoader, device: torch.device) -> np.array:
        """
        Generate predictions using the NLinear model.

        Args:
            data_loader (DataLoader): DataLoader for input data.
            device (torch.device): Device to run the model on.
            output_format (str): Format of the returned predictions ('numpy', 'list', 'series').

        Returns:
            np.array: Predictions in the specified format.
        """
        self.eval()
        prediction_result = []
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                predictions = self(x).detach().numpy()
                for pred_values in predictions:
                    prediction_result.append(pred_values.reshape(1, -1)[0])

        return prediction_result
    
    def final_predict(self, pred_data, pred_loader: Optional[DataLoader], device: torch.device, logger: logging.Logger):
        
        preds = []
        trues = []
        self.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(pred_loader):
                x, y = x.float().to(device), y.float().to(device)
                
                outputs = self(x)
                pred = outputs.detach().cpu().numpy()
                true = y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        logger.info(f'[NLinear Score] MSE:{mse:.4f}, MAE:{mae:.4f}, MAPE: {mape:.4f}, Corr: {corr:.4f}')
        
        pred_result = pd.DataFrame(
            {
                'true': [x[0] for x in np.concatenate(trues).tolist()],
                'pred': [x[0] for x in np.concatenate(preds).tolist()]
            }
        )
        pred_result.insert(0, 'model_name', 'nlinear')

        return pred_result
