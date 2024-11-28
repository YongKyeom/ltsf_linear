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


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        
        return res, moving_mean

class DLinearModel(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(
            self, 
            window_size: int = 96,
            forecast_size: int = 96,
            individual: bool = False,
            enc_in: int = 7,
            kernel_size: int = 25,
            logger: logging.Logger = None,
        ):
        super(DLinearModel, self).__init__()
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.individual = individual
        self.channels = enc_in
        self.logger = logger

        # Decompsition Kernel Size
        self.kernel_size = kernel_size
        self.decompsition = series_decomp(self.kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.window_size, self.forecast_size))
                self.Linear_Trend.append(nn.Linear(self.window_size, self.forecast_size))

        else:
            self.Linear_Seasonal = nn.Linear(self.window_size, self.forecast_size)
            self.Linear_Trend = nn.Linear(self.window_size, self.forecast_size)

    def forward(self, x):
        """
        Forward pass for the DLinear model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, feature_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, forecast_size, feature_size).
        """
        x = x.float()
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.forecast_size],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.forecast_size],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        
        return x.permute(0,2,1)
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        device: torch.device,
        epochs: int = 100,
        lr: float = 0.005,
        patience: int = 15,
        best_model_path = "./result/best_model__dlinear.pth"
    ):
        """
        Train the DLinear model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            device (torch.device): Device to run the model on.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=patience // 2, verbose=True)

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
        Generate predictions using the DLinear model.

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
        logger.info(f'[DLinear Score] MSE: {mse:.4f}, MAE:{mae:.4f}, MAPE: {mape:.4f}')
        
        pred_result = pd.DataFrame(
            {
                'true': [x[0] for x in np.concatenate(trues).tolist()],
                'pred': [x[0] for x in np.concatenate(preds).tolist()]
            }
        )
        pred_result.insert(0, 'model_name', 'nlinear')

        return pred_result


class DNLinearModel(nn.Module):
    """
    Decomposition + Normalize Linear
    """
    def __init__(
            self, 
            window_size: int = 96,
            forecast_size: int = 96,
            individual: bool = False,
            enc_in: int = 7,
            kernel_size: int = 25,
            logger: logging.Logger = None,
        ):
        super(DNLinearModel, self).__init__()
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.individual = individual
        self.channels = enc_in
        self.logger = logger

        # Decompsition Kernel Size
        self.kernel_size = kernel_size
        self.decompsition = series_decomp(self.kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.window_size, self.forecast_size))
                self.Linear_Trend.append(nn.Linear(self.window_size, self.forecast_size))

        else:
            self.Linear_Seasonal = nn.Linear(self.window_size, self.forecast_size)
            self.Linear_Trend = nn.Linear(self.window_size, self.forecast_size)
    
        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights for all layers
        """
        if isinstance(m, nn.Linear):
            # Xavier Initialization for Linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            # Kaiming (He) Initialization for Conv1d layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            # BatchNorm Initialization: weight to 1, bias to 0
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            # MultiheadAttention Initialization
            for name, param in m.named_parameters():
                if 'in_proj_weight' in name or 'out_proj.weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'in_proj_bias' in name or 'out_proj.bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        """
        Forward pass for the DNLinear model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, feature_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, forecast_size, feature_size).
        """
        x = x.float()
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last  # Distribution shift removal

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.forecast_size],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.forecast_size],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output

        x = x.permute(0,2,1) + seq_last  # Adding back the removed shift
        
        return x
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        device: torch.device,
        epochs: int = 100,
        lr: float = 0.005,
        patience: int = 15,
        best_model_path = "./result/best_model__dnlinear.pth"
    ):
        """
        Train the DLinear model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            device (torch.device): Device to run the model on.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=patience // 2, verbose=True)

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
        Generate predictions using the DNLinear model.

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
    
    def final_predict(self, pred_data, pred_loader: Optional[DataLoader], device: torch.device, logger: logging.Logger) -> pd.DataFrame:
        
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
        logger.info(f'[DNLinear Score] MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')
        
        pred_result = pd.DataFrame(
            {
                'true': [x[0] for x in np.concatenate(trues).tolist()],
                'pred': [x[0] for x in np.concatenate(preds).tolist()]
            }
        )
        pred_result.insert(0, 'model_name', 'nlinear')

        return pred_result
