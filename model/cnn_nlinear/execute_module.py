import logging
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Optional

from model.custom_loss import WeightedMSELoss
from utils.metrics import metric


class CNN_NLinear(nn.Module):
    def __init__(
        self,
        window_size: int = 96,
        forecast_size: int = 96,
        conv_kernel_size: int = 10,
        conv_filters: int = 32,
        in_channels: int = 1,
        dropout_rate: float = 0.1,
        logger: logging.Logger = None,
    ) -> None:
        super(CNN_NLinear, self).__init__()
        self.logger = logger
        self.forecast_size = forecast_size

        # Set padding for convolution
        padding = (conv_kernel_size - 1) // 2

        # Single convolution layer with BatchNorm and ReLU
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=conv_filters, kernel_size=conv_kernel_size, padding=padding
        )
        # To prevent overfitting, batch normalize, dropout
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.dropout1 = nn.Dropout(dropout_rate)
        # Adaptive pooling layer to condense information
        self.pool = nn.AdaptiveAvgPool1d(window_size)

        # Linear layer to map convolution filters to raw input dim
        self.linear = nn.Linear(conv_filters, in_channels)
        # To prevent overfitting, batch normalize, dropout
        self.bn2 = nn.BatchNorm1d(window_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Final Linear layer to preict the forecast
        self.final_linear = nn.Linear(window_size, forecast_size)
        
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
    
    def positional_encoding(self, length, d_model):
        """
        Generate positional encoding
        """
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0) 

    def forward(self, x: torch.Tensor, return_x_flag: bool = False) -> torch.Tensor:
        x = x.float()
        
        # Distribution shift removal
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        ## Input data
        x_input = x

        # Positional enconding for Conv1D
        # x = x + self.positional_encoding(x.size(1), x.size(2)).to(x.device)

        # Permute to [batch, feature_size, window_size]
        x = x.permute(0, 2, 1)

        # First and only convolution block
        x = self.dropout1(torch.tanh(self.bn1(self.conv1(x))))
        
        # Pooling layer to condense information
        x = self.pool(x)

        # Permute to [batch, window_size, feature_size]
        x = x.permute(0, 2, 1)

        # Linear formation for conv filters
        x = self.dropout2(torch.tanh(self.bn2(self.linear(x))))
        
        # Residual connection
        x_modify = x + x_input

        # Final linear layer to ensure output has correct dimensions
        output = self.final_linear(x_modify.permute(0, 2, 1)).permute(0, 2, 1)

        # Adding back the removed shift
        output = output + seq_last
        
        if return_x_flag is True:
            return output, x_modify + seq_last
        else:
            return output

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        device: torch.device,
        epochs: int = 100,
        lr: float = 0.005,
        patience: int = 30,
        custom_loss_flag: bool = False,
        best_model_path = "./result/best_model__cnn_nlinear.pth"
    ):
        if custom_loss_flag is True:
            criterion = WeightedMSELoss(weight_type='linear', alpha=0.1)
        else:
            criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=patience // 5, verbose=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.float().to(device), y.float().to(device)
                optimizer.zero_grad()
                y_pred = self(x)
                loss = criterion(y_pred, y)
                loss.backward()

                # Gradient clipping to stabilize training
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
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

                # Logging
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
        Generate predictions using the CNN_NLinear model.

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
        logger.info(f'[CNN + NLinear Score] MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')
        
        pred_result = pd.DataFrame(
            {
                'true': [x[0] for x in np.concatenate(trues).tolist()],
                'pred': [x[0] for x in np.concatenate(preds).tolist()]
            }
        )
        pred_result.insert(0, 'model_name', 'cnn_nlinear')

        return pred_result
