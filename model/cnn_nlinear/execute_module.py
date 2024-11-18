import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Optional


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
        self.bn1 = nn.BatchNorm1d(conv_filters)

        # Adaptive pooling layer to condense information
        self.pool = nn.AdaptiveAvgPool1d(window_size)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)
        # Linear layer to map the output to the forecast size
        self.linear = nn.Linear(window_size * conv_filters, forecast_size)

        # Final linear layer to reduce output to single channel for each forecast step
        self.final_linear = nn.Linear(forecast_size, forecast_size)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights for all layers
        """
        if isinstance(m, nn.Linear):
            # Xavier 초기화
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            # MultiheadAttention 초기화
            for name, param in m.named_parameters():
                if 'in_proj_weight' in name or 'out_proj.weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'in_proj_bias' in name or 'out_proj.bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        
        # Distribution shift removal
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        # Permute to [batch, feature_size, window_size]
        x = x.permute(0, 2, 1)

        # First and only convolution block
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        # Pooling layer to condense information
        x = self.pool(x)

        # Flatten the convolutional output
        x = x.view(x.size(0), -1)

        # Linear transformation to map to forecast size
        x = self.linear(x).view(x.size(0), -1)

        # Final linear layer to ensure output has correct dimensions
        x = self.final_linear(x).view(x.size(0), self.forecast_size, 1)

        # Adding back the removed shift
        x = x + seq_last
        
        return x

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        epochs: int = 50,
        lr: float = 0.001,
        patience: int = 10,
    ):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=True)

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

            # Validation loss calculation
            if val_loader:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        y_pred = self(x)
                        val_loss += criterion(y_pred, y).item()
                avg_val_loss = val_loss / len(val_loader)

                # Logging
                if self.logger is not None:
                    self.logger.info(
                        f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                    )

                # Apply learning rate decay
                scheduler.step(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if self.logger is not None:
                        self.logger.info("Save best model")
                    torch.save(self.state_dict(), "best_model__cnn_nlinear.pth")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if self.logger is not None:
                        self.logger.info("Early stopping triggered.")
                    break
            else:
                if self.logger is not None:
                    self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
                    self.logger.info(f"Save CNN_NLinear model")
                # Save the model in case of no validation data
                torch.save(self.state_dict(), "best_model_no_val__cnn_nlinear.pth")

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
        final_predictions = []
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                predictions = self(x).detach().numpy()
                for pred_values in predictions:
                    final_predictions.append(pred_values.reshape(1, -1)[0])

        return final_predictions
