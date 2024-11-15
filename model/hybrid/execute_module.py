import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Optional


import torch
import torch.nn as nn


class CNN_NLinear(nn.Module):
    def __init__(
        self,
        window_size: int,
        forecast_size: int,
        conv_kernel_size: int,
        conv_filters: int,
        in_channels: int,
        dropout_rate: float,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save residual connection
        # residual = x
        seq_last = x[:, -1:, :].detach()
        # Distribution shift removal
        x = x - seq_last

        # Permute to [batch, feature_size, window_size]
        x = x.permute(0, 2, 1)

        # First and only convolution block
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        # Pooling layer to condense information
        x = self.pool(x)

        # # Residual connection with broadcasting
        # x = x + residual

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
                x, y = x.to(device), y.to(device)
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

    # def predict(self, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    #     self.eval()
    #     predictions = []
    #     with torch.no_grad():
    #         for x, _ in data_loader:  # Only retrieve x, ignore y
    #             x = x.to(device)
    #             predictions.append(self(x).cpu())

    #     return torch.cat(predictions)

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


class HybridModel(nn.Module):
    def __init__(
        self,
        nlinear_model: "NLinear",
        cnn_nlinear_model: "CNN_NLinear",
        input_dim: int,
        hidden_dim: int,
        logger: logging.Logger = None,
    ) -> None:
        super(HybridModel, self).__init__()
        self.logger = logger
        self.nlinear_model = nlinear_model
        self.cnn_nlinear_model = cnn_nlinear_model

        # Attention layers
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.fc_out = nn.Linear(hidden_dim, 1)  # Combine NLinear and CNN outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute Query, Key, Value from input data
        Q = self.query(x)
        K = self.key(x)
        V = self.value(K)

        # Attention mechanism
        attention_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5))
        attention_output = torch.matmul(attention_weights, V)

        # Apply individual models
        nlinear_output = self.nlinear_model(x)
        cnn_nlinear_output = self.cnn_nlinear_model(x)

        # Combine model outputs using attention results
        combined_output = torch.stack((nlinear_output, cnn_nlinear_output), dim=-1)

        # Ensure attention_output shape matches combined_output for broadcasting
        attention_output = self.fc_out(attention_output).squeeze(-1)
        model_importance = self.softmax(attention_output).unsqueeze(-1).unsqueeze(-1)

        # Adjust model_importance to match combined_output shape
        model_importance = model_importance.expand_as(combined_output)

        # Weighted sum of NLinear and CNN outputs
        output = (combined_output * model_importance).sum(dim=-1)

        return output

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
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = self(x)
                loss = criterion(y_pred, y)
                loss.backward()

                # Gradient clipping to stabilize training
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
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

                if self.logger is not None:
                    self.logger.info(
                        f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                    )

                # Learning rate decay
                scheduler.step(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if self.logger is not None:
                        self.logger.info("Save best model")
                    # Save best model
                    torch.save(self.state_dict(), "best_model__hybrid.pth")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if self.logger is not None:
                        self.logger.info("Early stopping triggered.")
                    break
            else:
                if self.logger is not None:
                    self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
                # Save final model
                torch.save(self.state_dict(), "best_model_no_val__hybrid.pth")

    # def predict(self, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    #     self.eval()
    #     predictions = []
    #     attention_weights_debug = []

    #     with torch.no_grad():
    #         for x, _ in data_loader:
    #             x = x.to(device)
    #             output = self(x)

    #             if hasattr(self, "attention_weights"):
    #                 attention_weights_debug.append(self.attention_weights.cpu())

    #             predictions.append(output.cpu())

    #     if self.logger is not None and attention_weights_debug:
    #         self.logger.debug(f"Attention Weights: {torch.cat(attention_weights_debug)}")

    #     return torch.cat(predictions)

    def predict(self, data_loader: DataLoader, device: torch.device) -> np.array:
        """
        Generate predictions using the Hybrid model.

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
