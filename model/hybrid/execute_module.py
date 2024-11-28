import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from utils.metrics import metric


class HybridModel(nn.Module):
    def __init__(
        self, 
        models: list,
        window_size: int = 332,
        feature_dim: int = 1,
        dropout_rate: float = 0.1,
        logger: logging.Logger = None
    ):
        super(HybridModel, self).__init__()
        self.num_models = len(models)
        self.dropout_rate = dropout_rate
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.logger = logger
        
        # Attention weights for each model
        self.query_weights = nn.ParameterList([nn.Parameter(torch.Tensor(feature_dim, feature_dim)) for _ in range(self.num_models)])
        self.key_weights = nn.ParameterList([nn.Parameter(torch.Tensor(feature_dim, feature_dim)) for _ in range(self.num_models)])
        self.value_weights = nn.ParameterList([nn.Parameter(torch.Tensor(feature_dim, feature_dim)) for _ in range(self.num_models)])

        # Dropout & Normalization for Attention output
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_norm = nn.LayerNorm(normalized_shape=[window_size, feature_dim])
        
        # Residual Connection 
        self.residual_weight = nn.Parameter(torch.Tensor(1))
        
        # Weight 초기화
        self._initialize_weights()

        # Prior models
        self.models = nn.ModuleList(models)

        # Freeze model parameters
        for model in self.models:
            for _, param in model.named_parameters():
                param.requires_grad = False

    def _initialize_weights(self):
        for weight in self.query_weights:
            nn.init.xavier_uniform_(weight)
        for weight in self.key_weights:
            nn.init.xavier_uniform_(weight)
        for weight in self.value_weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        x = x.float()
        
        # Model outputs
        model_outputs = [model(x) for model in self.models]

        # Calculate Attention for each model output
        attention_outputs = []
        for i, output in enumerate(model_outputs):
            query = torch.matmul(x, self.query_weights[i])
            key = torch.matmul(output, self.key_weights[i])
            value = torch.matmul(output, self.value_weights[i])
            
            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.feature_dim ** 0.5)
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, value)
            
            output = self.attention_norm(self.dropout(output))
            attention_outputs.append(output.mean(dim=1))
            
        # Calculate weights using Softmax
        weights = torch.softmax(torch.stack(attention_outputs, dim=0), dim=0)

        # Calculate weighted sum of model outputs
        final_output = sum(w.unsqueeze(1) * output for w, output in zip(weights, model_outputs))

        # # Residual Connection
        # final_output = (1 - self.residual_weight) * final_output + self.residual_weight * sum(model_outputs) / self.num_models
        
        return final_output
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        device: torch.device,
        epochs: int = 50,
        lr: float = 0.001,
        patience: int = 10,
        best_model_path = "./result/best_model__hybrid.pth"
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
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
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

                # Learning rate decay
                scheduler.step(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if self.logger is not None:
                        self.logger.info("Save best model")
                    # Save best model
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
        Generate predictions using the Hybrid model.

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
        logger.info(f'[Hybrid model Score] MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')
        
        pred_result = pd.DataFrame(
            {
                'true': [x[0] for x in np.concatenate(trues).tolist()],
                'pred': [x[0] for x in np.concatenate(preds).tolist()]
            }
        )
        pred_result.insert(0, 'model_name', 'hybrid')

        return pred_result
