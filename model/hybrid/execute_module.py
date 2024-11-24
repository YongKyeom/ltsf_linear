import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Optional

from model.nlinear.execute_module import NLinearModel
from model.dlinear.execute_module import DLinearModel
from model.cnn_nlinear.execute_module import CNN_NLinear
from utils.metrics import metric


class HybridModel(nn.Module):
    def __init__(
        self,
        nlinear_model: NLinearModel,
        cnn_nlinear_model: CNN_NLinear,
        num_heads: int = 4,
        window_size: int = 336,
        forecast_size: int = 96,
        dropout_rate: float = 0.1,
        logger: logging.Logger = None,
    ) -> None:
        super(HybridModel, self).__init__()
        
        self.logger = logger
        self.num_heads = num_heads
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.embed_dim = window_size

        # Multihead Attention
        self.attention = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        # Output Layer
        self.out_proj = nn.Linear(self.embed_dim, 1)

        # Residual weight
        self.residual_weight = nn.Parameter(torch.Tensor(1))

        # Weight initialization
        self.apply(self._init_weights)

        # NLinear, CNN_NLinear model
        self.nlinear_model = nlinear_model
        self.cnn_nlinear_model = cnn_nlinear_model

        ## Freeze NLineaer, CNN_Linear model
        for name, param in self.nlinear_model.named_parameters():
            if name in ['linear.0.weight', 'linear.2.weight']:
                param.requires_grad = True
        for name, param in self.cnn_nlinear_model.named_parameters():
            if name in ['linear.0.weight', 'linear.2.weight']:
                param.requires_grad = True

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
        x = x.float()
        batch_size, window_size, _ = x.size()

        # x를 embedding 차원으로 확장
        x_embed = x.unsqueeze(2).expand(-1, -1, self.forecast_size, -1)
        x_embed = x_embed.permute(0, 2, 1, 3)

        # Output of NLinear, CNN_NLinear
        nlinear_output = self.nlinear_model(x)
        cnn_nlinear_output = self.cnn_nlinear_model(x)

        # Calculate Attention
        combined_inputs = x_embed.view(batch_size, self.forecast_size, window_size)
        combined_inputs = combined_inputs.permute(1, 0, 2)
        attn_output, attn_weights = self.attention(combined_inputs, combined_inputs, combined_inputs)

        # w <- Attention average
        attn_output_mean = attn_output.mean(dim=2)
        w = torch.sigmoid(attn_output_mean).permute(1, 0).unsqueeze(-1) 

        # Output = w * NLinear + (1-w) * CNN_NLinear
        weighted_sum = w * nlinear_output + (1 - w) * cnn_nlinear_output

        # Residual Connection
        final_output = weighted_sum + self.residual_weight * (nlinear_output + cnn_nlinear_output)

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
        logger.info(f'[Hybrid model Score] MSE:{mse:.4f}, MAE:{mae:.4f}, MAPE: {mape:.4f}, Corr: {corr:.4f}')
        
        pred_result = pd.DataFrame(
            {
                'true': [x[0] for x in np.concatenate(trues).tolist()],
                'pred': [x[0] for x in np.concatenate(preds).tolist()]
            }
        )
        pred_result.insert(0, 'model_name', 'hybrid')

        return pred_result
