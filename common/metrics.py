from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def compute_mae(y_true, y_pred):
    y_true_np = y_true.cpu().numpy().reshape(-1, y_true.shape[-1])
    y_pred_np = y_pred.cpu().numpy().reshape(-1, y_pred.shape[-1])
    return mean_absolute_error(y_true_np, y_pred_np)


def compute_rmse(y_true, y_pred):
    y_true_np = y_true.cpu().numpy().reshape(-1, y_true.shape[-1])
    y_pred_np = y_pred.cpu().numpy().reshape(-1, y_pred.shape[-1])
    return np.sqrt(mean_squared_error(y_true_np, y_pred_np))


def compute_mdape(y_true, y_pred):
    y_true_np = y_true.cpu().numpy().reshape(-1, y_true.shape[-1])
    y_pred_np = y_pred.cpu().numpy().reshape(-1, y_pred.shape[-1])
    return np.median(np.abs((y_true_np - y_pred_np) / y_true_np)) * 100


def compute_corr(y_true, y_pred):
    y_true_np = y_true.cpu().numpy().reshape(-1)
    y_pred_np = y_pred.cpu().numpy().reshape(-1)
    return np.corrcoef(y_true_np, y_pred_np)[0, 1]
