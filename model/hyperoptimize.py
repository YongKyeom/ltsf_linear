import numpy as np
import torch
import torch.nn as nn

from hyperopt import fmin, tpe, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss
from model.nlinear.execute_module import NLinearModel
from model.cnn_nlinear.execute_module import CNN_NLinear
from model.hybrid.execute_module import HybridModel
from typing import Dict, Any

from common.metrics import compute_mse
from config.config import SEED_NUM

def objective_nlinear(params: Dict[str, Any], train_loader, val_loader, test_loader, device) -> float:
    """
    NLinear모델의 Hyper-paremter 최적화를 위한 fmin 목적함수
    """
    model = NLinearModel(
        window_size=params["window_size"],
        forecast_size=params["forecast_size"],
        individual=params["individual"],
    ).to(device)
    model.train_model(train_loader, val_loader, test_loader, device)
    
    # Validation RMSE calculation
    criterion = nn.MSELoss()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y_true in val_loader:
            x, y_true = x.float().to(device), y_true.float().to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
            val_loss += loss
    val_loss /= len(val_loader)

    return {"loss": val_loss, "status": STATUS_OK}


def optimize_nlinear(space: Dict[str, Any], train_loader, val_loader, test_loader, device) -> Dict[str, Any]:
    """
    NLinear 모델 Hyper optimization 수행함수

    Returns:
        dict: Best hyper-parameter
    """
    trials = Trials()
    best = fmin(
        fn=lambda params: objective_nlinear(params, train_loader, val_loader, test_loader, device),
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        rstate=np.random.Generator(np.random.PCG64(SEED_NUM)),
        early_stop_fn=no_progress_loss(30)
    )

    return best


def objective_cnn_nlinear(params: Dict[str, Any], train_loader, val_loader, test_loader, device) -> float:
    """
    CNN_NLinear모델의 Hyper-paremter 최적화를 위한 fmin 목적함수
    """
    model = CNN_NLinear(
        window_size=params["window_size"],
        forecast_size=params["forecast_size"],
        conv_kernel_size=max(int(params["conv_kernel_size"]), 3),
        conv_filters=max(int(params["conv_filters"]), 3),
        in_channels=params["in_channels"],
        dropout_rate=min(max(params["dropout_rate"], 0), 0.5)
    ).to(device)
    model.train_model(train_loader, val_loader, test_loader, device)

    # Validation RMSE calculation
    criterion = nn.MSELoss()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y_true in val_loader:
            x, y_true = x.float().to(device), y_true.float().to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
            val_loss += loss
    val_loss /= len(val_loader)

    return {"loss": val_loss, "status": STATUS_OK}


def optimize_cnn_nlinear(space: Dict[str, Any], train_loader, val_loader, test_loader, device) -> Dict[str, Any]:
    """
    CNN_NLinear 모델 Hyper optimization 수행함수

    Returns:
        dict: Best hyper-parameter
    """
    trials = Trials()
    best = fmin(
        fn=lambda params: objective_cnn_nlinear(params, train_loader, val_loader, test_loader, device),
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        rstate=np.random.Generator(np.random.PCG64(SEED_NUM)),
        early_stop_fn=no_progress_loss(30)
    )

    return best


def objective_hybrid(params: Dict[str, Any], train_loader, val_loader, test_loader, device) -> float:
    """
    NLinear + CNN_NLinear 모델의 Hyper-paremter 최적화를 위한 fmin 목적함수
    """
    nlinear_model = NLinearModel(
        window_size=params["window_size"],
        forecast_size=params["forecast_size"],
        individual=params["individual"],
        feature_size=params["in_channels"],
    ).to(device)

    cnn_nlinear_model = CNN_NLinear(
        window_size=params["window_size"],
        forecast_size=params["forecast_size"],
        conv_kernel_size=max(int(params["conv_kernel_size"]), 3),
        conv_filters=max(int(params["conv_filters"]), 3),
        in_channels=params["in_channels"],
        dropout_rate=max(params["dropout_rate"], 0),
    ).to(device)

    model = HybridModel(
        nlinear_model=nlinear_model,
        cnn_nlinear_model=cnn_nlinear_model,
        window_size=params["window_size"],
        dropout_rate=params["dropout_rate"]
    ).to(device)
    model.train_model(train_loader, val_loader, test_loader, device)

    # Validation RMSE calculation
    criterion = nn.MSELoss()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y_true in val_loader:
            x, y_true = x.float().to(device), y_true.float().to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
            val_loss += loss
    val_loss /= len(val_loader)

    return {"loss": val_loss, "status": STATUS_OK}


def optimize_hybrid(space: Dict[str, Any], train_loader, val_loader, test_loader, device) -> Dict[str, Any]:
    """
    NLinear + CNN_NLinear 모델 Hyper optimization 수행함수

    Returns:
        dict: Best hyper-parameter
    """
    trials = Trials()
    best = fmin(
        fn=lambda params: objective_hybrid(params, train_loader, val_loader, test_loader, device),
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        rstate=np.random.Generator(np.random.PCG64(SEED_NUM)),
        early_stop_fn=no_progress_loss(30)
    )

    return best
