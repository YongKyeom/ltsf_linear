from hyperopt import fmin, tpe, Trials, STATUS_OK, hp
from model.nlinear.execute_module import NLinearModel
from model.hybrid.execute_module import HybridModel, CNN_NLinear
from typing import Dict, Any
import torch
from common.metrics import compute_rmse


def objective_nlinear(params: Dict[str, Any], train_loader, val_loader, device) -> float:
    """
    NLinear모델의 Hyper-paremter 최적화를 위한 fmin 목적함수
    """
    model = NLinearModel(
        window_size=params["window_size"],
        forecast_size=params["forecast_size"],
        individual=params["individual"],
    ).to(device)
    model.train_model(train_loader, val_loader, device)

    # Validation RMSE calculation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y_true in val_loader:
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            val_loss += compute_rmse(y_true, y_pred).item()
    val_loss /= len(val_loader)

    return {"loss": val_loss, "status": STATUS_OK}


def optimize_nlinear(space: Dict[str, Any], train_loader, val_loader) -> Dict[str, Any]:
    """
    NLinear 모델 Hyper optimization 수행함수

    Returns:
        dict: Best hyper-parameter
    """
    trials = Trials()
    best = fmin(
        fn=lambda params: objective_nlinear(params, train_loader, val_loader, device),
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
    )

    return best


def objective_hybrid(params: Dict[str, Any], train_loader, val_loader, device) -> float:
    """
    NLinear + CNN_NLinear 모델의 Hyper-paremter 최적화를 위한 fmin 목적함수
    """
    nlinear_model = NLinearModel(
        window_size=params["window_size"],
        forecast_size=params["forecast_size"],
        individual=params["individual"],
        feature_size=max(int(params["feature_size"]), 10),
    ).to(device)

    cnn_nlinear_model = CNN_NLinear(
        window_size=params["window_size"],
        forecast_size=params["forecast_size"],
        conv_kernel_size=max(int(params["conv_kernel_size"]), 3),
        conv_filters=max(int(params["conv_filters"]), 3),
        in_channels=params["in_channels"],
        dropout_rate=max(int(params["dropout_rate"]), 0),
    ).to(device)

    model = HybridModel(
        nlinear_model=nlinear_model,
        cnn_nlinear_model=cnn_nlinear_model,
        input_dim=params["in_channels"],
        hidden_dim=max(int(params["hidden_dim"]), 3),
    ).to(device)
    model.train_model(train_loader, val_loader, device)

    # Validation RMSE calculation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y_true in val_loader:
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            val_loss += compute_rmse(y_true, y_pred).item()
    val_loss /= len(val_loader)

    return {"loss": val_loss, "status": STATUS_OK}


def optimize_hybrid(space: Dict[str, Any], train_loader, val_loader, device) -> Dict[str, Any]:
    """
    NLinear + CNN_NLinear 모델 Hyper optimization 수행함수

    Returns:
        dict: Best hyper-parameter
    """
    trials = Trials()
    best = fmin(
        fn=lambda params: objective_hybrid(params, train_loader, val_loader, device),
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
    )

    return best
