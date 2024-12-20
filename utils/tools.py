import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import random

from typing import List


plt.switch_backend("agg")


def set_seed(seed: int = 2024) -> None:
    """
    고정된 랜덤 시드 설정

    Args:
        seed (int): 고정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)


def adjust_learning_rate(optimizer, epoch, learning_rate, lradj, logger):
    if lradj == "type1":
        lr_adjust = {
            epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))
        }
    elif lradj == "type2":
        lr_adjust = {
            2: 5e-5, 
            4: 1e-5, 
            6: 5e-6, 
            8: 1e-6, 
            10: 5e-7, 
            15: 1e-7, 
            20: 5e-8
        }
    elif lradj == "3":
        lr_adjust = {
            epoch: learning_rate if epoch < 10 else learning_rate * 0.1
        }
    elif lradj == "4":
        lr_adjust = {
            epoch: learning_rate if epoch < 15 else learning_rate * 0.1
        }
    elif lradj == "5":
        lr_adjust = {
            epoch: learning_rate if epoch < 25 else learning_rate * 0.1
        }
    elif lradj == "6":
        lr_adjust = {
            epoch: learning_rate if epoch < 5 else learning_rate * 0.1
        }
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        logger.info("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def test_params_flop(model, x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print(
            "INFO: Trainable parameter count: {:.2f}M".format(model_params / 1000000.0)
        )
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True
        )
        # print('Flops:' + flops)
        # print('Params:' + params)
        print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))


def generate_predictions(model, x_list: list, forecast_size: int = 96, return_weights_flag = False) -> List[np.ndarray]:
    """
    Generate predictions using the provided model and raw data.

    Args:
        model (torch.nn.Module): The PyTorch model used for predictions.
        x_list (list): The list of input data.
        forecast_size (int): The size of the forecast window.
        
    Returns:
        List[np.ndarray]: A list of predictions with the same dimension as the forecast windows.
    """
    # Convert lists to tensors
    x_tensor = torch.tensor(x_list, dtype=torch.float32)
    
    # Generate predictions
    model.eval()
    weights_ls = []
    with torch.no_grad():
        if return_weights_flag is True:
            predictions, weights = model(x_tensor, return_weights_flag)
        else:
            predictions = model(x_tensor)
    
    # Ensure predictions have the same shape as y_ls
    predictions = predictions.view(-1, forecast_size).numpy()
    
    if return_weights_flag is True:
        ## 모델별 Weights 리스트(nested list)로 변환
        weights = weights.view(-1, len(x_list)).numpy()
        weights = list(zip(*weights))
        
        return predictions.tolist(), weights
    else:
        return predictions.tolist() 
    

def generate_cnn_modify_x(model, x_list: list, window_size: int = 336) -> List[np.ndarray]:
    """
    Generate CNN_NLinear modified x using the provided model and raw data.

    Args:
        model (torch.nn.Module): The PyTorch model used for predictions.
        x_list (list): The list of input data.
        
    Returns:
        List[np.ndarray]: A list of modified x with CNN_NLinear.
    """
    # Convert lists to tensors
    x_tensor = torch.tensor(x_list, dtype=torch.float32)
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        _, modify_x = model(x_tensor, return_x_flag = True)
    
    # Ensure predictions have the same shape as y_ls
    modify_x = modify_x.view(-1, window_size).numpy()
    
    return modify_x.tolist()
