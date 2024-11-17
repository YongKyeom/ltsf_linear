import pandas as pd
import numpy as np
import torch
import sys
import warnings
import os

from datetime import datetime

from common.metrics import compute_mae, compute_rmse, compute_mdape, compute_corr
from common.visualize import plot_predictions
from common.logger import Logger
from config.config import CORE_CNT, DATE_COL_NM, TARGET_COL_NM, NLINEAR_PARAMETER, HYBRID_PARAMETER
from data.data_loader import load_data
from data.data_factor import data_provider, create_dataloaders
from model.nlinear.execute_module import NLinearModel
from model.hybrid.execute_module import HybridModel, CNN_NLinear
from model.hyperoptimize import optimize_hybrid


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

pd.options.display.float_format = "{:.2f}".format
pd.options.display.min_rows = 10
pd.options.display.max_rows = 100
pd.options.display.max_columns = None
pd.options.display.max_colwidth = 30


if __name__ == "__main__":
    ST_TIME = datetime.now()
    
    ## ------------------------------------ Logger ------------------------------------ ##
    CREATED_TIME = datetime.now()
    LOG_PATH = "./logs/"
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    LOGFILE_NM = "nLinear_hybrid_experiment_log"

    ## Define Logger
    logger = Logger(path=LOG_PATH, name=LOGFILE_NM, date=CREATED_TIME).logger

    logger.info("NLinear, CNN_NLiear Hybrid is Start")

    
    ## ------------------------------------ Load Data ------------------------------------ ##
    # Load and split data
    raw_data = load_data("./dataset/ETTh1.csv", date_col_nm=DATE_COL_NM, target_col_nm=TARGET_COL_NM)
    (
        (train_set, train_loader),
        (val_set, val_loader),
        (test_set, test_loader),
        (pred_set, pred_loader)
    ) =  create_dataloaders(
        embed = 'timeF', 
        train_only = False,
        batch_size = NLINEAR_PARAMETER['batch_size'],
        freq = 'h',
        data_type_list = ['train', 'val', 'test', 'pred'],
        seq_len = NLINEAR_PARAMETER['window_size'],
        label_len = NLINEAR_PARAMETER['forecast_size'],
        pred_len = NLINEAR_PARAMETER['forecast_size'],
        features = 'S',
        target = 'OT',
        root_path = './dataset',
        data_path = 'ETTh1.csv',
    )

    
    ## ------------------------------------ NLinear Training ------------------------------------ ##
    # Set torch
    device = torch.device("cpu")
    
    # Hyper paremter of NLinear
    logger.info("Training NLinear model")
    nlinear_params = NLINEAR_PARAMETER["default_space"]
    for key, values in NLINEAR_PARAMETER.items():
        if key not in ["opt_hyperpara", "space", "default_space"]:
            nlinear_params[key] = values
    # Initialize NLinear model
    nlinear_model = NLinearModel(
        window_size=nlinear_params["window_size"],
        forecast_size=nlinear_params["forecast_size"],
        individual=nlinear_params["individual"],
        feature_size=nlinear_params["feature_size"],
        logger=logger,
    ).to(device)
    # Train NLinear model
    nlinear_model.train_model(train_loader, val_loader, device)

    
    ## ------------------------------------ CNN_NLinear Training ------------------------------------ ##
    # Optimize Hybrid model or use default parameters
    logger.info("Training CNN_NLinear model")
    if HYBRID_PARAMETER["opt_hyperpara"] is True:
        hybrid_params = HYBRID_PARAMETER["space"]
    else:
        hybrid_params = HYBRID_PARAMETER["default_space"]
    for key, values in HYBRID_PARAMETER.items():
        if key not in ["opt_hyperpara", "space", "default_space"]:
            hybrid_params[key] = values

    if HYBRID_PARAMETER["opt_hyperpara"] is True:
        hybrid_best_params = optimize_hybrid(hybrid_params, train_loader, val_loader, device)
    else:
        hybrid_best_params = hybrid_params

    # Initialize CNN_NLinear model
    cnn_nlinear_model = CNN_NLinear(
        window_size=hybrid_best_params["window_size"],
        forecast_size=hybrid_best_params["forecast_size"],
        conv_kernel_size=hybrid_best_params["conv_kernel_size"],
        conv_filters=hybrid_best_params["conv_filters"],
        in_channels=hybrid_best_params["in_channels"],
        dropout_rate=hybrid_best_params["dropout_rate"],
        logger=logger,
    ).to(device)
    # Train CNN_NLinear model
    cnn_nlinear_model.train_model(train_loader, val_loader, device)


    ## ------------------------------------ NLinear + CNN_NLinear Training ------------------------------------ ##
    logger.info("Training Hybrid(NLinear + CNN_NLinear) model")
    ## Initialize Hybrid model
    hybrid_model = HybridModel(
        nlinear_model=nlinear_model,
        cnn_nlinear_model=cnn_nlinear_model,
        num_heads=hybrid_best_params["num_heads"],
        window_size=hybrid_best_params["window_size"],
        forecast_size=hybrid_best_params["forecast_size"],
        logger=logger,
    ).to(device)
    # Train Hybrid model
    hybrid_model.train_model(train_loader, val_loader, device)

    
    ## ------------------------------------ Predict for test set ------------------------------------ ##
    nlinear_predictions = nlinear_model.predict(test_loader, device)
    cnn_nlinear_predictions = cnn_nlinear_model.predict(test_loader, device)
    hybrid_predictions = hybrid_model.predict(test_loader, device)

    # Evaluate both models
    logger.info("Evaluating models.")
    # metrics = {"MAE": compute_mae, "RMSE": compute_rmse, "MdAPE": compute_mdape, "CORR": compute_corr}

    # for name, func in metrics.items():
    #     logger.info(f"NLinear {name}: {func(test_data, nlinear_predictions)}")
    #     logger.info(f"CNN_NLinear {name}: {func(test_data, cnn_nlinear_predictions)}")
    #     logger.info(f"Hybrid {name}: {func(test_data, hybrid_predictions)}")

    
    # ## ------------------------------------ Visualize Predict Result ------------------------------------ ##
    # # Plot predictions
    # plot_predictions(train_data, val_data, test_data, {"NLinear": nlinear_predictions, "Hybrid": hybrid_predictions})

    
    # ------------------------------------ End of Process ------------------------------------
    END_TIME = datetime.now()
    print('main.py Elapsed time: {!s}'.format(END_TIME - ST_TIME))
    exit()