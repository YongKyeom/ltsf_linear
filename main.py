import pandas as pd
import numpy as np
import torch
import sys
import warnings
import os

from datetime import datetime

from common.visualize import plot_predictions
from common.logger import Logger
from config.config import DATE_COL_NM, TARGET_COL_NM, NLINEAR_PARAMETER, HYBRID_PARAMETER, SEED_NUM
from data.data_loader import load_data
from data.data_factor import create_dataloaders
from model.nlinear.execute_module import NLinearModel
from model.dlinear.execute_module import DLinearModel, DNLinearModel
from model.cnn_nlinear.execute_module import CNN_NLinear
from model.hybrid.execute_module import HybridModel
from model.hyperoptimize import optimize_cnn_nlinear
from utils.tools import set_seed

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
    # Initialize NLinear model
    set_seed(SEED_NUM)
    nlinear_model = NLinearModel(
        window_size=NLINEAR_PARAMETER["window_size"],
        forecast_size=NLINEAR_PARAMETER["forecast_size"],
        individual=NLINEAR_PARAMETER["individual"],
        feature_size=NLINEAR_PARAMETER["feature_size"],
        logger=logger,
    ).to(device)
    # Train NLinear model
    nlinear_model.train_model(train_loader, val_loader, test_loader, device, NLINEAR_PARAMETER['epochs'], NLINEAR_PARAMETER['learning_rate'])


    ## ------------------------------------ DLinear Training ------------------------------------ ##
    # Initialize DLinear model
    logger.info("Training DLinear model")
    set_seed(SEED_NUM)
    dlinear_model = DLinearModel(
        window_size=NLINEAR_PARAMETER["window_size"],
        forecast_size=NLINEAR_PARAMETER["forecast_size"],
        individual=NLINEAR_PARAMETER["individual"],
        enc_in=NLINEAR_PARAMETER["feature_size"],
        logger=logger,
    ).to(device)
    # Train NLinear model
    dlinear_model.train_model(train_loader, val_loader, test_loader, device, NLINEAR_PARAMETER['epochs'], NLINEAR_PARAMETER['learning_rate'])


    ## ------------------------------------ DLinear + NLinear Training ------------------------------------ ##
    # Initialize DLinear model
    logger.info("Training DLinear + NLinear model")
    set_seed(SEED_NUM)
    dnlinear_model = DNLinearModel(
        window_size=NLINEAR_PARAMETER["window_size"],
        forecast_size=NLINEAR_PARAMETER["forecast_size"],
        individual=NLINEAR_PARAMETER["individual"],
        enc_in=NLINEAR_PARAMETER["feature_size"],
        kernel_size=NLINEAR_PARAMETER["kernel_size"],
        logger=logger,
    ).to(device)
    # Train NLinear model
    dnlinear_model.train_model(train_loader, val_loader, test_loader, device, NLINEAR_PARAMETER['epochs'], NLINEAR_PARAMETER['learning_rate'])
    
    
    ## ------------------------------------ CNN_NLinear Training ------------------------------------ ##
    # Optimize Hybrid model or use default parameters
    if HYBRID_PARAMETER["opt_hyperpara"] is True:
        hybrid_params = HYBRID_PARAMETER["space"]
    else:
        hybrid_params = HYBRID_PARAMETER["default_space"]
    for key, values in HYBRID_PARAMETER.items():
        if key not in ["opt_hyperpara", "space", "default_space"]:
            hybrid_params[key] = values

    if HYBRID_PARAMETER["opt_hyperpara"] is True:
        logger.info("Optimization of CNN_NLinear model")
        set_seed(SEED_NUM)
        hybrid_best_params = optimize_cnn_nlinear(hybrid_params, train_loader, val_loader, test_loader, device)

        logger.info(f"Best hyperparameters for CNN_NLinear model: {hybrid_best_params}")
    else:
        hybrid_best_params = hybrid_params

    # Train CNN_NLinear model
    logger.info("Training CNN_NLinear model")
    set_seed(SEED_NUM)
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
    cnn_nlinear_model.train_model(train_loader, val_loader, test_loader, device, hybrid_best_params['epochs'], hybrid_best_params['learning_rate'])


    ## ------------------------------------ NLinear + CNN_NLinear Training ------------------------------------ ##
    logger.info("Training Hybrid model")
    set_seed(SEED_NUM)
    ## Initialize Hybrid model
    hybrid_model = HybridModel(
        models=[nlinear_model, dlinear_model, dnlinear_model, cnn_nlinear_model],
        window_size=hybrid_best_params["window_size"],
        dropout_rate=hybrid_best_params["dropout_rate"],
        logger=logger,
    ).to(device)
    # Train Hybrid model
    hybrid_model.train_model(train_loader, val_loader, test_loader, device, hybrid_best_params['epochs'], hybrid_best_params['learning_rate'])

    
    ## ------------------------------------ Predict for test set ------------------------------------ ##
    nlinear_predictions = nlinear_model.predict(test_loader, device)
    cnn_nlinear_predictions = cnn_nlinear_model.predict(test_loader, device)
    hybrid_predictions = hybrid_model.predict(test_loader, device)

    # Evaluate both models
    logger.info("Evaluating models.")
    nlinear_pred_result = nlinear_model.final_predict(test_set, test_loader, device, logger)
    dlinear_pred_result = dlinear_model.final_predict(test_set, test_loader, device, logger)
    dnlinear_pred_result = dnlinear_model.final_predict(test_set, test_loader, device, logger)
    cnn_nlinear_pred_result = cnn_nlinear_model.final_predict(test_set, test_loader, device, logger)
    hybrid_pred_result = hybrid_model.final_predict(test_set, test_loader, device, logger)

    
    # ## ------------------------------------ Visualize Predict Result ------------------------------------ ##
    # # Plot predictions
    # plot_predictions(train_data, val_data, test_data, {"NLinear": nlinear_predictions, "Hybrid": hybrid_predictions})

    
    # ------------------------------------ End of Process ------------------------------------
    END_TIME = datetime.now()
    logger.info('main.py Elapsed time: {!s}'.format(END_TIME - ST_TIME))
    exit(0)
    