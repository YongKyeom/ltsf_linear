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
    LOGFILE_NM = "ltsf_linear_experiment_log"

    ## Define Logger
    logger = Logger(path=LOG_PATH, name=LOGFILE_NM, date=CREATED_TIME).logger

    logger.info("NLinear, DLinear, NDLinear, CNN_NLiear, Stacking(Hybrid) model is Start")

    
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
    
    # Initialize NLinear model
    logger.info("Training NLinear model")
    set_seed(SEED_NUM)
    nlinear_model = NLinearModel(
        window_size=NLINEAR_PARAMETER["window_size"],
        forecast_size=NLINEAR_PARAMETER["forecast_size"],
        individual=NLINEAR_PARAMETER["individual"],
        feature_size=NLINEAR_PARAMETER["feature_size"],
        logger=logger,
    ).to(device)
    # Train NLinear model
    nlinear_model.train_model(
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        device = device,
        epochs = NLINEAR_PARAMETER['epochs'], 
        lr = NLINEAR_PARAMETER['learning_rate'],
        patience = 30,
        custom_loss_flag = False,
        best_model_path = "./result/best_model__nlinear.pth",
    )


    # Initialize NLinear model(Custom Loss)
    logger.info("Training NLinear model(Custom Loss)")
    set_seed(SEED_NUM)
    nlinear_cl_model = NLinearModel(
        window_size=NLINEAR_PARAMETER["window_size"],
        forecast_size=NLINEAR_PARAMETER["forecast_size"],
        individual=NLINEAR_PARAMETER["individual"],
        feature_size=NLINEAR_PARAMETER["feature_size"],
        logger=logger,
    ).to(device)
    # Train NLinear model
    nlinear_cl_model.train_model(
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        device = device,
        epochs = NLINEAR_PARAMETER['epochs'], 
        lr = NLINEAR_PARAMETER['learning_rate'],
        patience = 30,
        custom_loss_flag = True,
        best_model_path = "./result/best_model__nlinear__custom_loss.pth",
    )


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
    dlinear_model.train_model(
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        device = device,
        epochs = NLINEAR_PARAMETER['epochs'], 
        lr = NLINEAR_PARAMETER['learning_rate'],
        patience = 30,
        custom_loss_flag = False,
        best_model_path = "./result/best_model__dlinear.pth",
    )


    # Initialize DLinear model(Custom Loss)
    logger.info("Training DLinear model(Custom Loss)")
    set_seed(SEED_NUM)
    dlinear_cl_model = DLinearModel(
        window_size=NLINEAR_PARAMETER["window_size"],
        forecast_size=NLINEAR_PARAMETER["forecast_size"],
        individual=NLINEAR_PARAMETER["individual"],
        enc_in=NLINEAR_PARAMETER["feature_size"],
        logger=logger,
    ).to(device)
    # Train NLinear model
    dlinear_cl_model.train_model(
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        device = device,
        epochs = NLINEAR_PARAMETER['epochs'], 
        lr = NLINEAR_PARAMETER['learning_rate'],
        patience = 30,
        custom_loss_flag = True,
        best_model_path = "./result/best_model__dlinear__custom_loss.pth",
    )


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
    dnlinear_model.train_model(
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        device = device,
        epochs = NLINEAR_PARAMETER['epochs'], 
        lr = NLINEAR_PARAMETER['learning_rate'],
        patience = 30,
        custom_loss_flag = False,
        best_model_path = "./result/best_model__dnlinear.pth",
    )
    

    # Initialize DLinear model(Custom Loss)
    logger.info("Training DLinear + NLinear model(Custom Loss)")
    set_seed(SEED_NUM)
    dnlinear_cl_model = DNLinearModel(
        window_size=NLINEAR_PARAMETER["window_size"],
        forecast_size=NLINEAR_PARAMETER["forecast_size"],
        individual=NLINEAR_PARAMETER["individual"],
        enc_in=NLINEAR_PARAMETER["feature_size"],
        kernel_size=NLINEAR_PARAMETER["kernel_size"],
        logger=logger,
    ).to(device)
    # Train NLinear model
    dnlinear_cl_model.train_model(
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        device = device,
        epochs = NLINEAR_PARAMETER['epochs'], 
        lr = NLINEAR_PARAMETER['learning_rate'],
        patience = 30,
        custom_loss_flag = True,
        best_model_path = "./result/best_model__dnlinear__custom_loss.pth",
    )

    
    ## ------------------------------------ CNN_NLinear Training ------------------------------------ ##
    # Optimize Hybrid model or use default parameters
    if HYBRID_PARAMETER["opt_hyperpara"] is True:
        hybrid_params = HYBRID_PARAMETER["space"]
    else:
        hybrid_params = HYBRID_PARAMETER["default_space"]
    for key, values in HYBRID_PARAMETER.items():
        if key not in ["opt_hyperpara", "space", "default_space"]:
            hybrid_params[key] = values

    hybrid_best_params = hybrid_params
    if HYBRID_PARAMETER["opt_hyperpara"] is True:
        logger.info("Optimization of CNN_NLinear model")
        set_seed(SEED_NUM)
        best_params = optimize_cnn_nlinear(hybrid_params, train_loader, val_loader, test_loader, device)

        for key, values in best_params.items():
            hybrid_best_params[key] = values

        logger.info(f"Best hyperparameters for CNN_NLinear model: {hybrid_best_params}")
    
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
    cnn_nlinear_model.train_model(
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        device = device,
        epochs = hybrid_best_params['epochs'], 
        lr = hybrid_best_params['learning_rate'],
        patience = 30,
        custom_loss_flag = False,
        best_model_path = "./result/best_model__cnn_nlinear.pth",
    )


    # Train CNN_NLinear model(Custom Loss)
    logger.info("Training CNN_NLinear model(Custom Loss)")
    set_seed(SEED_NUM)
    cnn_nlinear_cl_model = CNN_NLinear(
        window_size=hybrid_best_params["window_size"],
        forecast_size=hybrid_best_params["forecast_size"],
        conv_kernel_size=hybrid_best_params["conv_kernel_size"],
        conv_filters=hybrid_best_params["conv_filters"],
        in_channels=hybrid_best_params["in_channels"],
        dropout_rate=hybrid_best_params["dropout_rate"],
        logger=logger,
    ).to(device)
    # Train CNN_NLinear model
    cnn_nlinear_cl_model.train_model(
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        device = device,
        epochs = hybrid_best_params['epochs'], 
        lr = hybrid_best_params['learning_rate'],
        patience = 30,
        custom_loss_flag = True,
        best_model_path = "./result/best_model__cnn_nlinear__custom_loss.pth",
    )


    ## ------------------------------------ NLinear + CNN_NLinear Training ------------------------------------ ##
    ## Initialize Hybrid model
    logger.info("Training Hybrid model")
    set_seed(SEED_NUM)
    hybrid_model = HybridModel(
        models=[nlinear_model, dlinear_model, dnlinear_model, cnn_nlinear_model],
        window_size=hybrid_best_params["window_size"],
        dropout_rate=hybrid_best_params["dropout_rate"],
        logger=logger,
    ).to(device)
    # Train Hybrid model
    hybrid_model.train_model(
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        device = device,
        epochs = hybrid_best_params['epochs'], 
        lr = hybrid_best_params['learning_rate'],
        patience = 30,
        custom_loss_flag = False,
        best_model_path = "./result/best_model__hybrid.pth",
    )


    ## Initialize Hybrid model(Custom Loss)
    logger.info("Training Hybrid model(Custom Loss)")
    set_seed(SEED_NUM)
    hybrid_cl_model = HybridModel(
        models=[nlinear_model, dlinear_model, dnlinear_model, cnn_nlinear_model],
        window_size=hybrid_best_params["window_size"],
        dropout_rate=hybrid_best_params["dropout_rate"],
        logger=logger,
    ).to(device)
    # Train Hybrid model
    hybrid_cl_model.train_model(
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        device = device,
        epochs = hybrid_best_params['epochs'], 
        lr = hybrid_best_params['learning_rate'],
        patience = 30,
        custom_loss_flag = True,
        best_model_path = "./result/best_model__hybrid__custom_loss.pth",
    )

    
    ## ------------------------------------ Predict for test set ------------------------------------ ##
    # Evaluate both models
    logger.info("Evaluating models.")
    nlinear_pred_result = nlinear_model.final_predict(test_set, test_loader, device, logger)
    dlinear_pred_result = dlinear_model.final_predict(test_set, test_loader, device, logger)
    dnlinear_pred_result = dnlinear_model.final_predict(test_set, test_loader, device, logger)
    cnn_nlinear_pred_result = cnn_nlinear_model.final_predict(test_set, test_loader, device, logger)
    hybrid_pred_result = hybrid_model.final_predict(test_set, test_loader, device, logger)

    # Evaluate both models(Custom Loss)
    logger.info("Evaluating models(Custom Loss).")
    nlinear_cl_pred_result = nlinear_cl_model.final_predict(test_set, test_loader, device, logger)
    dlinear_cl_pred_result = dlinear_cl_model.final_predict(test_set, test_loader, device, logger)
    dnlinear_cl_pred_result = dnlinear_cl_model.final_predict(test_set, test_loader, device, logger)
    cnn_nlinear_cl_pred_result = cnn_nlinear_cl_model.final_predict(test_set, test_loader, device, logger)
    hybrid_cl_pred_result = hybrid_cl_model.final_predict(test_set, test_loader, device, logger)

    
    # ## ------------------------------------ Visualize Predict Result ------------------------------------ ##
    # # Plot predictions
    # plot_predictions(train_data, val_data, test_data, {"NLinear": nlinear_predictions, "Hybrid": hybrid_predictions})

    
    # ------------------------------------ End of Process ------------------------------------
    END_TIME = datetime.now()
    logger.info('main.py Elapsed time: {!s}'.format(END_TIME - ST_TIME))
    exit(0)
    