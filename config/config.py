import os
from hyperopt import hp

## CPU Multiprocess
CORE_CNT = min(os.cpu_count(), 8)
## 날짜컬럼 이름
DATE_COL_NM = "DATE_TIME"
## Y컬럼 이름
TARGET_COL_NM = "OT"

## NLinear 모델 Parameter
NLINEAR_PARAMETER = {
    "opt_hyperpara": True,
    "space": {
        "feature_size": hp.uniform("window_size", 64, 128),
    },
    "default_space": {
        "feature_size": 48,
    },
    "window_size": 96,
    "forecast_size": 96,
    "individual": False,
    "learning_rate": 0.0001,
    "epochs": 100,
    "batch_size": 32,
    "input_size": 30,
    "output_size": 15,
}
## Hybrid 모델 Parameter
HYBRID_PARAMETER = {
    "opt_hyperpara": False,
    "space": {
        "conv_kernel_size": hp.uniform("conv_kernel_size", 5, 50),
        "conv_filters": hp.uniform("conv_filters", 2, 64),
        "dropout_rate": hp.uniform("dropout_rate", 0, 0.5),
        "hidden_dim": hp.choice("hidden_dim", [5, 10, 15, 20, 30, 50]),
    },
    "default_space": {
        "conv_kernel_size": 21,
        "conv_filters": 16,
        "dropout_rate": 0.1,
        "hidden_dim": 128,
    },
    "window_size": 96,
    "forecast_size": 96,
    "individual": False,
    "learning_rate": 0.001,
    "in_channels": 1,
    "epochs": 100,
    "batch_size": 32,
    "input_size": 30,
    "output_size": 15,
}
