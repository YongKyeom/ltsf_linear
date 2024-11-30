import os
from hyperopt import hp

## CPU Multiprocess
CORE_CNT = min(os.cpu_count(), 8)
## 날짜컬럼 이름
DATE_COL_NM = "date"
## Y컬럼 이름
TARGET_COL_NM = "OT"
## SEED Number
SEED_NUM = 1000

## NLinear 모델 Parameter
NLINEAR_PARAMETER = {
    "feature_size": 1,
    "window_size": 336,
    "forecast_size": 96,
    "individual": False,
    "learning_rate": 0.005,
    "kernel_size": 25,
    "epochs": 500,
    "batch_size": 32,
}
## Hybrid 모델 Parameter
HYBRID_PARAMETER = {
    "opt_hyperpara": False,
    "space": {
        "conv_kernel_size": hp.quniform("conv_kernel_size", 16, 50, 2),
        "conv_filters": hp.quniform("conv_filters", 15, 60, 3),
    },
    "default_space": {
        "conv_kernel_size": 20,
        "conv_filters": 48,
    },
    "window_size": 336,
    "forecast_size": 96,
    "individual": False,
    "in_channels": 1,
    "learning_rate": 0.005,
    "epochs": 500,
    "batch_size": 32,
    "dropout_rate": 0.3,
}
