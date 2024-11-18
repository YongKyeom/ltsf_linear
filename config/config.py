import os
from hyperopt import hp

## CPU Multiprocess
CORE_CNT = min(os.cpu_count(), 8)
## 날짜컬럼 이름
DATE_COL_NM = "date"
## Y컬럼 이름
TARGET_COL_NM = "OT"

## NLinear 모델 Parameter
NLINEAR_PARAMETER = {
    "opt_hyperpara": True,
    "space": {
        "feature_size": hp.uniform("feature_size", 64, 128),
    },
    "default_space": {
        "feature_size": 48,
    },
    "window_size": 336,
    "forecast_size": 96,
    "individual": False,
    "learning_rate": 0.005,
    "epochs": 100,
    "batch_size": 32,
}
## Hybrid 모델 Parameter
HYBRID_PARAMETER = {
    "opt_hyperpara": False,
    "space": {
        "conv_kernel_size": hp.uniform("conv_kernel_size", 5, 50),
        "conv_filters": hp.uniform("conv_filters", 2, 64),
        "dropout_rate": hp.uniform("dropout_rate", 0, 0.3),
        "num_heads": hp.choice("num_heads", [4, 8, 16, 32]),
    },
    "default_space": {
        "conv_kernel_size": 21,
        "conv_filters": 32, # 128,
        "dropout_rate": 0.3,
        "num_heads": 4,
    },
    "window_size": 336,
    "forecast_size": 96,
    "individual": False,
    "learning_rate": 0.005,
    "in_channels": 1,
    "epochs": 100,
    "batch_size": 32,
    "input_size": 30,
    "output_size": 15,
}
