import pandas as pd
import torch
import os

from darts.datasets import ETTh1Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


def load_data(file_path: str = "ETTh1", date_col_nm: str = "date", target_col_nm: str = "OT") -> pd.DataFrame:
    """
    시계열 모델 학습 및 검증을 위한 Dataset을 Load하는 함수
    file_path Dataset 이름을 입력을 받은 경우, 해당 Dataset을 리턴함

    Args:
        file_path (str): Raw data 경로 혹은 Dataset 이름
        date_col_nm (str): 날짜 컬럼 이름. 해당 컬럼이 Dataset에 포함되는 경우, 날짜 컬럼을 Index로 변환함
        target_col_nm (str): Y컬럼 이름

    Returns:
        pd.DataFrame: Time series dataset
    """
    ## Load dataset
    if file_path == "ETTh1":
        try:
            ## Read from local csv
            raw_df = pd.read_csv('.data/ETTh1.csv')
        except:
            ## Load from darts.datasets
            raw_df = ETTh1Dataset().load().pd_dataframe()
            ## Add Date column
            raw_df.insert(0, date_col_nm, raw_df.index)
            ## To save to local
            save_path = './dataset'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            raw_df.to_csv(f'{save_path}/ETTh1.csv', index = False)
    else:
        raw_df = pd.read_csv(file_path)

    ## 날짜컬럼 Index 지정
    if date_col_nm in raw_df.columns:
        raw_df.set_index(keys=date_col_nm, inplace = True)

    print(f"Load dataset(cnt: {raw_df.shape[0]})")

    return raw_df[[target_col_nm]]


def split_data(
    data: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        data (pd.DataFrame): Full dataset.
        test_size (float): Proportion of data to be used for testing.
        val_size (float): Proportion of data to be used for validation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test data.
    """
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), shuffle=False)

    return train_data, val_data, test_data


class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, window_size: int = 96, forecast_size: int = 96):
        """
        TimeSeriesDataset for creating input-output pairs from time series data.

        Args:
            data (pd.DataFrame or pd.Series): 시계열 데이터 (index가 DATE_COL_NM, 값이 TARGET_COL_NM으로 구성된 단변량 데이터)
            window_size (int): 모델에 입력할 시퀀스 길이.
            forecast_size (int): 모델이 예측할 시퀀스 길이.
        """
        self.data = data.values
        self.window_size = window_size
        self.forecast_size = forecast_size

    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_size + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size : idx + self.window_size + self.forecast_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def inverse_transform(self, data, scaler):
        return scaler.inverse_transform(data)


def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    window_size: int = 96,
    forecast_size: int = 96,
    batch_size: int = 32,
    num_workers=0,
):
    """
    Create DataLoader objects for training, validation, and test data.

    Args:
        train_data (pd.DataFrame or pd.Series): 훈련 시계열 데이터.
        val_data (pd.DataFrame or pd.Series): 검증 시계열 데이터.
        test_data (pd.DataFrame or pd.Series): 테스트 시계열 데이터.
        window_size (int): 입력 시퀀스 길이.
        forecast_size (int): 예측 시퀀스 길이.
        batch_size (int): DataLoader의 배치 크기.
        num_workers (int): DataLoader의 워커 수 (멀티프로세싱 사용을 위해).

    Returns:
        tuple: Train / Valididation / Test DataLoader (train_loader, val_loader, test_loader)
    """
    train_dataset = TimeSeriesDataset(train_data, window_size, forecast_size)
    val_dataset = TimeSeriesDataset(
        pd.concat(
            [
                train_data[:-window_size], 
                val_data
            ], 
            axis=0, 
            ignore_index=True
        ), 
        window_size, forecast_size
    )
    test_dataset = TimeSeriesDataset(test_data, window_size, forecast_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
