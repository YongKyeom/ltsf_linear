
from data.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

def data_provider(
        embed, 
        train_only,
        scale,
        batch_size,
        freq,
        flag,
        seq_len,
        label_len,
        pred_len,
        features,
        target,
        root_path,
        data_path,
    ):
    Data = Dataset_ETT_hour
    timeenc = 0 if embed != 'timeF' else 1
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True

    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        scale=scale,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last
    )
    
    return data_set, data_loader


def create_dataloaders(
    embed: str = 'timeF', 
    train_only: bool = False,
    scale: bool = True,
    batch_size:int = 32,
    freq: str = 'h',
    data_type_list: list = ['train', 'valid', 'test', 'pred'],
    seq_len: int = 196,
    label_len: int = 96,
    pred_len: int = 96,
    features: str = 'S',
    target: str = 'OT',
    root_path: str = './dataset',
    data_path: str = 'ETTh1.csv',
):
    data_loader_ls = []
    for flag in data_type_list:
        data_loader_ = data_provider(
            embed = embed,
            train_only = train_only,
            scale = scale,
            batch_size = batch_size,
            freq = freq,
            flag = flag,
            seq_len = seq_len,
            label_len = label_len,
            pred_len = pred_len,
            features = features,
            target = target,
            root_path = root_path,
            data_path = data_path
        )
        data_loader_ls.append(data_loader_)
    
    return data_loader_ls
