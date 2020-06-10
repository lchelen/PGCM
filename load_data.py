import numpy as np
import os
import torch
import torch.utils.data
from normalization import MinMaxScaler, StandardScaler

base_dir = os.path.dirname(os.path.abspath(__file__))

def Load_Sydney_Demand_Data(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data

def Add_Window_Horizon(data, window=3, horizon=1):

    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    while index < end_index:
        X.append(data[index:index+window])
        Y.append(data[index+window:index+window+horizon])
        index = index + 1
    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y

def split_train_val_test(data, val_days, test_days, interval=60):

    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, mode='train'):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X = TensorFloat(X)
    Y = TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    if mode == 'train':
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


def get_dataloader(dataset, batch_size=128, window=12, horizon=1,
                   val_days=10, test_days=10, normalizer = 'max'):
    if dataset == 'SYDNEY':
        data = Load_Sydney_Demand_Data(os.path.join(base_dir, '1h_data_new3.csv'))
        print(data.shape)
        print('Load Sydney Dataset Successfully!')

    if normalizer == 'max':
        scaler = MinMaxScaler(data.min(), data.max())
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax Normalization')
    elif normalizer == 'std':
        scaler = StandardScaler(data.mean(), data.std())
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    else:
        scaler = None

    X, Y = Add_Window_Horizon(data, window, horizon)
    print(X.shape, Y.shape)

    x_tra, x_val, x_test = split_train_val_test(X, val_days, test_days)
    y_tra, y_val, y_test = split_train_val_test(Y, val_days, test_days)
    print(x_tra.shape, y_tra.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    train_dataloader = data_loader(x_tra, y_tra, batch_size, 'train')
    val_dataloader = data_loader(x_val, y_val, batch_size, 'val')
    test_dataloader = data_loader(x_test, y_test, batch_size, 'test')
    dataloader = data_loader(X,Y,batch_size,'all')
    return train_dataloader, val_dataloader, test_dataloader, scaler








