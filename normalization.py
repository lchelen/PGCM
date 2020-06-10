import numpy as np

class StandardScaler:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MinMaxScaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max
    def transform(self, data):
        return (data - self.min) / (self.max - self.min)
    def inverse_transform(self, data):
        return (data * (self.max - self.min) + self.min)


def one_hot_by_column(data):
    len = data.shape[0]
    for i in range(data.shape[1]):
        column = data[:, i]
        max = column.max()
        min = column.min()
        zero_matrix = np.zeros((len, max-min+1))
        zero_matrix[np.arange(len), column-min] = 1
        if i == 0:
            encoded = zero_matrix
        else:
            encoded = np.hstack((encoded, zero_matrix))
    return encoded


def minmax_by_column(data):
    for i in range(data.shape[1]):
        column = data[:, i]
        max = column.max()
        min = column.min()
        column = (column - min) / (max - min)
        column = column[:, np.newaxis]
        if i == 0:
            _normalized = column
        else:
            _normalized = np.hstack((_normalized, column))
    return _normalized

