import numpy as np

dataset = np.loadtxt('test.csv', delimiter=',', skiprows=1)

def minmax_scaler(dataset):
    scaled_dataset = np.zeros_like(dataset)
    for col in range(dataset.shape[1]):
        minval = dataset[:, col].min()
        maxval = dataset[:, col].max()
        scaled_dataset[:, col] = (dataset[:, col] - minval) / (maxval - minval)
        
    return scaled_dataset

scaled_data = minmax_scaler(dataset)
print(scaled_data)
print()

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
mms.fit(dataset)
scaled_data = mms.transform(dataset)
print(scaled_data)

scaled_data = mms.fit_transform(dataset)
print(scaled_data)
