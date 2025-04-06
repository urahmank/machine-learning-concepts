import numpy as np

dataset = np.loadtxt('test.csv', delimiter=',', skiprows=1)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)
scaled_data = ss.transform(dataset)
print(scaled_data)

print()

scaled_data = ss.fit_transform(dataset)
print(scaled_data)

def standard_scaler(dataset):
    scaled_dataset = np.zeros_like(dataset)
    for col in range(dataset.shape[1]):
        mu = np.mean(dataset[:, col])
        std = np.std(dataset[:, col])
        scaled_dataset[:, col] = (dataset[:, col] - mu) / std
    
    return scaled_dataset
        
scaled_data = standard_scaler(dataset)
print(scaled_data)
    