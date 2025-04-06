import numpy as np

dataset = np.loadtxt('test.csv', delimiter=',', skiprows=1)

from sklearn.preprocessing import MaxAbsScaler

mas = MaxAbsScaler()
mas.fit(dataset)
scaled_data = mas.transform(dataset)
print(scaled_data)
print()

scaled_data = mas.fit_transform(dataset)
print(scaled_data)