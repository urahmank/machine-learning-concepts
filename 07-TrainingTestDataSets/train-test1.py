import numpy as np

dataset = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)

np.random.shuffle(dataset)

dataset_x = dataset[:, :8]
dataset_y = dataset[:, 8]

tzone = int(len(dataset) * 0.8)

training_dataset_x = dataset_x[:tzone, :]
training_dataset_y = dataset_y[:tzone]

test_dataset_x = dataset_x[tzone:, :]
test_dataset_y = dataset_y[tzone:]

 