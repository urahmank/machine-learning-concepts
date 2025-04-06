import numpy as np

dataset = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

import pandas as pd

df = pd.DataFrame(dataset_x)
df.hist(figsize=(10, 5))
