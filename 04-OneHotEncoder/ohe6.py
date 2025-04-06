import numpy as np

def one_hot_encoder(dataset, column):
    ncategory = np.max(dataset[:, column]) + 1
    ohe = np.zeros((dataset.shape[0], int(ncategory)), dtype=np.float32)
    for index, category in enumerate(dataset[:, column]):
        ohe[index, int(category)] = 1

    dataset = np.delete(dataset, column, axis=1)
    dataset = np.hstack((dataset, ohe))

    return dataset

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('test2.csv')

oe = OrdinalEncoder()
df[['Cinsiyet', 'Şehir']] = oe.fit_transform(df[['Cinsiyet', 'Şehir']])
dataset = df.to_numpy()

dataset = one_hot_encoder(dataset, 0)
dataset = one_hot_encoder(dataset, 2)

print(dataset)


