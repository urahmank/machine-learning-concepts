import numpy as np
import pandas as pd

df = pd.read_csv('test2.csv')
print(df)

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()

df[['Cinsiyet', 'Şehir']] = oe.fit_transform(df[['Cinsiyet', 'Şehir']])
print(df)

from tensorflow.keras.utils import to_categorical

ohe_gender = to_categorical(df['Cinsiyet'])
ohe_city = to_categorical(df['Şehir'])

df.drop(['Cinsiyet', 'Şehir'], axis=1, inplace=True)

dataset = np.append(df.to_numpy(), ohe_gender, axis=1)
dataset = np.append(dataset, ohe_city, axis=1)
print(dataset)

