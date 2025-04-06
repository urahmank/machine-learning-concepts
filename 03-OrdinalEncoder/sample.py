import pandas as pd

df = pd.read_csv('test.csv')
print(df)

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()
df[['Cinsiyet', 'Şehir']] = oe.fit_transform(df[['Cinsiyet', 'Şehir']])
print(df)

