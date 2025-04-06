import pandas as pd

df = pd.read_csv('test2.csv')
print(df)

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

oe = OrdinalEncoder()

oe_data = oe.fit_transform(df[['Cinsiyet', 'Şehir']])
print(oe_data)

ohe = OneHotEncoder(sparse=False)

ohe_data = ohe.fit_transform(oe_data)
print(ohe_data)

df.drop(['Cinsiyet', 'Şehir'], axis=1, inplace=True)
df[['Cinsiyet-1', 'Cinsiyet-2', 'Şehir-1', 'Şehir-2', 'Şehir-3', 'Şehir-4']] = ohe_data
print(df)
dataset = df.to_numpy()
print(dataset)
