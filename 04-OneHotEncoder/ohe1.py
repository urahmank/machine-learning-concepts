import pandas as pd

df = pd.read_csv('test1.csv')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
df['Renk'] = le.fit_transform(df['Renk'])
print(df)

ohe = OneHotEncoder(sparse=False)
ohe_data = ohe.fit_transform(df['Renk'].to_numpy().reshape(-1, 1))
print(ohe_data)

df.drop(['Renk'], axis=1, inplace=True)
print(df)
df[['Renk-0', 'Renk-1', 'Renk-2']] = ohe_data
print(df)
