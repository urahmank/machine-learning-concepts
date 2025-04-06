import pandas as pd

df = pd.read_csv('test.csv')
print(df)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Şehir'] = le.fit_transform(df['Şehir'])
print(df)

