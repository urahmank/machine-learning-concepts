import pandas as pd

df = pd.read_csv('test-le.csv')
print(df)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['City'] = le.fit_transform(df['City'])
print(df)

