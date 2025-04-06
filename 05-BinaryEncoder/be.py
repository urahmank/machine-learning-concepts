import numpy as np
import pandas as pd

from category_encoders import BinaryEncoder

df = pd.read_csv('test.csv')
print(df)

be = BinaryEncoder()

be_df = be.fit_transform(df['Renk'])
be_df = be_df.loc[:, (be_df != 0).any(axis=0)]
print(be_df)

df.drop(['Renk'], axis=1, inplace=True)

dataset = np.append(df.to_numpy(), be_df.to_numpy(), axis=1)
print(dataset)