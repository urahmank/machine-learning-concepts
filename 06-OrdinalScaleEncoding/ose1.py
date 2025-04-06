import numpy as np
import pandas as pd

df = pd.read_csv('test.csv')
print(df)

edu_dict = {'Eğitim': {'İlkokul': 0, 'Ortaokul': 1, 'Lise': 2, 'Üniversite': 3}}
df.replace(edu_dict, inplace=True)
print(df)



