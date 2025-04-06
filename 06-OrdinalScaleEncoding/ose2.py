import pandas as pd

edu_dict = {'İlkokul': 0, 'Ortaokul': 1, 'Lise': 2, 'Üniversite': 3}
df = pd.read_csv('test.csv', converters={3: lambda edu: edu_dict[edu]})
print(df)
