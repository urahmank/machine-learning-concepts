import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

model = load_model('auto-mpg.hdf5')

with open('auto-mpg-mms.dat', 'rb') as f:
    mms = pickle.load(f)

predict_data = np.loadtxt('predict-data.csv', delimiter=',', dtype=np.float32)
ohe = OneHotEncoder(categories=[[1, 2, 3]], sparse=False)
ohe_data = ohe.fit_transform(predict_data[:, 6].reshape(-1, 1))
predict_data = np.delete(predict_data, 6, axis=1)
predict_data = np.append(predict_data, ohe_data, axis=1)
predict_data = mms.transform(predict_data)
predict_result = model.predict(predict_data)
print(predict_result)
