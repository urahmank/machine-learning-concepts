import numpy as np

dataset = np.loadtxt('auto-mpg.data', usecols=range(8), converters= {3: lambda s:  float(s) if s != b'?' else 0}, dtype=np.float32)
dataset = dataset[dataset[:, 3] != 0, :]

dataset_x = dataset[:, 1:]
dataset_y = dataset[:, 0]

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

ohe = OneHotEncoder(sparse=False)
ohe_data = ohe.fit_transform(dataset_x[:, 6].reshape(-1, 1))
dataset_x = np.delete(dataset_x, 6, axis=1)
dataset_x = np.append(dataset_x, ohe_data, axis=1)

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

mms = MinMaxScaler()
mms.fit(dataset_x)

training_dataset_x = mms.transform(training_dataset_x)
test_dataset_x = mms.transform(test_dataset_x)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Auto-MPG')
model.add(Dense(100, activation='relu', input_dim=dataset_x.shape[1], name='Hidden-1'))
model.add(Dense(100, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=200, validation_split=0.2)

import matplotlib.pyplot as plt 

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Loss - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, len(hist.history['loss']) + 1), hist.history['loss'])
plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Mean Absolute Error - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.plot(range(1, len(hist.history['mae']) + 1), hist.history['mae'])
plt.plot(range(1, len(hist.history['val_mae']) + 1), hist.history['val_mae'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()

test_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(test_result)):
    print(f'{model.metrics_names[i]} --> {test_result[i]}')
    
predict_data = np.array([[4., 113., 95., 2228., 14., 71, 0, 0, 1]], dtype='float32')
predict_data = mms.transform(predict_data)
predict_result = model.predict(predict_data)
print(predict_result)
    
predict_data = np.loadtxt('predict-data.csv', delimiter=',', dtype=np.float32)
ohe = OneHotEncoder(categories=[[1, 2, 3]], sparse=False)
ohe_data = ohe.fit_transform(predict_data[:, 6].reshape(-1, 1))
predict_data = np.delete(predict_data, 6, axis=1)
predict_data = np.append(predict_data, ohe_data, axis=1)
predict_data = mms.transform(predict_data)
predict_result = model.predict(predict_data)
print(predict_result)
    
model.save('auto-mpg.hdf5')

import pickle

with open('auto-mpg-mms.dat', 'wb') as f:
   pickle.dump(mms, f)
    

