import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

dataset = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

training_dataset_x, test_dataset_x,  training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.1)

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=8, name='Hidden-1'))
model.add(Dense(100, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=90, validation_split=0.20)

result = model.evaluate(test_dataset_x, test_dataset_y)
print(result)


predict_data = np.array([[10, 139, 80, 0, 0, 27.1, 1.441, 57], 
                         [7, 147, 76, 0, 0, 48, 0.257, 43], 
                         [4, 154, 62,31, 284, 32.8, 0.237, 23]])

predicted_results = model.predict(predict_data)

for result in predicted_results[:, 0]:
    if result > 0.5:
        print('Şeker Hastası')
    else:
        print('Şeker Hastası Değil')


"""
10,139,80,0,0,27.1,1.441,57,0
7,147,76,0,0,48.4,0.257,43,1
4,154,62,31,284,32.8,0.237,23,0
"""
