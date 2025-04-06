import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

dataset = np.loadtxt('diabetes.csv', skiprows=1, delimiter=',')
dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y,                                                                                  test_size=0.20)

model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu', name='Hidden-1'))
model.add(Dense(100, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=200, validation_split=0.2)
result = model.evaluate(test_dataset_x, test_dataset_y)

for i in range(len(result)):
    print('{} = {}'.format(model.metrics_names[i], result[i]))

model.save('diabetes.h5')
