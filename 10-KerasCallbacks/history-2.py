import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

EPOCHS = 90

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
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=EPOCHS, validation_split=0.20)

import matplotlib.pyplot as plt

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Validation Binary Accuracy - Eopch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Binary Accuracy')
plt.xticks(range(0, EPOCHS + 1, 10))
plt.plot(range(1, EPOCHS + 1), hist.history['binary_accuracy'])
plt.plot(range(1, EPOCHS + 1), hist.history['val_binary_accuracy'])
plt.legend(['Training Binary Accuracy', 'Validation Binary Accury'])
plt.show()