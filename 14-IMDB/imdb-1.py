import numpy as np
import pandas as pd
import re

class WordConverter:
    def __init__(self):
        self.word_set = set()
        
    def __call__(self, review):
        words = re.findall("[a-zA-Z0-9']+", review.lower())
        self.word_set.update(words)
        
        return np.array(words)
        
    def get_word_dict(self):
        return { word: index for index, word in enumerate(self.word_set)}

wc = WordConverter()
df = pd.read_csv('imdb.csv', converters={0: wc})
word_dict = wc.get_word_dict()
rev_word_dict = {index: word for word, index in word_dict.items()}

for i in range(len(df)):
    df.iloc[i, 0] = np.array([word_dict[word] for word in df.iloc[i, 0]])
  
"""
text = ' '.join([rev_word_dict[index] for index in df.iloc[0, 0]])
print(text)
"""

def vectorize(iterable, colsize):
    result = np.zeros((len(iterable), colsize), dtype=np.int8)
    for index, values in enumerate(iterable):
        result[index, values] = 1
        
    return result
        
dataset_x = vectorize(df.iloc[:, 0], len(word_dict))  

df.iloc[df.iloc[:, 1] == 'positive', 1] = 1
df.iloc[df.iloc[:, 1] == 'negative', 1] = 0
dataset_y = df.iloc[:, 1].to_numpy(dtype=np.int8)

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.20)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='IMDB')
model.add(Dense(100, activation='relu', input_dim=len(word_dict), name='Hidden-1'))
model.add(Dense(100, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=64, epochs=5, validation_split=0.20)

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
plt.title('Binary Accuracy - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Binary Accuracy')
plt.plot(range(1, len(hist.history['binary_accuracy']) + 1), hist.history['binary_accuracy'])
plt.plot(range(1, len(hist.history['val_binary_accuracy']) + 1), hist.history['val_binary_accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()

eval_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]} --> {eval_result[i]}')
 
predict_text = 'the movie was not good. There are many flaws. Players are act badly'
predict_words = re.findall("[a-zA-Z0-9']+", predict_text.lower())
predict_numbers = [word_dict[pw] for pw in predict_words]
predict_data = vectorize([predict_numbers], len(word_dict))
predict_result = model.predict(predict_data)

if predict_result[0][0] > 0.5:
    print('OLUMLU')
else:
    print('OLUMSUZ')


