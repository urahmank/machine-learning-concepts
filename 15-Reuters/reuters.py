import numpy as np
from tensorflow.keras.datasets import reuters

VOCAB_SIZE = 30000

category_list = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply', 'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas', 'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin', 'strategic metal','livestock','retail','ipi','iron-teel','rubber','heat','jobs','lei','bop','zinc',
'orange','pet-chem','dlr','gas','silver','wpi','hog','lead']

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = reuters.load_data(num_words=VOCAB_SIZE)

word_dict = reuters.get_word_index()
rev_word_dict = {index: word for word, index in word_dict.items()}

"""
def print_review(review):
    text = ' '.join([rev_word_dict[index - 3] for index in review if index > 2])
    print(text)
    
print_review(training_dataset_x[0])
"""

def vectorize(iterable, colsize):
    result = np.zeros((len(iterable), colsize), dtype=np.int8)
    for index, values in enumerate(iterable):
        result[index, values] = 1
        
    return result
  
training_dataset_x = vectorize(training_dataset_x, VOCAB_SIZE)
test_dataset_x = vectorize(test_dataset_x, VOCAB_SIZE)

training_dataset_y = vectorize(training_dataset_y, 46)
test_dataset_y = vectorize(test_dataset_y, 46)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='Reuters')
model.add(Dense(100, activation='relu', input_dim=VOCAB_SIZE, name='Hidden-1'))
model.add(Dense(100, activation='relu', name='Hidden-2'))
model.add(Dense(46, activation='softmax', name='Output'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=1, validation_split=0.20)

"""
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
plt.title('Categorical Accuracy - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Categorical Accuracy')
plt.plot(range(1, len(hist.history['categorical_accuracy']) + 1), hist.history['categorical_accuracy'])
plt.plot(range(1, len(hist.history['val_categorical_accuracy']) + 1), hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Accuracy', 'Validation Categorical Accuracy'])
plt.show()
"""

eval_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]} --> {eval_result[i]}')

predict_text = "qtly div 62 cts vs 58 cts in the prior quarter payable april 30 record march 20 reuter 3"

import re

def prepare_review(text):
    words = re.findall("[a-zA-Z-']+", text)
    words_nums = [word_dict[word] + 3 for word in words]
    
    return vectorize([words_nums], VOCAB_SIZE)

predict_data = prepare_review(predict_text)
predict_result = model.predict(predict_data.reshape(1, -1))
predict_category = np.argmax(predict_result)
print(category_list[predict_category])




