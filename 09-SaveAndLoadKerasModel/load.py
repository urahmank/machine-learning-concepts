# keras-model-load.py

import numpy as np
from tensorflow.keras.models import load_model

model = load_model('diabetes.h5')

predict_x = np.array([[5.0, 73.0, 60.0, 0., 0., 29.8, 0.368, 25.0],
                      [6.0, 43.0, 70.0, 0., 0., 29.8, 0.368, 25.0],
                      [6.0, 73.0, 70.0, 0., 0., 39.8, 0.668, 25.0]])

predict_result = model.predict(predict_x)

for i in range(len(predict_result)):
    if predict_esult[i, 0] < 0.5:
        print('Şeker hastası değil')
    else:
        print('Şeker hastası')
