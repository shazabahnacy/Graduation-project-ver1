import os
import numpy as np
import cv2
from keras.layers import Input, Conv2D, Activation, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential, Model
from keras.utils import to_categorical

classes = {0: 'aspocid', 1: 'ator', 2: 'concor', 3: 'nitromak', 4: 'plavix'}


def construct_cnn():
    model = Sequential()
    model.add(Conv2D(12, (2, 2), input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(36, (2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def load_existing_model_weights():
    model = construct_cnn()
    import os
    from pathlib import Path
    user_home = str(Path.home())
    model.load_weights(r"C:/Users/hp/PycharmProjects/classificationmodel/model_weights/shaza.h5")
    return model


size = 128, 128


def load_image(file):
    try:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return img
    except:
        return None


def classify_presc(file):
    model = load_existing_model_weights()
    img = load_image(file)

    img = np.asarray(img)
    img = img / 255.0
    img = img.reshape(-1, 128, 128, 3)

    return model.predict_classes(img, batch_size=32)


cls = classify_presc(r'C:/Users/hp/PycharmProjects/classificationmodel/nitromak test 2.jpeg')

print(classes[cls[0]])