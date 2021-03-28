import os
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model

image_dir = 'C:/Users/hp/PycharmProjects/classificationmodel/presc_dataset'
size = 128, 128
batch_size = 70
epochs = 50


####load images files#####
def listImagesFiles(image_dir):
    listOfFile = os.listdir(image_dir)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(image_dir, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + listImagesFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


####loading image pixels from image file#####
def load_image(file):
    try:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return img
    except:
        return None


####load the dataset########
img_data = []
lbl_data = []


def overview(total_rows):
    fig = plt.figure(figsize=(10, 10))
    idx = 0
    while idx < total_rows:
        ax = fig.add_subplot(10, 10, idx + 1)
        ax.imshow(img_data[idx], cmap=plt.cm.get.cmap('gray'))
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
        idx += 1

        plt.show()


def load_data():
    images_ = listImagesFiles(image_dir)
    print(images_)
    for file in images_:
        img = load_image(file)
        if img is not None:
            img_data.append(img)
            lbl_data.append(os.path.basename(os.path.dirname(file)))
        else:
            continue
            overview(50)
    return img_data, lbl_data


######getting number of images classes######
def getClassesNum():
    classes = set(lbl_data)
    # return classes._len_()
    return len(classes)

#######encoding images classes labels#######
def categorizeLabels(lbl_data):
    from sklearn.preprocessing import LabelEncoder
    return LabelEncoder().fit_transform(lbl_data)


######loading the dataset######
def split_dataset():
    from sklearn.model_selection import train_test_split
    img_data, lbl_data = load_data()
    img_data = np.asarray(img_data)

    img_data = img_data / 255.0

    lbl_data = categorizeLabels(lbl_data)
    lbl_data = np.asarray(lbl_data)
    lbl_data = to_categorical(lbl_data, num_classes=getClassesNum())
    x_train, x_test, y_train, y_test = train_test_split(img_data, lbl_data, test_size=0.20, random_state=10)
    return x_train, x_test, y_train, y_test


#######CNN Model construction######
def construct_cnn():
    model = Sequential()
    model.add(Conv2D(12, (2, 2), input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(36, (2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(getClassesNum(), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


######Intilaizing tarining callbacks#####
def init_callbacks():
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    base_path = 'C:/Users/hp/PycharmProjects/classificationmodel/model_weights/'
    # model_weights.{epoch:04d}---{val_Loss:.4f}---{val_accuracy:.4f}.h5
    trained_models_path = base_path + 'model_weights'
    model_names = trained_models_path + '.{epoch:04d}---{val_Loss:.4f}---{val_accuracy:.4f}.h5'
    model_checkpoint =tf.keras.callbacks.ModelCheckpoint(r"C:/Users/hp/PycharmProjects/classificationmodel/model_weights/shaza.h5", monitor='val_accuracy', verbose=1, save_best_only=True)
    callbacks = [model_checkpoint]
    return callbacks


###### Train the model########

def train_model():
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rotation_range=2, zoom_range=0.1, horizontal_flip=True)

    x_train, x_test, y_train, y_test = split_dataset()
    model = construct_cnn()

    model.fit_generator(train_datagen.flow(x_train, y_train, batch_size),
                        steps_per_epoch=len(x_train) / batch_size,
                       epochs=epochs, verbose=1,callbacks= init_callbacks(),
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

train_model()



