from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils

def get_model(X, y):
    num_labels = y.shape[1]
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',input_shape=X.shape[1:], activation="relu"))
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=X.shape[1:], activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',input_shape=X.shape[1:], activation="relu"))
    model.add(Conv2D(64, (3, 3), padding='same',input_shape=X.shape[1:], activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation="softmax"))
    return model