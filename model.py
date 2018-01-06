import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from sklearn.utils import shuffle

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

def make_train_and_test_set(data_x, data_y, testing_size):
    data_x, data_y = shuffle(data_x, data_y, random_state=0)
    testing_size = int(round(data_x.shape[0] * testing_size))

    print("Training Size is {0}".format(data_x.shape[0] - testing_size))
    print("Testing Size is {0}".format(testing_size))

    train_x = np.matrix(data_x[:-testing_size])
    train_y = np.matrix(data_y[:-testing_size])
    test_x = np.matrix(data_x[-testing_size:])
    test_y = np.matrix(data_y[-testing_size:])
    return train_x, train_y, test_x, test_y