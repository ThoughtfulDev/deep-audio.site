import numpy as np
from model import (make_train_and_test_set, get_model)

X = np.load("./data/matrices/data.dat")
y = np.load("./data/matrices/labels.dat")

train_x, train_y, test_x, test_y = make_train_and_test_set(X, y, testing_size=0.15)
train_x = np.array(train_x).reshape(-1, 10, 10, 1)
test_x = np.array(test_x).reshape(-1, 10, 10, 1)

model = get_model(train_x, train_y)
model.compile(loss='categorical_crossentropy', 
              metrics=['accuracy'], 
              optimizer="rmsprop")



model.fit(train_x, train_y, batch_size=100, epochs=100, validation_data=(test_x, test_y), verbose=1)

(loss, accuracy) = model.evaluate(test_x, test_y, batch_size=100, verbose=0)
print("Accuracy ", accuracy)

model.save("nn_model.h5")