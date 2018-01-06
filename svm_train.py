import numpy as np
from model import make_train_and_test_set
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

X = np.load("./data/matrices/data.dat")
y = np.load("./data/matrices/labels.dat")

y_new = []
for p in y:
    y_new.append([np.argmax(p)])
y = np.matrix(y_new)


train_x, train_y, test_x, test_y = make_train_and_test_set(X, y, testing_size=0.15)
train_y = np.array(train_y).flatten()
test_y = np.array(test_y).flatten()

clf = svm.SVC(kernel="poly", degree=3)
clf.fit(train_x, train_y)


score_pred = clf.predict(test_x)
score = accuracy_score(test_y,score_pred )
print("Score: {0}".format(score))

print("Dumping...")
joblib.dump(clf, 'svm_model.pkl')