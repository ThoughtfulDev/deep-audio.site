import librosa
import numpy as np
from keras.models import load_model
from sound_util import AudioSample
from sklearn.externals import joblib
import json

model = load_model("./nn_model.h5")
clf = joblib.load('svm_model.pkl')
data = json.load(open('./mappings.json'))


def get_nn_result(filename):
    S = AudioSample(filename)
    X = S.make_mfccs(100)
    X = np.array(X).reshape(-1, 10, 10, 1)
    y = model.predict(X)
    stats = []
    for idx, a in enumerate(y[0]):
        #print("Probability for", data[str(idx)], "=>", str(round( a*100, 4)), "%")
        stats.append(int(round(a*100)))
    index = np.argmax(y)
    return data[str(index)], stats

def get_svm_result(filename):
    S = AudioSample(filename)
    X = S.make_mfccs(100)
    X = X.reshape(1, -1)
    pred = clf.predict(X)
    return data[str(pred[0])]

#This needs to run once when using flask...no idea why
if __name__ == "__main__":
    A = get_nn_result('./warmup.mp3')
    import sys
    if not len(sys.argv) == 2:
        print("Usage: python use.py <filepath>")
        sys.exit(-1)
    else:
        a,b = get_nn_result(sys.argv[1])
        c = get_svm_result(sys.argv[1])
        print("Neural Network")
        print("------------------------")
        print("This sounds like {0}".format(a))
        for idx, s in enumerate(b):
            print("Probability for", data[str(idx)], "=>", s, "%")
        print("------------------------")
        print("SVM")
        print("------------------------")
        print("This sounds like {0}".format(c))
else:
    A = get_nn_result('./warmup.mp3')

