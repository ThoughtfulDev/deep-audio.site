import numpy as np
import librosa
from pathlib import Path
from sound_util import AudioSample

def process(path, features, labels):
    #add label
    if "taylor" in path:
        label = [1, 0, 0, 0]
    elif "michael" in path:
        label = [0, 1, 0, 0]
    elif "ed" in path:
        label = [0, 0, 1, 0]
    elif "ariana" in path:
        label = [0, 0, 0, 1]
    labels.append(label)

    #change audio
    print("Loading Audio...")
    S = AudioSample(path)
    print("Calc mfccs")
    features.append(S.make_mfccs(100))

    for i in range(3):
        print("Applying changes - Iteration {0}/3".format(i+1))
        S.speed_and_pitch()
        S.change_pitch()
        S.change_speed()
        S.change_dynamic_range()
        S.add_noise()
        S.time_shift()
        features.append(S.make_mfccs(100))
        labels.append(label)
        print("Done with Iteration {0}/3".format(i+1))
        S.reset()
        print("-------------------------------------------------")
    

    

PATHS = ['./data/michael_jackson', './data/taylor_swift', './data/ed_sheeran', './data/ariana_grande']

features = []
labels = []

for p in PATHS:
    pathlist = Path(p).glob('**/*.mp3')
    for path in pathlist:
        path_in_str = str(path)
        filename = path_in_str.split('/')[len(path_in_str.split('/')) - 1]
        print("Loading {0}".format(filename))
        process(path_in_str, features, labels)
        print("=============================================")

features = np.matrix(features)
labels = np.matrix(labels)

features.dump("./data/matrices/data.dat")
labels.dump("./data/matrices/labels.dat")