from librosa import load
from librosa.effects import (pitch_shift, time_stretch)
from librosa.feature import mfcc
import numpy as np
from sklearn.utils import shuffle


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

class AudioSample:
    def __init__(self, path, res_type='kaiser_fast'):
        self.X, self.sr = load(path, res_type=res_type)
        self.orig = self.X
        self.length = self.X.shape[0]
    
    def get_sample(self):
        return self.X
    def get_sr(self):
        return self.sr

    def reset(self):
        self.X = self.orig

    def make_mfccs(self, N):
        mfccs = mfcc(y=self.X, sr=self.sr, n_mfcc=N)
        return np.mean(mfccs.T, axis=0)

    def speed_and_pitch(self):
        length_change = np.random.uniform(low=0.9,high=1.1)
        speed_fac = 1.0  / length_change
        tmp = np.interp(np.arange(0,self.length,speed_fac),np.arange(0,self.length),self.X)
        minlen = min( self.orig.shape[0], tmp.shape[0])
        self.X *= 0
        self.X[0:minlen] = tmp[0:minlen]
    
    def change_pitch(self):
        bins_per_octave = 24
        pitch_pm = 4
        pitch_change =  pitch_pm * 2*(np.random.uniform()-0.5)
        self.X = pitch_shift(self.X, self.sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
    
    def change_speed(self):
        speed_change = np.random.uniform(low=0.9,high=1.1)
        tmp = time_stretch(self.X, speed_change)
        minlen = min( self.orig.shape[0], tmp.shape[0])
        self.X *= 0
        self.X[0:minlen] = tmp[0:minlen]
    
    def change_dynamic_range(self):
        dyn_change = np.random.uniform(low=0.5,high=1.1)
        self.X *= dyn_change
    
    def add_noise(self):
        noise_amp = 0.005*np.random.uniform()*np.amax(self.X) 
        self.X += noise_amp * np.random.normal(size=self.length)
    
    def time_shift(self):
        timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)
        start = int(self.length * timeshift_fac)
        if (start > 0):
            self.X = np.pad(self.X,(start,0),mode='constant')[0:self.X.shape[0]]
        else:
            self.X = np.pad(self.X,(0,-start),mode='constant')[0:self.X.shape[0]]


