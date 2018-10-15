import os
#import zipfile
#import argparse
import numpy as np
import _pickle as cp
from scipy import stats
import pandas as pd

#from io import BytesIO
from pandas import Series
from collections import Counter
from sklearn.utils import shuffle
import warnings




OPPORTUNITY_DATA_FILES = ['S1-Drill.dat',
                          'S1-ADL1.dat',
                          'S1-ADL2.dat',
                          'S1-ADL3.dat',
                          'S1-ADL4.dat',
                          'S1-ADL5.dat',
                          'S2-Drill.dat',
                          'S2-ADL1.dat',
                          'S2-ADL2.dat',
                          'S3-Drill.dat',
                          'S3-ADL1.dat',
                          'S3-ADL2.dat',
                          'S2-ADL3.dat',
                          'S3-ADL3.dat',
                          'S2-ADL4.dat',
                          'S2-ADL5.dat',
                          'S3-ADL4.dat',
                          'S3-ADL5.dat'
                          ]

def select_columns_opp(data):

    timestemps = data[:, 0:1]
    accelerometers = data[:, 1:37]
    inertialBACK = data[:, 37:46]
    inertialRVA = data[:, 50:59]
    inertialRLA = data[:, 63:72]
    inertialLUA = data[:, 76:85]
    inertialLLA = data[:, 89:98]
    inertialL_SHOE = data[:, 102:118]
    inertialR_SHOE = data[:, 118:134]
    inertialObjects = data[:, 134:194]
    binarySwitches = data[:, 194:207]

    inertialOtherObjects = data[:, 207:231]

    label_locomotion = data[:, 243]
    label_gestures = data[:, -1]
    # c = Counter(label_gestures)
    # print('counter gestures')
    # print(c)
    # d = Counter(label_locomotion)
    # print('counter locomotion')
    # print(d)

    data_x = np.concatenate([timestemps, accelerometers, inertialBACK, inertialRVA, inertialRLA, inertialLUA, inertialLLA, inertialL_SHOE, inertialR_SHOE, inertialObjects, inertialOtherObjects, binarySwitches], axis=1)
    return data_x, label_locomotion, label_gestures

    # return np.delete(data, features_delete, 1)


def adjust_idx_labels(data_y_loc, data_y_gest):
    data_y_loc[data_y_loc == 4] = 3
    data_y_loc[data_y_loc == 5] = 4

    data_y_gest[data_y_gest == 406516] = 1
    data_y_gest[data_y_gest == 406517] = 2
    data_y_gest[data_y_gest == 404516] = 3
    data_y_gest[data_y_gest == 404517] = 4
    data_y_gest[data_y_gest == 406520] = 5
    data_y_gest[data_y_gest == 404520] = 6
    data_y_gest[data_y_gest == 406505] = 7
    data_y_gest[data_y_gest == 404505] = 8
    data_y_gest[data_y_gest == 406519] = 9
    data_y_gest[data_y_gest == 404519] = 10
    data_y_gest[data_y_gest == 406511] = 11
    data_y_gest[data_y_gest == 404511] = 12
    data_y_gest[data_y_gest == 406508] = 13
    data_y_gest[data_y_gest == 404508] = 14
    data_y_gest[data_y_gest == 408512] = 15
    data_y_gest[data_y_gest == 407521] = 16
    data_y_gest[data_y_gest == 405506] = 17
    return data_y_loc, data_y_gest


def process_dataset_file(data):


    data_x, label_locomotion, label_gestures = select_columns_opp(data)
    print('...file has dim: '+ str(data_x.shape))

    label_locomotion, label_gestures = adjust_idx_labels(label_locomotion, label_gestures)
    label_locomotion = label_locomotion.astype(int)
    label_gestures = label_gestures.astype(int)

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    return data_x, label_locomotion, label_gestures

def features_normalize(dataset):
    for i in range(1, 199):
        mu = np.mean(dataset[:, [i]])
        sigma = np.std(dataset[:, [i]])
        dataset[:, [i]] = (dataset[:, [i]]-mu)/sigma
        if sigma==0:
            print('problem with variance at column: '+str(i))
    return dataset


def strided_axis0(a, L, overlap=1):
    if L==overlap:
        raise Exception("Overlap arg must be smaller than length of windows")
    S = L - overlap
    nd0 = ((len(a)-L)//S)+1
    if nd0*S-S!=len(a)-L:
        warnings.warn("Not all elements were covered")
    m,n = a.shape
    s0,s1 = a.strides
    return np.lib.stride_tricks.as_strided(a, shape=(nd0,L,n), strides=(S*s0,s0,s1))


def segment_signal(accelerometer_data, window_size_seconds, label_loc, label_gest): #give a day stream od accelerometer, returns a list of np.array of windowed signals of length ~window_size_seconds

    window_length = int(30*window_size_seconds)

    label_gest = np.expand_dims(label_gest,axis=1)
    label_loc = np.expand_dims(label_loc,axis=1)


    accelerometer_data = np.concatenate((accelerometer_data, label_gest), axis=1)
    accelerometer_data = np.concatenate((accelerometer_data, label_loc), axis=1)

    window_length = int(30*window_size_seconds)

    segments1 = strided_axis0(accelerometer_data[:, 1:], window_length, int(window_length/2))
    labels_g = np.array([stats.mode(i[:, -2])[0][0] for i in segments1])
    labels_l = np.array([stats.mode(i[:, -1])[0][0] for i in segments1])


    segments1 = segments1[:, :, 0:-2]

    return segments1, labels_l, labels_g
    

def save_data(window_size_seconds):
    directory = 'OpportunityUCIDataset/dataset/'

    data_x = np.empty((0, 211))
    loc_labels = np.empty((0))
    gest_labels = np.empty((0))

    for filename in OPPORTUNITY_DATA_FILES:
        print('Reading file: '+filename)
        data = pd.read_csv(directory+filename, header=None, sep=' ').values

        x, y_loc, y_gest = process_dataset_file(data)
        data_x = np.vstack((data_x, x))
        loc_labels = np.concatenate([loc_labels, y_loc])
        gest_labels = np.concatenate([gest_labels, y_gest])

    print('\nDataSet dimension before segment: '+str(data_x.shape))
    print('\nfeatures_normalize...')
    data_x = features_normalize(data_x)
    print('\nsegment_signal...')
    segments, loc_labels, gest_labels = segment_signal(data_x, window_size_seconds, loc_labels, gest_labels)
    print('\nDataSet dimension after segment: '+str(segments.shape))
 
    # the first 16 file are used for training
    nb_training_samples = int(0.92 * len(segments)) 

    #segments, gest_labels = shuffle(segments, gest_labels)
    X_train, y_train_loc, y_train_gest = segments[:nb_training_samples,:], loc_labels[:nb_training_samples], gest_labels[:nb_training_samples]
    X_test, y_test_loc, y_test_gest  = segments[nb_training_samples:,:], loc_labels[nb_training_samples:], gest_labels[nb_training_samples:]

    save_to = 'OpportunityUCIDataset/dataset/data_saved/'+str(window_size_seconds)+'_seconds'
    print('\nsaving...')
    np.savez(save_to, X_train=X_train, y_train_loc=y_train_loc, y_train_gest=y_train_gest, X_test=X_test, y_test_loc=y_test_loc, y_test_gest=y_test_gest)

    return X_train, y_train_loc, y_train_gest, X_test, y_test_loc, y_test_gest


def load_dataset(filename_path):

    loaded = np.load(filename_path)

    return loaded['X_train'], loaded['y_train_loc'], loaded['y_train_gest'], loaded['X_test'], loaded['y_test_loc'], loaded['y_test_gest']

if __name__ == "__main__":
    exit()



