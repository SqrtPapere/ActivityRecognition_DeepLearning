import os
import zipfile
import argparse
import numpy as np
import _pickle as cp
from scipy import stats

from io import BytesIO
from pandas import Series
from collections import Counter
from sklearn.utils import shuffle
import multiprocessing



OPPORTUNITY_DATA_FILES = ['S1-Drill.dat',
                          'S1-ADL1.dat',
                          'S1-ADL2.dat',
                          'S1-ADL3.dat',
                          'S1-ADL4.dat',
                          'S1-ADL5.dat',
                          'S2-Drill.dat',
                          'S2-ADL1.dat',
                          'S2-ADL2.dat',
                          'S2-ADL3.dat',
                          'S3-Drill.dat',
                          'S3-ADL1.dat',
                          'S3-ADL2.dat',
                          'S3-ADL3.dat',
                          'S2-ADL4.dat',
                          'S2-ADL5.dat',
                          'S3-ADL4.dat',
                          'S3-ADL5.dat'
                          ]
NORM_MAX_THRESHOLDS = [3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                       10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                       200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                       10000,  10000,  10000,  10000,  250, ]

NORM_MIN_THRESHOLDS = [-3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                       -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                       -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                       -10000, -10000, -10000, -10000, -250, ]


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
    c = Counter(label_gestures)
    print('counter gestures')
    print(c)
    d = Counter(label_locomotion)
    print('counter locomotion')
    print(d)

    data_x = np.concatenate([timestemps, accelerometers, inertialBACK, inertialRVA, inertialRLA, inertialLUA, inertialLLA, inertialL_SHOE, inertialR_SHOE, inertialObjects, inertialOtherObjects, binarySwitches], axis=1)
    return data_x, label_locomotion, label_gestures

    # return np.delete(data, features_delete, 1)


def adjust_idx_labels(data_y_loc, data_y_gest):
    print(data_y_gest)
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
    print('processing file of dim: '+ str(data_x.shape))

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
        print(i)
        sigma = np.std(dataset[:, [i]])
        dataset[:, [i]] = (dataset[:, [i]]-mu)/sigma
    return dataset

def windows(data_length, size): #returns a list of index [start, end] that can be iterated
    start = 0 #lo start del prossimo turno è l'end di questo
    while start < data_length-size:
        yield int(start), int(start + size)
        start += (size / 2)
        #start += size  # per ora facciamolo funzionare senza sovrapposizione


def segment_signal(accelerometer_data, window_size_seconds, label_loc, label_gest, out_seg): #give a day stream od accelerometer, returns a list of np.array of windowed signals of length ~window_size_seconds
    #window_length = int(calculate_fc(accelerometer_data)*window_size_seconds)
    window_length = int(30*window_size_seconds)

    segments = np.empty((0,window_length,210))
    labels_l = np.empty((0))
    labels_g = np.empty((0))
    for start in range(0, accelerometer_data.shape[0]- int(window_length/2), int(window_length/2)):
        windowed = accelerometer_data[start:start+window_length, 1:211]
        if windowed.shape[0] == window_length:
            l_l = [stats.mode(label_loc[start:start+window_length])[0][0]]
            labels_l = np.append(labels_l,l_l)
            l_g = [stats.mode(label_gest[start:start+window_length])[0][0]]
            labels_g = np.append(labels_g,l_g)
            print(windowed.shape)
            print('building segments: '+str(segments.shape))
            segments = np.concatenate((segments, windowed[None]),axis=0)
    out_seg = np.concatenate((out_seg, segments),axis=0)
    #return segments, labels_l, labels_g
    

def save_data():
    directory = '/home/francesco/Opportunity/OpportunityUCIDataset/dataset/'

    data_x = np.empty((0, 211))
    loc_labels = np.empty((0))
    gest_labels = np.empty((0))

    for filename in OPPORTUNITY_DATA_FILES:
        data = np.loadtxt(directory+filename)

        x, y_loc, y_gest = process_dataset_file(data)
        data_x = np.vstack((data_x, x))
        loc_labels = np.concatenate([loc_labels, y_loc])
        gest_labels = np.concatenate([gest_labels, y_gest])

    data_x = features_normalize(data_x)
    # Dataset is segmented into train and test
    threads = 4
    jobs = []
    segments = np.empty((0,150,210))
    for i in range(0, threads):
        start = i*int(len(data_x)/threads)
        if i==threads-1:
            end = len(data_x)
        else:
            end = start + int(len(data_x)/threads)
        out_list = list()
        thread = multiprocessing.Process(target=segment_signal, args=(data_x[start:end, :], 5, loc_labels, gest_labels, segments))
        jobs.append(thread)

  # Start the threads (i.e. calculate the random number lists)
    for j in jobs:
        j.start()

  # Ensure all of the threads have finished
    for j in jobs:
        j.join()

    #segments, loc_labels, gest_labels = segment_signal(data_x, 5, loc_labels, gest_labels)
    print('after segments')
    print(segments.shape)
    print(loc_labels.shape)
    print(gest_labels.shape)
 
    nb_training_samples = int(0.89 * len(segments))

    #segments, gest_labels = shuffle(segments, gest_labels)
    # The first 18 OPPORTUNITY data files define the traning dataset, comprising 557963 samples
    X_train, y_train_loc, y_train_gest = segments[:nb_training_samples,:], loc_labels[:nb_training_samples], gest_labels[:nb_training_samples]
    X_test, y_test_loc, y_test_gest  = segments[nb_training_samples:,:], loc_labels[nb_training_samples:], gest_labels[nb_training_samples:]

    obj = [(X_train, y_train_loc, y_train_gest), (X_test, y_test_loc, y_test_gest)]
    save_to = "OpportunityUCIDataset/dataset/data_saved/oppChallenge_5seconds_mt.data"
    f = open(os.path.join(save_to), 'wb')
    cp.dump(obj, f, protocol=-1)
    f.close()

    return X_train, y_train_loc, y_train_gest, X_test, y_test_loc, y_test_gest


def load_dataset(filename):

    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train_loc, y_train_gest = data[0]
    X_test, y_test_loc, y_test_gest = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train_loc = y_train_loc.astype(np.uint8)
    y_train_gest = y_train_gest.astype(np.uint8)

    return X_train, y_train_loc, y_train_gest, X_test, y_test_loc, y_test_gest

if __name__ == "__main__":
    exit()
    # from collections import Counter
    # c = Counter(y)
    # print(c)


