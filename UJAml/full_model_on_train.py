import numpy as np
import matplotlib.pyplot as plt
import os

import dataset_short_window #script for dataset management
import testset_short_window

from keras.utils import to_categorical
#from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout, BatchNormalization, concatenate, LSTM, Activation
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.models import load_model
from keras import backend as K

from keras.utils import plot_model
from collections import Counter
from sklearn.model_selection import KFold
from scipy import stats

import re

from sklearn.metrics import f1_score

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from pathlib import Path


def plotting(history, name):
    """Saves accuracy and loss plots using the output of model.fit.

    Args:
        history: output of model.fit()

    """
    plt.style.use("ggplot")
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name+"_accuracy.png")
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name+"_loss.png")


def translate_for_categorical(vector):

    label_renamer = dict()
    unique_lab = np.unique(vector)

    for i, k in enumerate(unique_lab):
        label_renamer[str(k)]=str(i)

    print('\nConversion of labels using map: ')
    print(label_renamer)
    return [float(label_renamer[str(elem)]) for elem in vector]

def data_and_split(directory, window, already_windowed, not_already_labelled):
    # First boolean argument to False if there  isn't an already segmented version with dim window_seconds
    # Second boolean argument to False if you already labelled the data
    data, label, sensors = dataset_short_window.get_dataset(directory, window, already_windowed, not_already_labelled)
    print('\nUnique labels in Dataset')
    print(np.unique(label))
    print('\nDataSet counter: ')
    c = Counter(label)
    print(c)

    label = translate_for_categorical(label)

    print('\nNew unique labels in Dataset')
    print(np.unique(label))
    classes = len(np.unique(label))
    print('\nTraining to detect '+str(classes)+'/25 classes')

    accelerometer = np.array([x[0] for x in data], dtype="float")
    sensor = np.array([x[1] for x in data], dtype="float")
    times = np.array([[np.sin(2*np.pi*x[2]/24), np.cos(2*np.pi*x[2]/24)] for x in data], dtype="float")

    Y = to_categorical(label, num_classes=classes)

    return classes, [accelerometer, sensor, times], Y

def create_model(window, classes):
    sensors=28
    accel_fc = 50
    sensorShape = (window, sensors)
    accelerShape = (window*accel_fc, 3)
    timeShape = (2,)

    accelerometer_input = Input(shape=accelerShape, dtype='float', name='accelerometer_input')
    sensor_input = Input(shape=sensorShape, dtype='float', name='sensor_input')
    time_input = Input(shape=timeShape, dtype='float', name='time_input')

    time_out = (time_input)

    x = LSTM(60, return_sequences=True)(sensor_input)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    sensor_out = (x)

    y = Conv1D(18, 2, padding="same", activation='relu')(accelerometer_input)
    y = MaxPooling1D(pool_size=2, strides=2)(y)
    y = Conv1D(36, 2, padding="valid", activation='relu')(y)
    y = MaxPooling1D(pool_size=2, strides=2)(y)
    y = Conv1D(56, 2, activation='relu')(y)
    y = MaxPooling1D(pool_size=2, strides=2)(y)
    y = Dropout(0.3)(y)
    accel_out = Flatten()(y)

    z = concatenate([accel_out, sensor_out, time_out])
    z = BatchNormalization()(z)

    z = Dense(64, activation='relu')(z)
    z = Dense(64, activation='relu')(z)

    main_output = Dense(classes, activation='softmax', name='main_output')(z)

    model = Model(inputs=[accelerometer_input, sensor_input, time_input], outputs=[main_output])
    optimizer = optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #plot_model(model, show_shapes=True, to_file='model.png')
    #print(model.summary()) 
    return model

use_saved_model = True
window = 3

test_directory = 'Data/Test/2017-11-21/2017-11-21-C'

if use_saved_model:
    model = load_model('best_model_all_train.h5')
else:
    # remember to change header in 2017-11-20-C-sensors.csv DATE->TIMESTAMP
    directory = 'Data/Training/'

    accel_fc = 50

    print('\nWindowing at '+str(window)+' seconds')

    classes, data, Y = data_and_split(directory, window, False, False)
    train_accel, train_sensor, train_times = data[0], data[1], data[2]

    print('\nUnique labels in TrainSet')
    c = Counter(np.argmax(Y, axis=1))
    print(c)

    print('\nBuild Model...')
    model = create_model(window, classes)

    print('\nTraining...')
    history = model.fit([train_accel, train_sensor, train_times], Y, epochs=25, batch_size=32, verbose=0)

    print('\nSaving model')
    model.save('best_model_all_train.h5')

# grab data from testset
data, sensors = testset_short_window.get_dataset(test_directory, window)

test_accel = np.array([x[0] for x in data], dtype="float")
test_sensor = np.array([x[1] for x in data], dtype="float")
test_times = np.array([[np.sin(2*np.pi*x[2]/24), np.cos(2*np.pi*x[2]/24)] for x in data], dtype="float")

# predict data from test
y_pred = model.predict([test_accel, test_sensor, test_times])
y_pred = np.argmax(y_pred,axis=1)

# grouping each 30 seconds -> 10 predictions
grouped_pred = []
for i in range(0, y_pred.shape[0], 10):
    portion = y_pred[i:i+10]
    l = [stats.mode(portion)[0][0]]
    grouped_pred.extend(l)

print(len(grouped_pred))
print(grouped_pred)


convert_dict = {'0':'idle', '1':'act01', '2':'act02', '3':'act03',\
    '4':'act04', '5':'act05', '6':'act06', '7':'act07', '8':'act08',\
    '9':'act09', '10':'act10', '11':'act11', '12':'act12', '13':'act13',\
    '14':'act14', '15':'act15', '16':'act16', '17':'act17', '18':'act18',\
    '19':'act19', '20':'act20', '21':'act21', '22':'act22', '23':'act23',\
    '24':'act24' }
for el in grouped_pred:

    print(convert_dict[str(el)])

# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# scores_acc.append(scores[1]*100)


# print('\nF1-SCORE BOARD: ')
# print('macro: '+str(f1_score(testY_true, y_pred, average='macro')))
# print('micro: '+str(f1_score(testY_true, y_pred, average='micro')))
# print('weighted: '+str(f1_score(testY_true, y_pred, average='weighted')))
# print('None: ')
# print(f1_score(testY_true, y_pred, average=None))

# print("%.2f%% (+/- %.2f%%)" % (np.mean(scores_acc), np.std(scores_acc)))












