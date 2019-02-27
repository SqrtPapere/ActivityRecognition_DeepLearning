import numpy as np
#import matplotlib.pyplot as plt
import os
import pandas as pd
import dataset_Adj_plu_prox as dataset_short_window # script for dataset management
import utils_module as utils

#from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout, MaxPooling2D
from keras.layers import BatchNormalization, concatenate, LSTM, Activation, TimeDistributed, Conv2D
from keras.models import Model
from keras import optimizers

from keras.models import Sequential, load_model

from keras import backend as K

from keras.utils import plot_model, to_categorical
from collections import Counter
from sklearn.metrics import confusion_matrix, f1_score
from scipy import stats

import itertools
#import cv2
import re
from sklearn.utils import resample, shuffle

import copy
#from scipy.misc import imsave
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import pickle


#from pathlib import Path

def test_single_module(y_pred, y_true, window):
    y_pred = np.array([i+1 for i in np.argmax(y_pred, axis=1)])
    grouped_pred = []
    matrix_prediction = np.empty((int(30/window), 0))
    for i in range(0, y_pred.shape[0], int(30/window)):
        portion = np.array(y_pred[i:i+int(30/window)])
        pad_amount = int(30/window)-portion.shape[0]
        if pad_amount:
            portion = np.pad(portion, (0,pad_amount), 'edge')
        matrix_prediction = np.column_stack((matrix_prediction, portion))
        l = [stats.mode(portion)[0][0]]
        grouped_pred.extend(l)

    # grouping each 30 seconds
    matrix_prediction = np.transpose(matrix_prediction)

    partial = 0
    partial_tp = 0
    for el, (true1, true2) in zip(grouped_pred, y_true):
        partial+=1
        converted = utils.convert_dict[str(el)]
        #print(converted + ' | '+ true1 + ', '+ true2)
        if converted == str(true1) or converted == str(true2):
            partial_tp +=1

    return partial_tp, partial




def divide_data(list_of_samples, squeezed=False):

    accelerometer = np.array([x[0] for x in list_of_samples], dtype="float")
    sensor = np.array([x[1] for x in list_of_samples], dtype="float")
    times = np.array([x[2] for x in list_of_samples], dtype="float")
    proximity = np.array([x[3] for x in list_of_samples], dtype="float")
    pressure = np.array([x[4] for x in list_of_samples], dtype="float")

    if squeezed is False:
        return [accelerometer, sensor, times, proximity, pressure]
    else: 
        return accelerometer, sensor, times, proximity, pressure


def copy_9_into_12(data):

    new_samples = []

    for sample in data:
        new = sample
        if np.sum(new[1][:, 4, 3]) > 0:
            new[1][:, 4, 3] = 0
            new[0] = utils.DA_Rotation(utils.DA_Permutation(new[0], nPerm=4))

            new_samples.append(new)
    
    print('New 12 samples:')
    print(len(new_samples))
    return np.array(new_samples)


def apply_oversampling(data, label):

    data = np.array(data)

    nine_index = np.array([i for i, x in enumerate(label) if x == 9])
    nine_data = data[nine_index]

    augmented_12 = copy_9_into_12(nine_data)

    print('data before adding augmented12')
    print(data.shape)
    print(label.shape)
    data = np.concatenate((data, augmented_12), axis=0)
    print('data after adding augmented12: '+str(data.shape))
    label = np.concatenate((label, np.array([12] * augmented_12.shape[0])))

    data = divide_data(data)

    return data, label


def data_and_split(directory, window, already_windowed, not_already_labelled):
    # First boolean argument to False if there  isn't an already segmented version with dim window_seconds
    # Second boolean argument to False if you already labelled the data
    data, label, sensors = dataset_short_window.get_dataset(directory, window, already_windowed, not_already_labelled)
    
    print('\nUnique labels in Dataset')
    print(np.unique(label))
    print('\nDataSet counter: ')
    c = Counter(label)
    print(c)

    data, label = apply_oversampling(data, label)

    label = np.array([i-1 for i in label])
    print('\nNew unique labels in Dataset')
    print(np.unique(label))
    classes = 24
    print('\nTraining to detect '+str(classes)+'/25 classes')

    Y = to_categorical(label, num_classes=classes)

    return classes, data, Y

def create_model(window, classes):
    sensors=28
    accel_fc = 50

    sensorShape = (window, 11, 13, 1)
    accelerShape = (window*accel_fc, 3)
    timeShape = (2,)
    proxShape = (15,)
    pressureShape = (5, 10, 1)

    accelerometer_input = Input(shape=accelerShape, dtype='float', name='accelerometer_input')
    sensor_input = Input(shape=sensorShape, dtype='float', name='sensor_input')
    time_input = Input(shape=timeShape, dtype='float', name='time_input')
    prox_input = Input(shape=proxShape, dtype='float', name='prox_input')
    pressure_input = Input(shape=pressureShape, dtype='float', name='pressure_input')

    #d = Dense(100, activation='relu')(time_input)
    time_out = BatchNormalization()(time_input)

    k = Conv2D(8, (2, 2), activation='relu', padding='same')(pressure_input)
    k = Conv2D(16, (4, 3), activation='relu', padding='valid')(k)
    k = Flatten()(k)
    k = Dropout(0.5)(k)
    k = BatchNormalization()(k)
    pressure_out = (k)

    #y = Dense(100, activation='relu')(prox_input)
    y = BatchNormalization()(prox_input)
    prox_out = (y)

    x = TimeDistributed(Conv2D(18, (2, 2), activation='relu', padding='same'))(sensor_input)
    x = TimeDistributed(Conv2D(32, (5, 5), activation='relu', padding='same'))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(100, return_sequences=False, dropout=0.5)(x)
    x = BatchNormalization()(x)

    sensor_out = (x)

    filter_size = int((window/5)*2)

    y = Conv1D(18, filter_size, padding="same", activation='relu')(accelerometer_input)
    y = MaxPooling1D(pool_size=filter_size, strides=2)(y)
    y = Conv1D(36, filter_size, padding="valid", activation='relu')(y)
    y = MaxPooling1D(pool_size=filter_size, strides=2)(y)
    y = Conv1D(56, filter_size, activation='relu')(y)
    y = MaxPooling1D(pool_size=filter_size, strides=2)(y)
    y = Conv1D(12, filter_size, activation='relu')(y)
    y = Dropout(0.3)(y)
    y = BatchNormalization()(y)

    accel_out = Flatten()(y)

    z = concatenate([accel_out, sensor_out, time_out, prox_out, pressure_out])

    z = BatchNormalization()(z)

    z = Dense(128, activation='relu')(z)
    z = Dense(64, activation='relu')(z)

    main_output = Dense(classes, activation='softmax', name='main_output')(z)

    model = Model(inputs=[accelerometer_input, sensor_input, time_input, prox_input, pressure_input], outputs=[main_output])

    optimizer = optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #plot_model(model, show_shapes=True, to_file='model.png')
    print(model.summary()) 
    return model


directory = 'Data/Training/'
test_directory = 'Data/Test/'

#already_windowed = '_seconds3d_corrected'
accel_fc = 50

test_beast = True

windows = [5, 6, 10, 15, 30]
plot_confusion = True

if not test_beast:

    for window in windows:

        already_windowed = 'acc_sens_press_prox_v1'
        model_name = 'models/'+str(window)+'acc_sens_press_prox_v1.h5'
        use_saved_model = False 

        
        classes, data, Y = data_and_split(directory, window, already_windowed, not_already_labelled=False)
        print(data[1].shape)
        plot_confusion = True

        train_accel, train_sensor, train_times, train_prox, train_press = data[0], data[1], data[2], data[3], data[4]
        
        train_sensor = np.reshape(train_sensor, (-1, window, 11, 13, 1))
        train_press = np.reshape(train_press, (-1, 5, 10, 1))

        print('\nTrainSensor shape:')
        print(train_sensor.shape)
        trainY = Y

        print('\nTotal Training samples: '+str(train_accel.shape[0]))

        print('\nUnique labels in TrainSet')
        c = Counter(np.argmax(trainY, axis=1))
        print(c)
        print(len(np.unique(np.argmax(trainY, axis=1))))

        print('\nBuild Model...')
        model = None
        model = create_model(window, classes)

        print('\nTraining...')

        history = model.fit([train_accel, train_sensor, train_times, train_prox, train_press], trainY, \
            epochs=10, batch_size=64, verbose=1)
        
        print('\nSaving model')
        model.save(model_name)

        print('\nEvaluating...')
        print('\nWindowing at '+str(window)+' seconds')


        data, sensors, label_test = dataset_short_window.get_testset(test_directory, window)

        tp = 0
        total = 0
        print('\nUsing model '+str(window)+' seconds:')
        for key, value in data.items():

            print('\n'+key)

            test_accel, test_sensor, test_times, test_proximity, test_pressure = divide_data(value, True)

            test_sensor = np.reshape(test_sensor, (-1, window, 11, 13, 1))
            test_pressure = np.reshape(test_pressure, (-1, 5, 10, 1))

            y_pred = model.predict([test_accel, test_sensor, test_times, test_proximity, test_pressure])

            #y_proba = np.max(y_pred, axis=1)

            partial_tp, partial = test_single_module(y_pred, label_test[key], window)
            tp+=partial_tp
            total+=partial
            print('\n'+str(partial_tp/partial)+' %')

        print(tp)
        print(total)
        print('\nAccuracy: '+str(tp/total))

else:
    predictions = {}

    y_predicted = []
    y_true = []
    inv_convert_dict = {v: k for k, v in utils.convert_dict.items()}


    for w in windows:
        model_name = 'models/'+str(w)+'acc_sens_press_prox_v1.h5'
        model = None
        model = load_model(model_name)
        data, sensors, label_test = dataset_short_window.get_testset(test_directory, w)

        print('\nPredictions with '+str(w)+' seconds classifier:')
        t = 0
        t_p = 0
        for key, value in data.items():

            print('\n'+key)

            test_accel, test_sensor, test_times, test_proximity, test_pressure = divide_data(value, True)

            test_sensor = np.reshape(test_sensor, (-1, w, 11, 13, 1))
            test_pressure = np.reshape(test_pressure, (-1, 5, 10, 1))

            y_pred = model.predict([test_accel, test_sensor, test_times, test_proximity, test_pressure])
            
            matrix_prediction = []
            for i in range(0, y_pred.shape[0], int(30/w)):
                portion = np.array(y_pred[i:i+int(30/w), :])
                if portion.shape[0]==int(30/w):
                    matrix_prediction.append(np.mean(portion, axis=0))
                
            matrix_prediction = np.array(matrix_prediction)
            
            pred = np.array([i+1 for i in np.argmax(matrix_prediction, axis=1)])
            for el, (true1, true2) in zip(pred, label_test[key]):
                t+=1
                converted = utils.convert_dict[str(el)]  # for ex. converts 4 in Act4
                print(converted + ' | '+ true1 + ', '+ true2)
                if converted == str(true1) or converted == str(true2):
                    t_p+=1

            if predictions.get(key) is None:
                predictions[key] = matrix_prediction
                
            else:
                
                predictions[key] += matrix_prediction

        print(str(t_p) + '/' + str(t) + ' with '+str(w)+ ' model.')
    tp = 0
    total = 0
    for key, value in predictions.items():
        result = []

        predictions[key] /= len(windows)
        probabilities = predictions[key]
        y_pred = np.array([i+1 for i in np.argmax(predictions[key], axis=1)])

        partial = 0
        partial_tp = 0
        print('\n'+key)

        for idx, (el, (true1,true2)) in enumerate(zip(y_pred, label_test[key])):
            total+=1
            partial+=1
            y_predicted.append(el)

            converted = utils.convert_dict[str(el)]  # for ex. converts 4 in Act4
            print(converted + ' | '+ true1 + ', '+ true2)
            np.set_printoptions(suppress=True)
            if converted == str(true1) or converted == str(true2):
                partial_tp +=1
                tp+=1
                y_true.append(el)
                result.append([el, el])


            else:
                y_true.append(int(inv_convert_dict[true1]))
                result.append([el, int(inv_convert_dict[true1])])

        with open("result_test/result_"+str(key)+".txt", "wb") as fp:
            pickle.dump(result, fp)

            # elif str(true1)=='Act12' and converted in ['Act09', 'Act11']:
            #     partial_tp +=1
            #     tp+=1
        print('\nPartial: '+str(partial_tp/partial)+' %')

    print(tp)
    print(total)
    print('\nTotal: '+str(tp/total))

    # utils.plot_bar_samples(y_true)

    if plot_confusion:

        utils.plot_f1_score(y_true, y_predicted)
        cnf_matrix = confusion_matrix(y_true, y_predicted, labels=np.unique(y_true))
        utils.plot_confusion_matrix(cnf_matrix, classes=np.unique(y_true), normalize=True, title='Normalized confusion matrix')







