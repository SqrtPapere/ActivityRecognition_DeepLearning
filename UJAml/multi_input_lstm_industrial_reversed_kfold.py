import numpy as np
import matplotlib.pyplot as plt
import os

import dataset #script for dataset management

from keras.utils import to_categorical
#from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout, BatchNormalization, concatenate, LSTM, Activation
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import backend as K

from keras.utils import plot_model
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import re

from sklearn.metrics import f1_score

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from pathlib import Path


def save_sensor_images(train_sensor, trainY):
    """Saves the activity images in png.
    Stores resized activity images in a folder for each label.
    Requires OpenCV
    Args:
        train_sensor: An array with elements of dimension (height, width)
        trainY: An array with the labels of the samples in train_sensor
    """
    count = 1
    start = 'images_from_sensors/'

    for img, lbl in zip(train_sensor, trainY):
        if not os.path.exists(start+str(lbl)):
            os.makedirs(start+str(lbl))
        height, width = img.shape[:2]
        res = cv2.resize(img,(8*width, 8*height), interpolation = cv2.INTER_NEAREST)
        imageio.imwrite(start+str(lbl)+'/'+str(count)+'.png', res)
        count+=1


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


def plot_image_from_df(img_np, title = 'noTitle'):
    """Plots the activity images.
    Stores resized activity images in a folder for each label.
    Args:
        img_np: Numpy array 
        title: The title of the figure to be plotted
    """
    height, width = img_np.shape[:2]
    res = cv2.resize(img_np,(8*width, 8*height), interpolation = cv2.INTER_NEAREST)
    for i in range(0,res.shape[0], 8):
        cv2.line(res, (0, i), (res.shape[1], i), (255, 255, 0), 1, 1)
    for i in range(0,res.shape[1], 8):
        cv2.line(res, (i, 0), (i, res.shape[0]), (0.5, 0.5, 0.5), 1, 1)
    for i in range(0,res.shape[1], 8*45):
        cv2.line(res, (i, 0), (i, res.shape[0]), (1, 0, 0), 1, 1)

    cv2.imshow(title, res)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

def translate2(text, conversion_dict, before=None):
    """Utility that converts a string using a dictionary as mapping
    Stores resized activity images in a folder for each label.
    Used here to convert label set with missing values to range(0, num_classes).
    Args:
        text: The string that should be converted
        conversion_dict: the dict object used as mapping
    """
    # if empty:
    if not text: return float(text)
    # preliminary transformation:
    before = before or str.lower
    t = before(text)
    for key, value in conversion_dict.items():
        t = t.replace(key, value)
    return float(t)

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
    data, label, sensors = dataset.get_dataset(directory, window, already_windowed, not_already_labelled)
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
    proximity = np.array([x[3] for x in data], dtype="float")
    # train_proximity = np.array([x[2] for x in trainX], dtype="float")

    Y = to_categorical(label, num_classes=classes)

    #return classes, {'accelerometer_input': accelerometer, 'sensor_input': sensor, 'time_input': times, 'prox_input': proximity}, Y
    return classes, [accelerometer, sensor, times, proximity], Y

def create_model(window, classes):
    sensors=28
    accel_fc = 50
    sensorShape = (window, sensors)
    accelerShape = (window*accel_fc, 3)
    timeShape = (2,)
    proxShape = (15,)

    accelerometer_input = Input(shape=accelerShape, dtype='float', name='accelerometer_input')
    sensor_input = Input(shape=sensorShape, dtype='float', name='sensor_input')
    time_input = Input(shape=timeShape, dtype='float', name='time_input')
    prox_input = Input(shape=proxShape, dtype='float', name='prox_input')

    prox_out = (prox_input)

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

    z = concatenate([accel_out, sensor_out, time_out, prox_out])
    z = BatchNormalization()(z)

    z = Dense(64, activation='relu')(z)
    z = Dense(64, activation='relu')(z)

    main_output = Dense(classes, activation='softmax', name='main_output')(z)

    model = Model(inputs=[accelerometer_input, sensor_input, time_input, prox_input], outputs=[main_output])
    optimizer = optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #plot_model(model, show_shapes=True, to_file='model.png')
    #print(model.summary()) 
    return model



# remember to change header in 2017-11-20-C-sensors.csv DATE->TIMESTAMP
directory = 'Data/Training/'

accel_fc = 50
window = 4
classes, data, Y = data_and_split(directory, 4, True, False)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores_acc = []
histories = {'train':[], 'val':[]}

for train_indices, test_indices in kfold.split(np.zeros(Y.shape[0]), np.argmax(Y, axis=1)):
    train_accel, train_sensor, train_times, train_proximity = data[0][train_indices], data[1][train_indices], data[2][train_indices], data[3][train_indices]
    test_accel, test_sensor, test_times, test_proximity = data[0][test_indices], data[1][test_indices], data[2][test_indices], data[3][test_indices]
    trainY = Y[train_indices]
    testY = Y[test_indices]

    print('\nUnique labels in TrainSet')
    c = Counter(np.argmax(trainY, axis=1))
    print(c)
    print(len(np.unique(np.argmax(trainY, axis=1))))
    print('\nUnique labels in TestSet')
    c = Counter(np.argmax(testY, axis=1))
    print(c)
    print(len(np.unique(np.argmax(testY, axis=1))))

    print('\nBuild Model...')
    model = None
    model = create_model(window, classes)

    print('\nTraining...')
    history = model.fit([train_accel, train_sensor, train_times, train_proximity], trainY, epochs=25, batch_size=32, verbose=0)
    # histories['train'].append(history.history['acc'])
    # histories['val'].append(history.history['val_acc'])


    print('\nEvaluating...')

    scores = model.evaluate([test_accel, test_sensor, test_times, test_proximity], testY, verbose=0)

    print('\nWindowing at '+str(window)+' seconds')

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    scores_acc.append(scores[1]*100)
    y_pred = model.predict([test_accel, test_sensor, test_times, test_proximity])
    testY_true = np.argmax(testY, axis=1)
    y_pred = np.argmax(y_pred,axis=1)

    print('\nF1-SCORE BOARD: ')
    print('macro: '+str(f1_score(testY_true, y_pred, average='macro')))
    print('micro: '+str(f1_score(testY_true, y_pred, average='micro')))
    print('weighted: '+str(f1_score(testY_true, y_pred, average='weighted')))
    print('None: ')
    print(f1_score(testY_true, y_pred, average=None))

print("%.2f%% (+/- %.2f%%)" % (np.mean(scores_acc), np.std(scores_acc)))

def plot_cross_validation(history):
    #plt.style.use("ggplot")
    train_h = np.array(history['train'])
    val_h = np.array(history['val'])
    train_mean = train_h.mean(0)
    val_mean = val_h.mean(0)
    train_std = train_h.std(0)
    val_std = val_h.std(0)
    plt.grid()

    plt.fill_between(list(range(len(train_mean))), train_mean - train_std, train_mean + train_std, alpha=0.1, color="b")
    plt.fill_between(list(range(len(train_mean))), val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")

    plt.plot(train_mean, 'd-', color="b", label="Training score")
    plt.plot(val_mean, 'd-', color="g", label="Test score")
    plt.show()
    # plt.plot(train_mean)
    # plt.plot(val_mean)
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig(name+"_accuracy.png")
    # plt.clf()
    

#plot_cross_validation(histories)

#model.save('last_model.h5')

#plotting(history, Path(__file__).stem)













