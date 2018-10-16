import numpy as np
import matplotlib.pyplot as plt
import os

import dataset #script for dataset management

from keras.utils import to_categorical
#from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Input, GlobalMaxPooling1D, Dropout, Merge, BatchNormalization, concatenate, LSTM, Activation
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import backend as K

from keras.utils import plot_model

import cv2

import re
import imageio

from sklearn.metrics import f1_score


import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

def save_sensor_images(train_sensor, trainY):
    count = 1

    for img, lbl in zip(train_sensor, trainY):

        start = '/Users/francescopegoraro/Google Drive/MasterThesis/images_from_sensors/'

        if not os.path.exists(start+str(lbl)):
            os.makedirs(start+str(lbl))
        height, width = img.shape[:2]
        res = cv2.resize(img,(8*width, 8*height), interpolation = cv2.INTER_NEAREST)
        imageio.imwrite(start+str(lbl)+'/'+str(count)+'.png', res)
        count+=1


def plotting(history):
    plt.style.use("ggplot")
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_image_from_df(img_np, title = 'noTitle'):
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

def translate(text, conversion_dict, before=None):

    # if empty:
    if not text: return float(text)
    # preliminary transformation:
    before = before or str.lower
    t = before(text)
    for key, value in conversion_dict.items():
        t = t.replace(key, value)
    return float(t)
    
label_renamer = {'12':'11', '15':'12', '16':'13','17':'14', '18':'15','19':'16', '20':'17', '21':'18','22':'19', '23':'20','24':'21'}

directory = '/Users/francescopegoraro/Google Drive/MasterThesis/DataMulti/Training/'

window = 28
accel_fc = 50

data, label, sensors = dataset.get_dataset(directory, window, True)


print(len(label))
#classes = 25
print(np.unique(label))
label = [translate(str(elem), label_renamer) for elem in label]


classes = len(np.unique(label))


print('unique labels:')
print(len(np.unique(label)))
print(np.unique(label))
(trainX, testX, trainY, testY) = train_test_split(data, label, test_size=0.25, random_state=37)

print(trainX[0][2])
train_accelerometer = np.array([x[0] for x in trainX], dtype="float")
train_sensor = np.array([x[1] for x in trainX], dtype="float")
train_times = np.array([[np.sin(2*np.pi*x[2]/24), np.cos(2*np.pi*x[2]/24)] for x in trainX], dtype="float")
train_proximity = np.array([x[3] for x in trainX], dtype="float")
# train_proximity = np.array([x[2] for x in trainX], dtype="float")

test_accelerometer = np.array([x[0] for x in testX] , dtype="float")
test_sensor = np.array([x[1] for x in testX], dtype="float")
test_times = np.array([[np.sin(2*np.pi*x[2]/24), np.cos(2*np.pi*x[2]/24)] for x in testX], dtype="float")
test_proximity = np.array([x[3] for x in testX], dtype="float")

# test_proximity = np.array([x[2] for x in testX], dtype="float")



trainY = to_categorical(trainY)
testY_true = testY
testY = to_categorical(testY)



print('numero finestre in Train: '+str(len(trainX)))



num_validation_samples = int(0.2 * len(trainX))

print('num validation sample: '+str(num_validation_samples))

accel_val = train_accelerometer[-num_validation_samples:]
sens_val = train_sensor[-num_validation_samples:]
times_val = train_times[-num_validation_samples:]
prox_val = train_proximity[-num_validation_samples:]

y_val = trainY[-num_validation_samples:]

print('dimensione dati sensori: '+str(sens_val.shape))


train_accelerometer = train_accelerometer[:-num_validation_samples]
train_sensor = train_sensor[:-num_validation_samples]
train_times = train_times[:-num_validation_samples]
train_proximity = train_proximity[:-num_validation_samples]

trainY = trainY[:-num_validation_samples]

print(train_sensor.shape)
print(train_accelerometer.shape)
print(train_times.shape)
print(train_proximity.shape)

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

x = LSTM(50)(sensor_input)
x = BatchNormalization()(x)

# x = Conv1D(25, 5, padding="valid", activation='relu')(sensor_input)
# x = MaxPooling1D(pool_size=2)(x)
# #x = Dropout(0.3)(x)
# x = Conv1D(25, 3, activation='relu')(x)
# x = MaxPooling1D(pool_size=2)(x)
# #x = Dropout(0.3)(x)
sensor_out = (x)
#46%

# y = LSTM(500, return_sequences=True)(accelerometer_input)

# y = LSTM(125)(y)

y = Conv1D(64, 30, padding="same", activation='relu')(accelerometer_input)

y = MaxPooling1D(pool_size=20, strides=20)(y)
y = Conv1D(64, 15, padding="valid", activation='relu')(y)
y = MaxPooling1D(pool_size=20, strides=20)(y)

y = Dropout(0.3)(y)
# y = Conv1D(30, 10, activation='relu')(y)
# y = MaxPooling1D(pool_size=20)(y)
# y = Conv1D(20, 4, activation='relu')(y)
# y = MaxPooling1D(pool_size=10)(y)
# y = Dropout(0.1)(y)
accel_out = Flatten()(y)

z = concatenate([accel_out, sensor_out, time_out, prox_out])
#z = concatenate([sensor_out, time_out, prox_out])
z = BatchNormalization()(z)

z = Dense(64, activation='relu')(z)
z = Dense(64, activation='relu')(z)

main_output = Dense(classes, activation='softmax', name='main_output')(z)

model = Model(inputs=[accelerometer_input, sensor_input, time_input, prox_input], outputs=[main_output])
optimizer = optimizers.Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

plot_model(model, show_shapes=True, to_file='model.png')

print(model.summary()) 


history = model.fit({'accelerometer_input': train_accelerometer, 'sensor_input': train_sensor, 'time_input': train_times, 'prox_input': train_proximity}, {'main_output': trainY}, validation_data=([accel_val, sens_val, times_val, prox_val], y_val), epochs=50, batch_size=32)

scores = model.evaluate({'accelerometer_input': test_accelerometer, 'sensor_input': test_sensor, 'time_input': test_times, 'prox_input': test_proximity}, testY)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

from collections import Counter
c = Counter(testY_true)
print(c)

y_pred = model.predict({'accelerometer_input': test_accelerometer, 'sensor_input': test_sensor, 'time_input': test_times, 'prox_input': test_proximity})

y_pred = np.argmax(y_pred,axis=1)
print(f1_score(testY_true, y_pred, average='macro'))
print(f1_score(testY_true, y_pred, average='micro'))
print(f1_score(testY_true, y_pred, average='weighted'))
print(f1_score(testY_true, y_pred, average=None))

# model.save('my_model3.h5')

plotting(history)













