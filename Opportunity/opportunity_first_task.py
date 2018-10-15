

import opportunity_dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense
from keras.layers import Input, GlobalMaxPooling1D, Dropout, BatchNormalization, concatenate
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import backend as K

from keras.utils import plot_model

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

def play():

    window_size_seconds = 5
    fc = 30


   dataset_path = 'OpportunityUCIDataset/dataset/data_saved/5_seconds.npy'

    # if not already saved:

    save = True

    if save:
        X_train, y_train_loc, y_train_gest, X_test, y_test_loc, y_test_gest = opportunity_dataset.save_data(window_size_seconds)
    else:
        X_train, y_train_loc, y_train_gest, X_test, y_test_loc, y_test_gest = opportunity_dataset.load_dataset(dataset_path)

    classes = len(np.unique(y_train_gest))
    print('Training with '+str(classes)+' classes!')

    y_train_gest = to_categorical(y_train_gest)
    y_test_gest = to_categorical(y_test_gest)

    num_validation_samples = int(0.2 * len(X_train))

    print('\nTotal Training dimension: '+str(X_train.shape))

    print('\nvalidation sample: '+str(num_validation_samples))
    

    x_val = X_train[-num_validation_samples:]
    y_val = y_train_gest[-num_validation_samples:]
    X_train = X_train[:-num_validation_samples]
    y_train_gest = y_train_gest[:-num_validation_samples]

    accelerShape = (window_size_seconds*fc, X_train.shape[-1])

    accelerometer_input = Input(shape=accelerShape, dtype='float', name='accelerometer_input')


    y = Conv1D(64, 30, padding="same", activation='relu')(accelerometer_input)

    y = MaxPooling1D(pool_size=20, strides=2)(y)
    y = Conv1D(64, 15, padding="valid", activation='relu')(y)
    y = MaxPooling1D(pool_size=20, strides=20)(y)

    y = Dropout(0.3)(y)

    z = Flatten()(y)

    z = Dense(64, activation='relu')(z)
    z = Dense(64, activation='relu')(z)

    main_output = Dense(classes, activation='softmax', name='main_output')(z)

    model = Model(inputs=[accelerometer_input], outputs=[main_output])
    optimizer = optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    plot_model(model, show_shapes=True, to_file='model.png')

    print(model.summary()) 


    history = model.fit({'accelerometer_input': X_train}, {'main_output': y_train_gest}, validation_data=([x_val], y_val), epochs=10, batch_size=32)

    scores = model.evaluate({'accelerometer_input': X_test }, y_test_gest)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

if __name__ == "__main__":
    play()

