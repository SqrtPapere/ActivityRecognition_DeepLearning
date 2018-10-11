

import opportunity_dataset_mt
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Input, GlobalMaxPooling1D, Dropout, BatchNormalization, concatenate
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import backend as K

from keras.utils import plot_model

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

def play():


    dataset = '/home/francesco/Opportunity/OpportunityUCIDataset/dataset/data_saved/oppChallenge_5seconds.data'

    # if not already saved:

    save = True

    if save:
        X_train, y_train_loc, y_train_gest, X_test, y_test_loc, y_test_gest = opportunity_dataset_mt.save_data()
    else:
        X_train, y_train_loc, y_train_gest, X_test, y_test_loc, y_test_gest = opportunity_dataset_mt.load_dataset(dataset)


    print(X_train.shape)

    classes = len(np.unique(y_train_gest))
    print(y_train_gest)
    print(classes)

    y_train_gest = to_categorical(y_train_gest)
    y_test_gest = to_categorical(y_test_gest)

    num_validation_samples = int(0.2 * len(X_train))

    print('num validation sample: '+str(num_validation_samples))
    exit()
    x_val = X_train[-num_validation_samples:]


    y_val = y_train_gest[-num_validation_samples:]


    X_train = X_train[:-num_validation_samples]


    y_train_gest = y_train_gest[:-num_validation_samples]
    accelerShape = (150, 210)

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as sess:
        K.set_session(sess)
        accelerometer_input = Input(shape=accelerShape, dtype='float', name='accelerometer_input')


        y = Conv1D(64, 30, padding="same", activation='relu')(accelerometer_input)

        y = MaxPooling1D(pool_size=20, strides=2)(y)
        y = Conv1D(64, 15, padding="valid", activation='relu')(y)
        y = MaxPooling1D(pool_size=20, strides=20)(y)

        y = Dropout(0.3)(y)

        z = Flatten()(y)

        print('help')
        z = Dense(64, activation='relu')(z)
        z = Dense(64, activation='relu')(z)

        main_output = Dense(classes, activation='softmax', name='main_output')(z)

        model = Model(inputs=[accelerometer_input], outputs=[main_output])
        optimizer = optimizers.Adam(lr=0.001)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        plot_model(model, show_shapes=True, to_file='model.png')

        print(model.summary()) 


        history = model.fit({'accelerometer_input': X_train}, {'main_output': y_train_gest}, validation_data=([x_val], y_val), epochs=50, batch_size=32)

        scores = model.evaluate({'accelerometer_input': X_test }, y_test_gest)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



