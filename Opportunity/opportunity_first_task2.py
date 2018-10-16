

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

from sklearn.metrics import f1_score

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

def plotting(history):
    plt.style.use("ggplot")
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("accuracy.png")
    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("loss.png")

def f1(y_true, y_pred):
    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def play():

    window_size_seconds = 1

    
    fc = 30

    # if not already saved:

    use_saved = True
    dataset_path = 'OpportunityUCIDataset/dataset/data_saved/'+str(window_size_seconds)+'_seconds.npz'

    if not use_saved:
        X_train, y_train_loc, y_train_gest, X_test, y_test_loc, y_test_gest = opportunity_dataset.save_data(window_size_seconds)
    else:
        X_train, y_train_loc, y_train_gest, X_test, y_test_loc, y_test_gest = opportunity_dataset.load_dataset(dataset_path)

    classes = 18
    t_classes = len(np.unique(y_test_gest))

    gest_classes = 18

    print('Training with '+str(classes)+' classes!')
    print('Testing with '+str(t_classes)+' classes!')
    print(classes)

    y_train_gest = to_categorical(y_train_gest)
    Y_gest_true = y_test_gest
    y_test_gest = to_categorical(y_test_gest)

    num_validation_samples = int(0.2 * len(X_train))

    print('\nTotal Training dimension: '+str(X_train.shape))
    print('\nTotal Test dimension: '+str(X_test.shape))


    print('\nvalidation sample: '+str(num_validation_samples))
    

    x_val = X_train[-num_validation_samples:]
    y_val = y_train_gest[-num_validation_samples:]
    X_train = X_train[:-num_validation_samples]
    y_train_gest = y_train_gest[:-num_validation_samples]

    multiShape = (window_size_seconds*fc, X_train.shape[-1])

    multi_input = Input(shape=multiShape, dtype='float', name='multi_input')

    y = Conv1D(64, 5, padding="valid", activation='relu')(multi_input)
    y = MaxPooling1D(pool_size=5, strides=1)(y)
    y = Dropout(0.3)(y)
    y = Conv1D(64, 5, padding="valid", activation='relu')(y)
    y = MaxPooling1D(pool_size=5, strides=1)(y)

    z = Flatten()(y)
    z = Dense(128, activation='relu')(z)
    z = Dense(128, activation='relu')(z)

    main_output = Dense(classes, activation='softmax', name='main_output')(z)

    model = Model(inputs=[multi_input], outputs=[main_output])
    optimizer = optimizers.Adam(lr=0.0009)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    plot_model(model, show_shapes=True, to_file='model.png')

    print(model.summary()) 


    history = model.fit({'multi_input': X_train}, {'main_output': y_train_gest}, validation_data=(x_val, y_val), epochs=20, batch_size=64)

    plotting(history)
    scores = model.evaluate({'multi_input': X_test}, y_test_gest)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    from collections import Counter
    c = Counter(Y_gest_true)
    print('\nCounter for TestSet: ')
    print(c)

    y_pred = model.predict({'multi_input': X_test})
    K.clear_session()

    y_pred = np.argmax(y_pred,axis=1)
    print('\nF1-SCORE BOARD: ')
    print('macro: '+str(f1_score(Y_gest_true, y_pred, average='macro')))
    print('micro: '+str(f1_score(Y_gest_true, y_pred, average='micro')))
    print('weighted: '+str(f1_score(Y_gest_true, y_pred, average='weighted')))
    print('None: ')
    print(f1_score(Y_gest_true, y_pred, average=None))

if __name__ == "__main__":
    play()

