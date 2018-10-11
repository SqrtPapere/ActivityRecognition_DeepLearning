import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from pandas.api.types import is_string_dtype
from datetime import datetime, date

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Input, GlobalMaxPooling1D, Dropout, Merge, BatchNormalization
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import backend as K

from keras.utils import plot_model

import cv2

import re

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)




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


def read_data(file_path_data, file_path_label):
    data = pd.read_csv(file_path_data, sep=';', parse_dates=['TIMESTAMP'])
    label = pd.read_csv(file_path_label, sep=';', parse_dates=['DATE BEGIN', 'DATE END'])
    data['ACTIVITY'] = '0'
    data['HOUR'] = [d.time() for d in data['TIMESTAMP']]
    #data.drop(data.index[[100:]])
    #data = data[3138:9185] #remove
    for index, row in data.iterrows():
        for ind, activity_row in label.iterrows():
            if activity_row['DATE BEGIN'] < row['TIMESTAMP'] < activity_row['DATE END']:
                data.loc[index,'ACTIVITY'] = activity_row['ACTIVITY']
                break
    data['ACTIVITY'] = data['ACTIVITY'].str.replace('Act0','')
    data['ACTIVITY'] = data['ACTIVITY'].str.replace('Act','')
    #reduce similar classes
    # data['ACTIVITY'] = data['ACTIVITY'].str.replace(r'\b2\b','4')
    # data['ACTIVITY'] = data['ACTIVITY'].str.replace(r'\b3\b','4')
    # data['ACTIVITY'] = data['ACTIVITY'].str.replace(r'\b6\b','7')
    # data['ACTIVITY'] = data['ACTIVITY'].str.replace(r'\b5\b','7')



    #name_csv = file_path_label.replace('-activity','').split('.')[0].split('/')[-1]

    start = file_path_label.split('/')[:-1]
    s = '/'.join(start)

    data.to_csv(s + '/labelled.csv', index=False)

    return data


def create_ImageDataframe(start_image, datas, index_names):
    df_temp = pd.DataFrame(start_image, index=index_names)
    #data['STATE'][]
    start_time = datas.iloc[0]['HOUR']

    df_temp[start_image.shape[1]] = 0.5  #stopper
    i = 0
    for index, row in datas.iterrows():

        position = int((datetime.combine(date.min, row['HOUR']) - datetime.combine(date.min, start_time)).total_seconds())           #delta+=pd.Timedelta(t2 - t1).seconds / 3600.0
        df_temp.loc[row['OBJECT'], position] = float(row['STATE'])
        #df_row = list(df_temp.loc[row['OBJECT']])
        i+=1

    for index, row in df_temp.iterrows():
        i = 0
        while i<len(row):
            elem = row[i]
            l = row.values
            if elem==1.0:
                try:
                    li = l[i:].tolist()
                    next_end = li.index(0.5) + i
                    df_temp.loc[index][i+1:next_end] = 1.0
                    df_temp.loc[index][next_end] = 0.0
                    i = next_end
                except ValueError:
                    next_end = len(row)
                    df_temp.loc[index][i+1:next_end] = 1.0
                    i = next_end
            elif elem == 0.5:
                df_temp.loc[index][i] = 0.0
                i+=1
            else:
                i+=1
    return df_temp




def sensor_value_to_binary(data, sensor_names):
    data['STATE'] = data['STATE'].str.replace('Open','1').str.replace('Close','0.5')
    data['STATE'] = data['STATE'].str.replace('No movement','0.5').str.replace('No Movement','0.5').str.replace('Movement','1')
    data['STATE'] = data['STATE'].str.replace('No pressure','0.5').str.replace('No Pressure','0.5').str.replace('Pressure','1')
    data['STATE'] = data['STATE'].str.replace('No present','0.5').str.replace('No Present','0.5').str.replace('Present','1')

    number_of_activity = len(np.unique(data['ACTIVITY']))
    grouped_activity = data.groupby([(data.ACTIVITY != data.ACTIVITY.shift()).cumsum()])
    #images = np.zeros((len(sensor_names), activity_length, len(grouped_activity)))
    
    blank = np.zeros((len(sensor_names), data.shape[0]))

    # total_row_names = [str(el) for el in range(1,31)]
    # row_names = np.unique(data['OBJECT']).tolist()
    # total_row_names[0:len(row_names)] = row_names

    activities = []
    for k, dg in grouped_activity:
        activity = dg.ACTIVITY.tolist()[0]
        delta = 0.0
        start_time_activity = dg.iloc[0]['HOUR']
        #this is done to get more accurate images
        if k+1>len(grouped_activity):
            end_time_activity = dg.iloc[-1]['HOUR']
        else:
            group = grouped_activity.get_group(k+1)
            end_time_activity = group.iloc[0]['HOUR']

        duration = datetime.combine(date.min, end_time_activity) - datetime.combine(date.min, start_time_activity)            #delta+=pd.Timedelta(t2 - t1).seconds / 3600.0

        start_image = np.zeros((len(sensor_names), int(duration.total_seconds()+1)))
        activity_df = create_ImageDataframe(start_image, data[dg.index[0]:dg.index[-1]+1], sensor_names)
        activities.extend([int(activity)]*start_image.shape[1])
        #images[:, :, k-1] = activity_df.values[:,:activity_length]
        if k-1 == 0:
            image = activity_df.values
        else:
            image = np.concatenate((image, activity_df.values), axis=1)

    return image, activities

def scan_dir(folder, df_list):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if os.path.isfile(path):
            if 'sensors' in path:
                label_file = path.replace('sensors','activity')
                data = read_data(path, label_file)
                df_list.append(data) 
        else:
                scan_dir(path, df_list)

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
    
def windows(data_length, size): #returns a list of index [start, end] that can be iterated
    start = 0
    while start < data_length:
        yield int(start), int(start + size)
        start += (size / 2)
        #start += size

def segment_signal(data, lab, window_size = 90):
    segments = []
    labels = np.empty((0))
    for idx,elem in enumerate(data):
        for (start, end) in windows(elem.shape[1], window_size):
            window = elem[:,start:end]
            if(window.shape[1] == window_size):
                l = [stats.mode(lab[idx][start:end])[0][0]]
                if np.count_nonzero(window): #remove windows made of all 0
                    segments.append(np.transpose(window))
                    labels = np.append(labels, l)
    return segments, labels


sensors = []
with open('info/sensors.txt', 'r') as filehandle:  
    for line in filehandle:
        currentPlace = line[:-1]
        sensors.append(currentPlace)

dfs = []
directory = '/Users/francescopegoraro/Dropbox/MasterThesis/Data/Training/'
scan_dir(directory, dfs)

total_videos = []
total_labels = []

for element in dfs:
    video, labels = sensor_value_to_binary(element, sensors)
    total_videos.append(video)
    total_labels.append(labels)

classes = 25
window = 30

segs, labs = segment_signal(total_videos, total_labels, window)

# for i,j in zip(segs, labs):
#     if j == 0.0:
#         plot_image_from_df(i)

from collections import Counter
c = Counter(labs)
print(c)

segs = np.array(segs, dtype="float")
labs = np.array(labs)

print(segs.shape)

(trainX, testX, trainY, testY) = train_test_split(segs, labs, test_size=0.25, random_state=37)
trainY = to_categorical(trainY)
testY = to_categorical(testY)


print(trainX.shape)


num_validation_samples = int(0.1 * trainX.shape[0])

x_train = trainX[:-num_validation_samples]
y_train = trainY[:-num_validation_samples]

x_val = trainX[-num_validation_samples:]
y_val = trainY[-num_validation_samples:]

print(trainX[0].shape)

model = Sequential()
inputShape = (window, len(sensors))
model.add(Conv1D(25, 3, padding="valid", input_shape=inputShape, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(25, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(classes, activation='softmax'))

optimizer = optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])

plot_model(model, show_shapes=True, to_file='model.png')

print(model.summary()) 


history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=250, batch_size=32, verbose=0)

print('TestSet dimension')
print(testX.shape)
scores = model.evaluate(testX, testY)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# model.save('my_model3.h5')

plotting(history)













