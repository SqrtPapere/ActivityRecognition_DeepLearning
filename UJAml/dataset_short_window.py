import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from pandas.api.types import is_string_dtype
from datetime import date, time, datetime
#import cv2
import re
import pickle
import math


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

def get_sensors_names(file_path):
    sensors = []
    with open(file_path, 'r') as filehandle:  
        for line in filehandle:
            currentPlace = line[:-1]
            sensors.append(currentPlace)
    return sensors

def feature_normalize(dataset):

    only_accelerometer_data = [tris[0] for tris in dataset]
    stacked_data = pd.concat(only_accelerometer_data)
    mu_x = np.mean(stacked_data['X'],axis = 0)
    sigma_x = np.std(stacked_data['X'],axis = 0)
    mu_y = np.mean(stacked_data['Y'],axis = 0)
    sigma_y = np.std(stacked_data['Y'],axis = 0)
    mu_z = np.mean(stacked_data['Z'],axis = 0)
    sigma_z = np.std(stacked_data['Z'],axis = 0)
    for tris in dataset:
        tris[0]['X'] = (tris[0]['X'] - mu_x)/sigma_x
        tris[0]['Y'] = (tris[0]['Y'] - mu_y)/sigma_y
        tris[0]['Z'] = (tris[0]['Z'] - mu_z)/sigma_z
    return dataset

def label_data(file_path_data, file_path_label):
    data = pd.read_csv(file_path_data, sep=';', parse_dates=['TIMESTAMP'])
    label = pd.read_csv(file_path_label, sep=';', parse_dates=['DATE BEGIN', 'DATE END'])
    data['ACTIVITY'] = '0'
    for index, row in data.iterrows():
        for ind, activity_row in label.iterrows():
            if activity_row['DATE BEGIN'] < row['TIMESTAMP'] < activity_row['DATE END']:
                data.loc[index,'ACTIVITY'] = activity_row['ACTIVITY']
                break
    data['ACTIVITY'] = data['ACTIVITY'].str.replace('Act0','')
    data['ACTIVITY'] = data['ACTIVITY'].str.replace('Act','')

    new_name = file_path_data[:-4]+'-Labelled.csv'
    data.to_csv(new_name, index=False)

def create_labelled(folder):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        
        if os.path.isfile(path):
            if 'sensors.' in path:
                label_file = path.replace('sensors','activity')
                sensor_data = label_data(path, label_file)
                accelerometer_file = path.replace('sensors','acceleration')

                accelerometer_data = label_data(accelerometer_file, label_file)
        else:
                create_labelled(path)


def read_data(file_path_data):
    data = pd.read_csv(file_path_data, sep=',', parse_dates=['TIMESTAMP'])
    return data

def scan_dir(folder, df_list):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if os.path.isfile(path):

            if 'sensors-Labelled' in path:
                #print('\nReading files in:'+ str(path))
                sensor_data = read_data(path)
                accelerometer_file = path.replace('sensors','acceleration')
                accelerometer_data = read_data(accelerometer_file)
                df_list.append([accelerometer_data, sensor_data])
                break
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

def create_ImageDataframe(start_image, datas, index_names):
    df_temp = pd.DataFrame(start_image, index=index_names)

    start_time = datas.iloc[0]['TIMESTAMP']
    df_temp[start_image.shape[1]-1] = 0.5  #stopper

    for index, row in datas.iterrows():
        position = int((datetime.combine(date.min, row['TIMESTAMP'].time()) - datetime.combine(date.min, start_time.time())).total_seconds())
        if position>=start_image.shape[1]:
            position = start_image.shape[1]-1
        df_temp.loc[row['OBJECT'], position] = float(row['STATE'])

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

    times = []
    activities = []
    start_time_activity = data.iloc[0]['TIMESTAMP']
    end_time_activity = data.iloc[-1]['TIMESTAMP']
    duration = (datetime.combine(date.min, end_time_activity.time()) - datetime.combine(date.min, start_time_activity.time())).total_seconds()
    start_image = np.zeros((len(sensor_names), int(duration)))
    activity_df = create_ImageDataframe(start_image, data, sensor_names)
    activities.extend([int(1)]*start_image.shape[1])

    return activity_df.values, activities



    
def windows(data_length, size): #returns a list of index [start, end] that can be iterated
    start = 0 #lo start del prossimo turno è l'end di questo
    while start < data_length-size:
        yield int(start), int(start + size)
        start += (size / 2)
        #start += size  # per ora facciamolo funzionare senza sovrapposizione


def segment_sensor_signal(sensor_data, window_size_seconds, sensor_names):
    data, lab = sensor_value_to_binary(sensor_data, sensor_names)
    segments = []
    labels = np.empty((0))
    for (start, end) in windows(data.shape[1], window_size_seconds):
        window = data[:,start:end]
        if(window.shape[1] == window_size_seconds):
            l = [stats.mode(lab[start:end])[0][0]]
            segments.append(np.transpose(window))
            labels = np.append(labels, l)
    return segments, labels


def segment_accelerometer_signal(accelerometer_data, window_size_seconds): #give a day stream od accelerometer, returns a list of np.array of windowed signals of length ~window_size_seconds
    #window_length = int(calculate_fc(accelerometer_data)*window_size_seconds)
    hour_data = []
    window_length = int(50*window_size_seconds)

    segments = np.empty((0,window_length,3))
    labels = np.empty((0))
    for (start, end) in windows(accelerometer_data.shape[0], window_length):

        x = accelerometer_data["X"][start:end]
        y = accelerometer_data["Y"][start:end]
        z = accelerometer_data["Z"][start:end]

        if(len(accelerometer_data["TIMESTAMP"][start:end]) == window_length):
            l = [stats.mode(accelerometer_data["ACTIVITY"][start:end])[0][0]]
            segments = np.vstack([segments,np.dstack([x,y,z])])
            labels = np.append(labels,l)
            hour_data.append(accelerometer_data["TIMESTAMP"][start].hour)

    return segments, labels, hour_data


def window_signals(list_of_tris, window_in_sec, sensor_list):
    data = []
    labels_acc = np.empty((0))
    labels_sens = np.empty((0))


    for tris in list_of_tris:

        sensors, sens_lab = segment_sensor_signal(tris[1], window_in_sec, sensor_list)

        accelerometer, accel_lab, hour = segment_accelerometer_signal(tris[0], window_in_sec)

        min_length = min(len(accelerometer), len(sensors)) #add other sources here
        labels_acc = np.append(labels_acc, accel_lab[:min_length])

        accelerometer = accelerometer[:min_length]
        sensors = sensors[:min_length]
        hour = hour[:min_length]


        for acc, sens, h in zip(accelerometer, sensors, hour):
            data.append([acc, sens, h])

    return data, labels_acc

def nearest_ind(items, pivot):
    time_diff = np.abs([date - pivot for date in items])
    return time_diff.argmin(0)

def align_sources(list_of_tris): #given a list of tris, truncate them at the same start and end
    for tris in list_of_tris:

        acc_start = tris[0]['TIMESTAMP'].iloc[0]
        sens_start = tris[1]['TIMESTAMP'].iloc[0]

        acc_end = tris[0]['TIMESTAMP'].iloc[-1]
        sens_end = tris[1]['TIMESTAMP'].iloc[-1]

        latest_start = max([acc_start, sens_start])

        nearest_acc_start = nearest_ind(tris[0]['TIMESTAMP'].tolist(), latest_start)
        nearest_sens_start = nearest_ind(tris[1]['TIMESTAMP'].tolist(), latest_start)

 
        tris[0] = tris[0].truncate(before=nearest_acc_start).reset_index(drop=True) 
        tris[1] = tris[1].truncate(before=nearest_sens_start).reset_index(drop=True) 

        tris[0].loc[0, 'TIMESTAMP'] = latest_start
        tris[0].loc[tris[0]['TIMESTAMP'] < latest_start, 'TIMESTAMP'] = latest_start

        tris[1].loc[0, 'TIMESTAMP'] = latest_start
        tris[1].loc[tris[1]['TIMESTAMP'] < latest_start, 'TIMESTAMP'] = latest_start

    return list_of_tris

def load_dataset(filename_path):
    print(filename_path)
    loaded = np.load(filename_path)

    return loaded['windowed_data'], loaded['labels'], loaded['sensors']

def get_dataset(directory, window_size_seconds, just_read, not_labelled):

    start = 'DataAligned/'
    if not os.path.exists(start):
        os.makedirs(start)
        
    if just_read:
        load_from = start+str(window_size_seconds)+'_seconds.npz'
        windowed_data, labels, sensors = load_dataset(load_from)

    else:

        sensor_name_path = 'info/sensors.txt'

        dfs = []

        if not_labelled:
            print('labelling all dataset...(takes a while)')
            create_labelled(directory)

        print('\nReading files...')
        scan_dir(directory, dfs)

        sensors = get_sensors_names(sensor_name_path)

        print('aligning sources... ')
        dfs = align_sources(dfs)

        print('normalizing features... ')
        dfs = feature_normalize(dfs)

        print('numero di misurazioni: '+str(len(dfs)))

        print('windowing signals... ')
        windowed_data, labels = window_signals(dfs, window_size_seconds, sensors)


        save_to = start+str(window_size_seconds)+'_seconds'
        np.savez(save_to, windowed_data=windowed_data, labels=labels, sensors=sensors)
        print('\nSaved '+save_to)


    return windowed_data, labels, len(sensors)













