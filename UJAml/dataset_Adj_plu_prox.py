import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
import os
from pandas.api.types import is_string_dtype
from datetime import date, time, datetime
import re
import math
import utils_module as utils
from pandas import ExcelWriter, ExcelFile
from collections import deque

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
    print('\nLabelling file: '+file_path_data)
    data = pd.read_csv(file_path_data, sep=';', parse_dates=['TIMESTAMP'])
    label = pd.read_csv(file_path_label, sep=';', parse_dates=['DATE BEGIN', 'DATE END'])
    data['ACTIVITY'] = '0'
    for ind, activity_row in label.iterrows():
        mask = (data['TIMESTAMP'] >= activity_row['DATE BEGIN']) & (data['TIMESTAMP'] <= activity_row['DATE END'])
        data.loc[ mask, 'ACTIVITY' ] = activity_row['ACTIVITY']

    data['ACTIVITY'] = data['ACTIVITY'].str.replace('Act0','')
    data['ACTIVITY'] = data['ACTIVITY'].str.replace('Act','')

    new_name = file_path_data[:-4]+'-Labelled.csv'
    data.to_csv(new_name, sep=';', index=False)

def create_labelled(folder):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            if 'acceleration.' in path:
                label_file = path.replace('acceleration','activity')
                accelerometer_data = label_data(path, label_file)
                break
        else:
                create_labelled(path)

def read_data(file_path_data, separator = ','):
    data = pd.read_csv(file_path_data, sep=separator, parse_dates=['TIMESTAMP'])
    return data

def scan_dir_test(folder, df_list, df_names):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if os.path.isfile(path):

            if 'sensors.' in path:
                sensor_data = read_data(path, ';')
                accelerometer_file = path.replace('sensors','acceleration')
                accelerometer_data = read_data(accelerometer_file, ';')
                proximity_file = path.replace('sensors','proximity')
                proximity_data = read_data(proximity_file, ';')
                pressure_file = path.replace('sensors','floor')
                pressure_data = read_data(pressure_file, ';')
                df_list.append([accelerometer_data, sensor_data, proximity_data, pressure_data])
                df_names.append(path)
                break
        else:
                scan_dir_test(path, df_list, df_names)

def scan_dir(folder, df_list):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):

            if 'acceleration-Labelled' in path:
                print(path)

                accelerometer_data = read_data(path, ';')
                sensor_file = path.replace('acceleration-Labelled','sensors')
                sensor_data = read_data(sensor_file, ';')
                proximity_file = path.replace('acceleration-Labelled','proximity')
                proximity_data = read_data(proximity_file, ';')
                pressure_file = path.replace('acceleration-Labelled','floor')
                pressure_data = read_data(pressure_file, ';')
                df_list.append([accelerometer_data, sensor_data, proximity_data, pressure_data])
                break
        else:
                scan_dir(path, df_list)

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
    start_time_activity = data.iloc[0]['TIMESTAMP']
    end_time_activity = data.iloc[-1]['TIMESTAMP']
    duration = (datetime.combine(date.min, end_time_activity.time()) - datetime.combine(date.min, start_time_activity.time())).total_seconds()
    start_image = np.zeros((len(sensor_names), int(duration)))
    activity_df = create_ImageDataframe(start_image, data, sensor_names)

    return activity_df.values


def windows(data_length, size, test_set=False): #returns a list of index [start, end] that can be iterated
    start = 0 
    while start <= data_length-size:
        yield int(start), int(start + size)
        if test_set:
            start += size
        else:
            start += (size / 2)


def segment_sensor_signal(sensor_data, window_size_seconds, sensor_names, test_set=False):
    data = sensor_value_to_binary(sensor_data, sensor_names)
    count = 0
    segments = []
    data_3d = np.zeros([data.shape[1], 11, 13])
    for idx, sens in enumerate(sensor_names):
        position = utils.sensor_to_position[sens]
        data_3d[:, position[0], position[1]] = data[idx]

    for (start, end) in windows(data.shape[1], window_size_seconds, test_set):
        window_3d = data_3d[start:end, :, :]
        if(window_3d.shape[0] == window_size_seconds):
            segments.append(window_3d)

    return segments


def segment_accelerometer_signal(accelerometer_data, window_size_seconds, test_set=False): #give a day stream od accelerometer, returns a list of np.array of windowed signals of length ~window_size_seconds
    fc = 50
    hour_data = []
    window_length = int(fc*window_size_seconds)
    window_times = []

    segments = np.empty((0,window_length,3))
    labels = np.empty((0))

    for (start, end) in windows(accelerometer_data.shape[0], window_length, test_set):

        x = accelerometer_data["X"][start:end]
        y = accelerometer_data["Y"][start:end]
        z = accelerometer_data["Z"][start:end]

        if(len(accelerometer_data["TIMESTAMP"][start:end]) == window_length):
            l = 0
            if not test_set:
                l = [stats.mode(accelerometer_data["ACTIVITY"][start:end])[0][0]]
            window_times.append((accelerometer_data["TIMESTAMP"][start], accelerometer_data["TIMESTAMP"][end-1]))
            segments = np.vstack([segments,np.dstack([x,y,z])])
            labels = np.append(labels,l)
            hour_data.append(accelerometer_data["TIMESTAMP"][start].hour)

    return segments, hour_data, window_times, labels


def segment_pressure_signal(pressure_data, windows_timestamps): 

    pressure_data = pressure_data.set_index(['TIMESTAMP'])
    segments = []

    for (start, end) in windows_timestamps:
        floor_2d = np.zeros([5, 10])
        windowed_press = pressure_data.loc[start:end]

        for index, row in windowed_press.iterrows():
            pos = row['DEVICE'].split(',')
            try:
                r = int(pos[0])-1
                c = int(pos[1])-1
            except ValueError as ex:
                continue
            
            floor_2d[r, c] = 1


        segments.append(floor_2d) 

    return segments

def segment_proximity_signal(proximity_data, windows_timestamps, proximity_names): 

    prox_data = proximity_data.set_index(['TIMESTAMP'])
    segments = []

    for (start, end) in windows_timestamps:
        df = pd.DataFrame(0, index=[0], columns=proximity_names)
        windowed_prox = prox_data.loc[start:end]
        for index, row in windowed_prox.iterrows():
            df[row['OBJECT']] = 1

        segments.append(np.squeeze(df.values)) # squeeze: [[array]]->[array]

    return segments

def window_signals(list_of_tris, window_in_sec, sensor_list, test_set=False, remove_0=True):
    labels_acc = np.empty((0))

    proximity_name_path = 'info/proximity_sensors.txt'
    proximity_list = utils.get_sensors_names(proximity_name_path)
    if test_set:

        data = [list() for _ in range(0, len(list_of_tris))]

        for idx, tris in enumerate(list_of_tris):

            accelerometer, hour, window_times, accel_lab = segment_accelerometer_signal(tris[0], window_in_sec, test_set)
            sensors = segment_sensor_signal(tris[1], window_in_sec, sensor_list, test_set)
            proximity = segment_proximity_signal(tris[2], window_times, proximity_list)
            pressure = segment_pressure_signal(tris[3], window_times)

            min_length = min(len(accelerometer), len(sensors)) #add other sources here
            labels_acc = np.append(labels_acc, accel_lab[:min_length])

            accelerometer = accelerometer[:min_length]
            sensors = sensors[:min_length]
            hour = hour[:min_length]
            for acc, sens, h, prox, press in zip(accelerometer, sensors, hour, proximity, pressure):
                time_feature = np.array([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)])
                data[idx].append([acc, sens, time_feature, prox, press])

    else:
        data = []

        for idx, tris in enumerate(list_of_tris):

            accelerometer, hour, window_times, accel_lab = segment_accelerometer_signal(tris[0], window_in_sec, test_set)
            sensors = segment_sensor_signal(tris[1], window_in_sec, sensor_list)

            min_length = min(len(accelerometer), len(sensors))

            accelerometer, accel_lab = accelerometer[:min_length], accel_lab[:min_length]
            sensors, hour = sensors[:min_length], hour[:min_length]

            # remove 0 
            if remove_0:
                selected_index = np.array([i for i, x in enumerate(accel_lab) if x != 0])
                accel_lab = accel_lab[selected_index]
                accelerometer = accelerometer[selected_index]
                hour = np.array(hour)[selected_index]
                window_times = np.array(window_times)[selected_index]
                sensors = np.array(sensors)[selected_index]

            proximity = segment_proximity_signal(tris[2], window_times, proximity_list)
            pressure = segment_pressure_signal(tris[3], window_times)

            for acc, sens, h, prox, press, lab, (a, b) in zip(accelerometer, sensors, hour, proximity, pressure, accel_lab, window_times):
                time_feature = np.array([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)])
                data.append([acc, sens, time_feature, prox, press])
            labels_acc = np.append(labels_acc, accel_lab[:min_length])
    return data, labels_acc


def align_sources(list_of_tris): #given a list of tris, truncate them at the same start and end
    for tris in list_of_tris:

        acc_start = tris[0]['TIMESTAMP'].iloc[0]
        sens_start = tris[1]['TIMESTAMP'].iloc[0]

        acc_end = tris[0]['TIMESTAMP'].iloc[-1]
        sens_end = tris[1]['TIMESTAMP'].iloc[-1]

        latest_start = max([acc_start, sens_start])

        nearest_acc_start = tris[0].TIMESTAMP.searchsorted(latest_start)[0]
        nearest_sens_start = tris[1].TIMESTAMP.searchsorted(latest_start)[0]

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

def get_testset(directory, window_size_seconds):

    label_position = {'2017-11-09-A':(3, 45), '2017-11-09-B':(57, 131),\
        '2017-11-09-C':(142, 201), '2017-11-13-A':(213, 272), \
        '2017-11-13-B':(282, 354), '2017-11-13-C':(366, 417), \
        '2017-11-21-B':(492, 500), '2017-11-21-C':(512, 571) }

    truth_data = pd.read_excel('Data/ground truth.xlsx')
    truth_data = truth_data[['Unnamed: 14', 'Unnamed: 15']]

    sensor_name_path = 'info/sensors.txt'
    dfs = []
    dfs_names = []
    print('\nReading files...')
    scan_dir_test(directory, dfs, dfs_names)

    label_by_name = {}
    for el in dfs_names:
        name = el.split("/")
        name = name[-2]
        label_by_name[name] = truth_data.loc[label_position[name][0]:label_position[name][1]].values

    sensors = utils.get_sensors_names(sensor_name_path)

    print('normalizing features... ')
    dfs = feature_normalize(dfs)

    print('numero di misurazioni: '+str(len(dfs)))

    print('windowing signals... ')
    windowed_data, _ = window_signals(dfs, window_size_seconds, sensors, test_set=True)

    named_windowed_data = {}
    for el, n in zip(windowed_data, dfs_names):
        n_list = n.split("/")
        name = n_list[-2]
        named_windowed_data[name] = el

    return named_windowed_data, len(sensors), label_by_name

def get_dataset(directory, window_size_seconds, just_read, not_labelled, save=True):

    start = 'DataAligned/'
    if not os.path.exists(start):
        os.makedirs(start)

    stringed_w = str(window_size_seconds)

    if just_read != '':

        load_from = start + stringed_w + just_read+ ".npz"
        windowed_data, labels, sensors = load_dataset(load_from)

    else:
        sensor_name_path = 'info/sensors.txt'
        dfs = []

        if not_labelled:
            print('labelling all dataset...(now fast)')
            create_labelled(directory)
        print('\nReading files...')
        scan_dir(directory, dfs)

        if len(dfs)==0:
            print('\nProbably you need to set not_already_labelled to True in main!')

        sensors = utils.get_sensors_names(sensor_name_path)

        print('aligning sources... ')
        dfs = align_sources(dfs)
        print('normalizing features... ')
        dfs = feature_normalize(dfs)

        print('numero di misurazioni: '+str(len(dfs)))

        print('windowing signals... ')
        windowed_data, labels = window_signals(dfs, window_size_seconds, sensors)

        if save:
            save_to = start+str(window_size_seconds)+'acc_sens_press_prox_v1'
            np.savez(save_to, windowed_data=windowed_data, labels=labels, sensors=sensors)
            print('\nSaved '+save_to)


    return windowed_data, labels, len(sensors)



#correct_wrong_timestamps()










