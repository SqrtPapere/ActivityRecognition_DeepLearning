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
import shutil
import warnings
import itertools
from sklearn.metrics import confusion_matrix, f1_score
from collections import Counter, OrderedDict
from transforms3d.axangles import axangle2mat  

convert_dict = {'0':'Idle', '1':'Act01', '2':'Act02', '3':'Act03',\
    '4':'Act04', '5':'Act05', '6':'Act06', '7':'Act07', '8':'Act08',\
    '9':'Act09', '10':'Act10', '11':'Act11', '12':'Act12', '13':'Act13',\
    '14':'Act14', '15':'Act15', '16':'Act16', '17':'Act17', '18':'Act18',\
    '19':'Act19', '20':'Act20', '21':'Act21', '22':'Act22', '23':'Act23',\
    '24':'Act24' }

convert_dict2 = {'0':'Idle', '1':'Take medication', '2':'Prepare breakfast', '3':'Prepare lunch',\
    '4':'Prepare dinner', '5':'Breakfast', '6':'Lunch', '7':'Dinner', '8':'Eat a snack',\
    '9':'Watch TV', '10':'Enter the SmartLab', '11':'Play a videogame', '12':'Relax on the sofa', '13':'Leave the SmarLab',\
    '14':'Visit in the SmartLab', '15':'Put waste in the bin', '16':'Wash hands', '17':'Brush teeth', '18':'Use the toilet',\
    '19':'Wash dishes', '20':'Laundry', '21':'Work at the table', '22':'Dressing', '23':'Go to the bed',\
    '24':'Wake up' }

object_to_position = {'TV CONTROLLER':(3, 3), 'BOOK':(1, 0), 'ENTRANCE DOOR':(0,12),\
    'MEDICINE BOX':(10, 11), 'FOOD CUPBOARD':(7, 13), 'FRIDGE':(7, 12), 'POT DRAWER':(5, 12),\
    'WATER BOTTLE':(8, 13), 'GARBAGE CAN':(4, 10), 'WARDROBE DOOR':(5, 1), 'PYJAMAS DRAWER':(6, 1), \
    'BED':(8, 1),'BATHROOM TAP':(6, 7), 'TOOTHBRUSH':(7, 7), 'LAUNDRY BASKET':(8, 10)}

sensor_to_position = {'M01':(0, 9), 'TV0':(4, 3), 'SM1':(4,9), 'SM3':(10, 5), 'SM4':(10, 2),\
    'SM5':(1, 3), 'D01':(6, 10), 'D02':(9, 11), 'D03':(5, 0), 'D04':(7, 11), 'D05':(9, 10), \
    'D07':(10, 4),'D08':(5, 11), 'D09':(9, 9), 'D10':(6, 11), 'H01':(10, 9), 'C01':(10, 11),\
    'C02':(10, 8), 'C03':(5, 10), 'C04':(8, 10), 'C05':(7, 10),'C07':(3, 1), 'C08':(4, 8),\
    'C09':(6, 5), 'C10':(9, 5), 'C12':(9, 8), 'C13':(6, 0), 'C14':(8, 3), 'C15':(9,12), 'S09':(1, 1)}

def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))

def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)



def plot_cross_validation(history_a, history_l, name):
    #plt.style.use("ggplot")
    train_h, val_h = np.array(history_a['train']), np.array(history_a['val'])
    train_mean, val_mean = train_h.mean(0), val_h.mean(0)
    train_std = train_h.std(0)
    val_std = val_h.std(0)
    plt.grid()

    plt.fill_between(list(range(len(train_mean))), train_mean - train_std, train_mean + train_std, alpha=0.1, color="b")
    plt.fill_between(list(range(len(train_mean))), val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")

    plt.plot(train_mean, 'd-', color="b", label="Training score")
    plt.plot(val_mean, 'd-', color="g", label="Test score")
    plt.savefig('images/'+name+"_acc.png")
    print('saved cv plot!')
    plt.clf()

    train_l = np.array(history_l['train'])
    val_l = np.array(history_l['val'])
    train_mean_l = train_l.mean(0)
    val_mean_l = val_l.mean(0)
    train_std_l = train_l.std(0)
    val_std_l = val_l.std(0)
    plt.grid()

    plt.fill_between(list(range(len(train_mean_l))), train_mean_l - train_std_l, train_mean_l + train_std_l, alpha=0.1, color="b")
    plt.fill_between(list(range(len(train_mean_l))), val_mean_l - val_std_l, val_mean_l + val_std_l, alpha=0.1, color="g")

    plt.plot(train_mean_l, 'd-', color="b", label="Training score")
    plt.plot(val_mean_l, 'd-', color="g", label="Test score")
    plt.savefig('images/'+name+"_loss.png")

def plot_f1_score(y_true, y_predicted):

    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)

    index = np.array([i for i, x in enumerate(y_true) if x != 0])
    y_true = y_true[index]
    y_predicted = y_predicted[index]

    print('\nF1-SCORE BOARD: ')
    print('macro: '+str(f1_score(y_true, y_predicted, average='macro')))
    print('micro: '+str(f1_score(y_true, y_predicted, average='micro')))
    print('weighted: '+str(f1_score(y_true, y_predicted, average='weighted')))
    print('None: ')
    values = f1_score(y_true, y_predicted, average=None)
    print(values)

    print(np.unique(y_true))
    c= Counter(y_true)
    print(c)
    indexes = np.unique(np.concatenate((y_true, y_predicted),0))
    pos = list(range(0,len(indexes)))
    plt.bar(indexes, values)
    plt.xticks(indexes, indexes)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()

    cmap=plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm*100
        cm = cm.astype(int)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    where_are_NaNs = np.isnan(cm)
    cm[where_are_NaNs] = 0

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def plot_image(img_np, title = 'noTitle'):
    cv2.imshow(title, img_np)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()


def plot_image_from_df(img_np, enlarge=8, title = 'noTitle'):
    """Plots the activity images.
    Stores resized activity images in a folder for each label.
    Args:
        img_np: Numpy array 
        title: The title of the figure to be plotted
    """
    height, width = img_np.shape[:2]
    res = cv2.resize(img_np,(enlarge*width, enlarge*height), interpolation = cv2.INTER_NEAREST)
    for i in range(0,res.shape[0], enlarge):
        cv2.line(res, (0, i), (res.shape[1], i), (255, 255, 0), 1, 1)
    for i in range(0,res.shape[1], enlarge):
        cv2.line(res, (i, 0), (i, res.shape[0]), (0.5, 0.5, 0.5), 1, 1)
    for i in range(0,res.shape[1], enlarge*15):
        cv2.line(res, (i, 0), (i, res.shape[0]), (1, 0, 0), 1, 1)

    #cv2.imwrite(title+'.png', res)
    cv2.imshow(title, res)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()


def plot_bar_samples(labels, ordered=False):
    c = Counter()
    for l in range(0, 25):
        c[l] = 0
    c.update(labels)
    print(c)
    if ordered:
        values, labels = zip(*sorted(zip(c.values(), c.keys()), reverse=True))
    else:
        labels, values = zip(*c.items())



    indexes = np.arange(0, 25)
    width = 1

    #labels = [ utils.convert_dict2[str(i)] for i in labels]

    plt.bar(indexes, values)
    plt.xticks(indexes, labels, rotation=45)

    #plt.subplots_adjust(bottom=0.4, top=0.99)

    plt.show()

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

def strided_axis0(a, L, overlap=1):
    if L==overlap:
        raise Exception("Overlap arg must be smaller than length of windows")
    S = L - overlap
    nd0 = ((len(a)-L)//S)+1
    if nd0*S-S!=len(a)-L:
        warnings.warn("Not all elements were covered")
    m,n = a.shape
    s0,s1 = a.strides
    return np.lib.stride_tricks.as_strided(a, shape=(nd0,L,n), strides=(S*s0,s0,s1))


def nearest_ind(items, pivot):
    time_diff = np.abs([date - pivot for date in items])
    return time_diff.argmin(0)

def translate_for_categorical(vector):

    label_renamer = dict()
    unique_lab = np.unique(vector)

    for i, k in enumerate(unique_lab):
        label_renamer[str(k)]=str(i)

    print('\nConversion of labels using map: ')
    print(label_renamer)
    return [float(label_renamer[str(elem)]) for elem in vector]

def correct_wrong_timestamps():

    exit()


    # print('\nRemoving error character in: 2017-11-21-A-acceleration.csv')

    # acc_test_er = 'Data/Test/2017-11-21/2017-11-21-A/2017-11-21-A-acceleration.csv'

    # data_acc_er = pd.read_csv(acc_test_er, sep=';', parse_dates = ['TIMESTAMP'])

    # step = pd.to_timedelta(1/50, unit='s')
    # for i in range(0, len(data_acc_er)-1):

    #     if pd.isna(data_acc_er.loc[i+1,'TIMESTAMP']):
    #         data_acc_er.loc[i+1,'TIMESTAMP'] = data_acc_er.loc[i,'TIMESTAMP']+ step

    #     elif data_acc_er.loc[i,'TIMESTAMP'] > data_acc_er.loc[i+1,'TIMESTAMP']:
    #         data_acc_er.loc[i+1,'TIMESTAMP'] = data_acc_er.loc[i,'TIMESTAMP']+ step
        
        

    # data_acc_er.to_csv(acc_test_er, index=False, sep=';')


    # print('\nRemoving error character in: 2017-11-21-A-proximity.csv')
    # prox_test_er = 'Data/Test/2017-11-21/2017-11-21-A/2017-11-21-A-proximity.csv'

    # with open(prox_test_er, 'r', encoding="utf-8", errors="ignore") as f:
    #     lines = f.readlines()
    # lines = lines[:-2]
    # with open(prox_test_er, 'w') as f:
    #     for item in lines:
    #         f.write(item)


    print('\nRemoving error character in: 2017-10-31-C-floor.csv')
    error_file = 'Data/Training/2017-10-31/2017-10-31-C/2017-10-31-C-floor.csv'
    with open(error_file, 'r', encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    lines = lines[:-1]
    with open(error_file, 'w') as f:
        for item in lines:
            f.write(item)


    print('\nRemoving error character in: 2017-11-08-A-floor.csv')
    error_file = 'Data/Training/2017-11-08/2017-11-08-A/2017-11-08-A-floor.csv'
    with open(error_file, 'r', encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    lines = lines[:-1]
    with open(error_file, 'w') as f:
        for item in lines:
            f.write(item)
    
    print('\nRemoving error character in: 2017-10-31-B-floor.csv')
    error_file = 'Data/Training/2017-10-31/2017-10-31-B/2017-10-31-B-floor.csv'
    with open(error_file, 'r', encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    lines = lines[:-1]
    with open(error_file, 'w') as f:
        for item in lines:
            f.write(item)

    print('\nChanging wrong header in file: 2017-11-20-C-sensors.csv')
    wrong_header_file = 'Data/Training/2017-11-20/2017-11-20-C/2017-11-20-C-sensors.csv'
    header_df = pd.read_csv(wrong_header_file, sep=';')
    header_df.columns = ['TIMESTAMP', 'OBJECT', 'STATE', 'HABITANT']
    header_df.to_csv(wrong_header_file, index=False, sep=';')

    print('\nRemoving corrupted file from Test...')
    empty_acc_dir = 'Data/Test/2017-11-21/2017-11-21-A/'
    shutil.rmtree(empty_acc_dir)

    prox_file = 'Data/Training/2017-11-03/2017-11-03-C/2017-11-03-C-proximity.csv'
    print('\nTruncating corrupted file: '+prox_file)

    wrong_time = '2017-11-03 17:12:00.411'
    prox = pd.read_csv(prox_file, sep=';', parse_dates=['TIMESTAMP'])
    prox = prox.truncate(after=prox.loc[prox['TIMESTAMP'] == wrong_time].iloc[1].name-1)
    prox.to_csv(prox_file, index=False, sep=';')

    step = pd.to_timedelta(1/50, unit='s')
    path = 'Data/Training/'
    files = ['2017-11-03/2017-11-03-B/2017-11-03-B-acceleration.csv',\
        '2017-11-20/2017-11-20-B/2017-11-20-B-acceleration.csv',\
        '2017-11-10/2017-11-10-A/2017-11-10-A-acceleration.csv',\
        '2017-10-31/2017-10-31-C/2017-10-31-C-acceleration.csv']
    
    print('\n')
    for el in files:
        file_path = path+el
        print('Correcting: '+file_path)
        df = pd.read_csv(file_path, sep=';', parse_dates=['TIMESTAMP'])

        for i in range(0, len(df)):
            if i == len(df) - 1:
                pass
            else:
                if df.loc[i,'TIMESTAMP'] > df.loc[i+1,'TIMESTAMP']:
                    df.loc[i+1,'TIMESTAMP'] = df.loc[i,'TIMESTAMP'] + step
        df.to_csv(file_path, index=False, sep=';')

if __name__ == '__main__':
    correct_wrong_timestamps()










