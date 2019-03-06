import os
import csv
import numpy
from numpy import array
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten,Conv2DTranspose,concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,UpSampling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import *
from keras import optimizers
import h5py
import tensorflow as tf
import keras
from random import shuffle
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape,Permute
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,Lambda,BatchNormalization,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam , SGD
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
import sys
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from PIL import ImageFilter

s = sys.argv

s = list(s)

print(s)

train_file = 0
test_file = 0
img_path = 0

if len(s) == 4 :
    try :
        train_file = str(s[2])
    except :
        train_file = 'training_set.csv'
    try :
        test_file = str(s[3])
    except :
        test_file = 'test.csv'
    try :
        img_path = str(s[1])
        si = list(img_path)
        if si[-1] != '/' :
            img_path = img_path + '/'
    except :
        img_path = 'images/'
elif len(s) == 2 :
    try:
        img_path = str(s[1])
        si = list(img_path)
        if si[-1] != '/':
            img_path = img_path + '/'
    except:
        img_path = 'images/'
    train_file = 'training_set.csv'
    test_file = 'test.csv'
elif len(s) == 3:
    try:
        img_path = str(s[1])
        si = list(img_path)
        if si[-1] != '/':
            img_path = img_path + '/'
    except:
        img_path = 'images/'
    try:
        train_file = str(s[2])
    except:
        train_file = 'training_set.csv'
    test_file = 'test.csv'
else :
    train_file = 'training_set.csv'
    test_file = 'test.csv'
    img_path = 'images/'

print(len(s))


csvreader = []
with open(train_file, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    csvreader = list(csvreader)

csvreader = csvreader[1:]
img_rows,img_cols,img_channels = 640,480,3
#print(csvreader)

imlist = []
print(len(csvreader))

label = numpy.empty((len(csvreader),4))
#print(label)
for i in range(len(csvreader)):
    imlist.append(csvreader[i][0])
    label[i][0] = float(csvreader[i][1])
    label[i][1] = float(csvreader[i][2])
    label[i][2] = float(csvreader[i][3])
    label[i][3] = float(csvreader[i][4])


img_rows,img_cols,img_channels =128,128,1

path1 = img_path

#creating training image set by resizing and converting to grayscale

immatrix = array([array(Image.open(path1 + im).resize((img_rows,img_cols)).convert('L')).flatten()
                   for im in imlist], 'f')



print(len(imlist))
print(label.shape)


X_train, X_test, y_train, y_test = train_test_split(immatrix, label, test_size=0.025, random_state=42)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalization of pixel values
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')





def iou_metric_bbox(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x1, y1, x2, y2]

    # AOG = Area of Groundtruth box
    AG = K.abs(K.transpose(y_true)[1] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose
                                                                             (y_true)[3] - K.transpose(y_true)[2] + 1)

    # AOP = Area of Predicted box
    AP = K.abs(K.transpose(y_pred)[1] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose
                                                                             (y_pred)[3] - K.transpose(y_pred)[2] + 1)

    # overlaps are the co-ordinates of intersection box
    area_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    area_1 = K.minimum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    area_2 = K.maximum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    area_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (area_1 -area_0 + 1) * (area_3 - area_2 + 1)

    # area of union of both boxes
    union = AG + AP - intersection

    # iou calculation
    iou = intersection / union

    #avoiding divide by zero
    if union == 0:
        union = 0.0001

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    return (iou)


def iou_bbox(y_true, y_pred):
    num_images = K.int_shape(y_pred)[-1]
    # print(y_pred.shape)
    if y_pred.shape[1] != 4:
        raise Exception(
            'BBox metric takes columns in the format. (x1,x2,y1,y2).'
            'Target shape should have 4 values in column. No of columns found: {} .Please consider changing metric function for this problem.'.format(
                y_pred.shape[1]))
    if y_true.shape[1] != 4:
        raise Exception(
            'BBox metric takes columns in the format. (x1,x2,y1,y2).'
            'Source shape should have 4 values in column. No of columns found: {} .Please consider changing metric function for this problem.'.format(
                y_pred.shape[1]))
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_images):
        total_iou = total_iou + iou_metric_bbox(y_true, y_pred)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_images






def getImageArr(path1,width,height):
    try:
        img = Image.open(path1)
        img = img.resize((width,height)).convert('L')
        img = np.array(img)
        img = img.astype(np.float32)
        img = img / 255.0

        img = img.reshape(width,height,1)
        #print(img.shape)
        return img
    except :
        return img


def lr_sch(epoch):
    # 200 total
    if epoch < 50:
        return 1e-3
    if 50 <= epoch < 400:
        return 1e-4
    if epoch >= 400:
        return 1e-5


lr_scheduler = keras.callbacks.LearningRateScheduler(lr_sch)
lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_iou_bbox', factor=0.2, patience=10, mode='max', min_lr=1e-3)

input_case = keras.layers.Input(shape=(img_rows, img_cols, img_channels))

def model(dropout) :
    start_neurons = 16
    x = Conv2D(64, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
              )(input_case)
    x = Conv2D(64, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
              )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(dropout)(x)

    # Block 2
    x = Conv2D(128, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
               )(x)
    x = Conv2D(128, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
               )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),)(x)
    x = Dropout(dropout)(x)

    # Block 3
    x = Conv2D(256, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
               )(x)
    x = Conv2D(256, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
               )(x)
    x = Conv2D(256, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
              )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(dropout)(x)
    x = Conv2D(512, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
              )(x)
    x = Conv2D(512, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
               )(x)
    x = Conv2D(512, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
              )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(dropout)(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
              )(x)
    x = Dropout(dropout)(x)
    x = Conv2D(512, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
               )(x)
    x = Conv2D(512, (3, 3),
               activation='elu',
               padding='same', kernel_initializer='he_normal',
               )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),)(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='elu')(x)

    x = Dense(4096, activation='elu',)(x)

    out = Dense(4,)(x)




    return out

print('model1')
l = iou_bbox

out1 = model(0.5)
out2 = model(0.6)
out3 = model(0.4)

out = keras.layers.concatenate([out1,out2,out3],axis=1)
out = Dense(4)(out)

model1 = keras.Model(input_case,out)
model1.compile(loss='logcosh', optimizer='adamax', metrics=['mse', l])

fname  = 'full4.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=fname, verbose=1, monitor='val_iou_bbox', mode='max',
                                               save_weights_only=True, save_best_only=True)
hist = model1.fit(X_train, y_train, batch_size=30, epochs=1, verbose=1, validation_data=(X_test, y_test),callbacks=[checkpointer,lr_reducer,lr_scheduler])


model1.load_weights(fname)

print('model4')

out1 = model(0.5)
out2 = model(0.5)
out3 = model(0.5)

out = keras.layers.concatenate([out1,out2,out3],axis=1)
out = Dense(4)(out)

model4 = keras.Model(input_case,out)
model4.compile(loss='logcosh', optimizer='adamax', metrics=['mse', l])

fname  = 'full5.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=fname, verbose=1, monitor='val_iou_bbox', mode='max',
                                               save_weights_only=True, save_best_only=True)
hist = model4.fit(X_train, y_train, batch_size=30, epochs=1, verbose=1, validation_data=(X_test, y_test),callbacks=[checkpointer,lr_reducer,lr_scheduler])


model4.load_weights(fname)

print('model2')

out4 = model(0.5)

model2 = keras.Model(input_case,out4)

model2.compile(loss='logcosh', optimizer='adamax', metrics=['mse', l])

fname  = 'full3.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=fname, verbose=1, monitor='val_iou_bbox', mode='max',
                                               save_weights_only=True, save_best_only=True)
hist = model2.fit(X_train, y_train, batch_size=100, epochs=1, verbose=1, validation_data=(X_test, y_test),callbacks=[checkpointer,lr_reducer,lr_scheduler])

model2.load_weights(fname)

print('model3')

out5 = model(0.6)

model3 = keras.Model(input_case,out5)

model3.compile(loss='logcosh', optimizer='adamax', metrics=['mse', l])

fname  = 'full34.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=fname, verbose=1, monitor='val_iou_bbox', mode='max',
                                               save_weights_only=True, save_best_only=True)
hist = model3.fit(X_train, y_train, batch_size=100, epochs=1, verbose=1, validation_data=(X_test, y_test),callbacks=[checkpointer,lr_reducer,lr_scheduler])

model3.load_weights(fname)



def clarify(n,m) :
    if n < 0 :
        return 0
    if n > m:
        return m
    return n


csvreader = []
with open(test_file, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    csvreader = list(csvreader)

csvreader = csvreader[1:]
#print(csvreader)

imlist = []
print(len(csvreader))

label = numpy.empty((len(csvreader),4))

for i in range(len(csvreader)):
    imlist.append(csvreader[i][0])



with open('result.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['image_name'] + ['x1'] + ['x2'] + ['y1'] + ['y2'])
    for im in imlist:
        img = getImageArr(path1 + im,img_rows,img_cols)
        img = img.reshape(1,img_rows,img_cols,img_channels)
        res1 = model1.predict(img)
        res2 = model2.predict(img)
        res3 = model3.predict(img)
        res4 = model4.predict(img)
        res = abs(res1[0] + res2[0] + res3[0] + res4[0])
        res = res / 4
        x1 = round(res[0])
        x2 = round(res[1])
        y1 = round(res[2])
        y2 = round(res[3])

        x1 = clarify(x1,640)
        x2 = clarify(x2,640)
        y1 = clarify(y1,480)
        y2 = clarify(y2,480)

        spamwriter.writerow([im]+[x1]+[x2]+[y1]+[y2])




