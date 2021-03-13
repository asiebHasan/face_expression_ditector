# importing necessary modules
import os as os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
print(tf.__version__)

# reading dataset(.csv)
df = pd.read_csv('fer2013.csv')
# print(df.info())
# print(df["Usage"].value_counts())
# print(df.head())

X_train, Y_train, X_test, Y_test = [],[],[],[]

# adding Training Data to train sets and testing data to testing sets
for index,row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val,'float32'))
            Y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val,'float32'))
            Y_test.append(row['emotion'])
    except:
        print("error at index:{index} and row:{row}")

# converting sets to np array for keras support only np array
X_train = np.array(X_train, 'float32')
Y_train = np.array(Y_train, 'float32')
X_test = np.array(X_test, 'float32')
Y_test = np.array( Y_test, 'float32' )

# Normalizing data betweed 0 and 1

X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test,axis=0)


num_features = 64   #later used for cov2d filter
num_labels = 7
batch_size = 64
epochs = 100
width,height = 48,48

# reshaping data as keras accept(48p*48p)
X_train = X_train.reshape(X_train.shape[0],width,height,1)
Y_train = np_utils.to_categorical(Y_train,num_classes=num_labels)

X_test =X_test.reshape(X_test.shape[0],width,height,1 )
Y_test = np_utils.to_categorical(Y_test,num_classes=num_labels)


# designing CNN(Convolution Neural Network)

# creating a model of linear stack of layers
model = Sequential()

# 1st layer convolution layer

# adding  2D convolution layer for producing a tensorflow output as we r using 2d images
model.add(Conv2D(num_features,kernel_size=(3,3),activation='relu', input_shape =(X_train.shape[1:])))
model.add(Conv2D(num_features,kernel_size=(3,3),activation='relu'))
# picking the maximum value by pool_size from the pool
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# randomly drop some data from layers so our model does not overfit
# Dropout rate 0.5
model.add(Dropout(0.5))

# 2nd layer layer convolution layer

model.add(Conv2D(num_features,(3,3),activation='relu'))
model.add(Conv2D(num_features,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

#3rd layer layer convolution layer
model.add(Conv2D(2*num_features,(3,3),activation='relu'))
model.add(Conv2D(2*num_features,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

# converting data into a 1D array and making it ready for next layer
model.add(Flatten())

model.add(Dense(2*2*2*2*num_features, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2*2*2*2*num_features, activation='relu'))
model.add(Dropout(0.2))

# creating a dense layer of 7 neurons
model.add(Dense(num_labels, activation='softmax'))

# printing summery of the model
model.summary()

# defining loss fuction and optimizer
model.compile(loss= categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])

# Training data
model.fit(X_train, Y_train, batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test,Y_test),
    shuffle = True)


# Saving the model as HDF5 format for next usage
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
