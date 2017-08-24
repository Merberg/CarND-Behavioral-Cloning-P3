# Code to read in lines from the .csv file
import os
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

sampleLines = []
with open('examples/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #Skip the header
        if line[3].find('steering') == -1:
            sampleLines.append(line)

training_samples, validation_samples = train_test_split(sampleLines, test_size=0.2)
print("{} Training Samples".format(len(training_samples)))
print("{} Validation Samples".format(len(validation_samples)))

# A Generator to read in the data in partitions
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'examples/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Generators for training and validation
training_generator = generator(training_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Code for compiling and training the model
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Cropping2D, Dense, Flatten
from keras.layers import Input, Lambda, MaxPooling2D

# Model based on the Nvidia architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(training_generator,
                    samples_per_epoch=len(training_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=3)

model.save('model.h5')
