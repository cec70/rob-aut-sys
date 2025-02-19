# -*- coding: utf-8 -*-
"""
Detection of arrows for robot control using CNN and ROS

"""

# Load required packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

from PIL import Image                                                            
import glob

import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


np.random.seed(7)
    

# List of arrow classes
namesList = ['up', 'down', 'left', 'right']

# Folder names of train and testing images
imageFolderPath = os.getcwd()
imageFolderTrainingPath = imageFolderPath + r'/Database_arrows/train'
imageFolderTestingPath = imageFolderPath + r'/Database_arrows/validation'
imageTrainingPath = []
imageTestingPath = []

# Full path to training and testing images
for i in range(len(namesList)):
    trainingLoad = imageFolderTrainingPath + '//' + namesList[i] + '/*.jpg'
    testingLoad = imageFolderTestingPath + '//' + namesList[i] + '/*.jpg'
    imageTrainingPath = imageTrainingPath + glob.glob(trainingLoad)
    imageTestingPath = imageTestingPath + glob.glob(testingLoad)
    
# Print number of images for training and testing
print(len(imageTrainingPath))
print(len(imageTestingPath))

# Resize images to speed up training process
updateImageSize = [128, 128]
tempImg = Image.open(imageTrainingPath[0]).convert('L')
tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
[imWidth, imHeight] = tempImg.size

# Create space to load training images
x_train = np.zeros((len(imageTrainingPath), imHeight, imWidth, 1))
# Create space to load testing images
x_test = np.zeros((len(imageTestingPath), imHeight, imWidth, 1))

# Load training images
for i in range(len(x_train)):
    tempImg = Image.open(imageTrainingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_train[i, :, :, 0] = np.array(tempImg, 'f')
    
# Load testing images
for i in range(len(x_test)):
    tempImg = Image.open(imageTestingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_test[i, :, :, 0] = np.array(tempImg, 'f')

# Create space to load training labels
y_train = np.zeros((len(x_train),));
# Create space to load testing labels
y_test = np.zeros((len(x_test),));

# Load training labels
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTrainingPath)/len(namesList))):
        y_train[countPos,] = i
        countPos = countPos + 1
    
# Load testing labels
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTestingPath)/len(namesList))):
        y_test[countPos,] = i
        countPos = countPos + 1
        
# Convert training labels to one-hot format
y_train = tf.keras.utils.to_categorical(y_train, len(namesList));
# Convert testing labels to one-hot format
y_test = tf.keras.utils.to_categorical(y_test, len(namesList));
        
# Create the CNN model
model = Sequential()
# First feature layer composed of a convolution, batch normalization and maxpooling layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 128, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
# Second feature layer composed of a convolution, batch normalization and maxpooling layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
# Third feature layer composed of a convolution, batch normalization and maxpooling layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

# Flatten or linearised the information from the previous layer
model.add(Flatten())
# Fully connected layer with 64 neurons and relu activation
model.add(Dense(64, activation='relu'))
# Fully connected layer with 128 neurons and relu activation
model.add(Dense(128, activation='relu'))
# Fully connected layer with 4 neurons and softmax activation
model.add(Dense(4, activation='softmax'))

# Define learning method, metrics and loss analysis
sgd = tf.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trains the model with a define number of epochs, batch size and validation
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the performance of the train model using the test dataset
score = model.evaluate(x_test, y_test, batch_size=5)

# Saves the trained model
idFile = 'arrows_training_model';
modelPath = idFile + '.h5';
model.save(modelPath);

# Displays the recognition accuracy
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# Plot accuracy curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plot loss curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print('OK')