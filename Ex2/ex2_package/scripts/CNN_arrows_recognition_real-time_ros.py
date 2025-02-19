#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection of arrows for robot control using CNN and ROS

"""

# Load required packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image                                                            
import glob
import os
import random

# Load required packages to connect to ROS 
import rospy
from geometry_msgs.msg import Twist


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# List of arrow classes
namesList = ['up', 'down', 'left', 'right']

# Load the testing database
imageFolderPath = r'/rosdata/ros_ws_loc/src/ex2_package/scripts/Database_arrows'
imageFolderTrainingPath = imageFolderPath + r'/train'
imageFolderTestingPath = imageFolderPath + r'/validation'
imageTrainingPath = []
imageTestingPath = []

for i in range(len(namesList)):
    trainingLoad = imageFolderTrainingPath + '//' + namesList[i] + '/*.jpg'
    testingLoad = imageFolderTestingPath + '//' + namesList[i] + '/*.jpg'
    imageTrainingPath = imageTrainingPath + glob.glob(trainingLoad)
    imageTestingPath = imageTestingPath + glob.glob(testingLoad)

updateImageSize = [128, 128]
tempImg = Image.open(imageTrainingPath[0]).convert('L')
tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
[imWidth, imHeight] = tempImg.size

# Create space to load testing images
x_test = np.zeros((len(imageTestingPath), imHeight, imWidth, 1))

# Load testing images  
for i in range(len(x_test)):
    tempImg = Image.open(imageTestingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_test[i, :, :, 0] = np.array(tempImg, 'f')
    
# Create space to load testing labels
y_test = np.zeros((len(x_test),));

# Load testing labels
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTestingPath)/len(namesList))):
        y_test[countPos,] = i
        countPos = countPos + 1
        
# Convert testing labels to one-hot format
y_test = tf.keras.utils.to_categorical(y_test, len(namesList));

# Load the CNN model previously trained
model = load_model(r'/rosdata/ros_ws_loc/src/ex2_package/scripts/arrows_training_model.h5')

# Initalise node
rospy.init_node('arrow_recognition')
rospy.Rate(10)

# ROS Publisher for controlling the robot
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

# Function to control the robot based on arrow prediction
def control_robot(prediction):
    twist = Twist()

    # Set default linear and angular speeds
    twist.linear.x = 0.2
    twist.angular.z = 0.0

    # Adjust speeds based on arrow prediction
    if prediction == 0:  # up
        twist.linear.x = 0.3
        twist.angular.z = 0.0
    elif prediction == 1:  # down
        twist.linear.x = -0.3
        twist.angular.z = 0.0
    elif prediction == 2:  # left
        twist.linear.x = 0.3
        twist.angular.z = -0.4
    elif prediction == 3:  # right
        twist.linear.x = 0.3
        twist.angular.z = 0.4
        
    # Publish the twist message
    pub.publish(twist)

# Loop to read input digits
while(True):
    # Select an arrow image randmonly from the test dataset
    random_index = random.randint(0, len(imageTestingPath) - 1)
    arrow_path = imageTestingPath[random_index]
    
    # Select the corresponding class label
    actual_label = arrow_path.split('/')[-2]
    
    if actual_label == 'right':
        actual_label = 'left'
    elif actual_label == 'left':
        actual_label = 'right'
    
    # Prepare the image with the correct shape for the CNN
    test_img = Image.open(arrow_path).convert('L')
    test_img.thumbnail(updateImageSize, Image.ANTIALIAS)
    test_img_array = np.array(test_img, 'f')
    test_img_array = np.expand_dims(test_img_array, axis=-1)
    test_img_array = np.expand_dims(test_img_array, axis=0)
    
    # Use the input image for prediction with the pre-trained model
    prediction = np.argmax(model.predict(test_img_array))
    
    if prediction == 2:
        prediction = 3
    elif prediction == 3:
        prediction = 2
    
    # Display the actual and predicted images 
    plt.imshow(test_img_array[0, :, :, 0], cmap='gray')
    plt.title(f'Actual: {actual_label} ---- Predicted: {namesList[prediction]}')
    plt.show()
    
    # Use the predicted output to control a mobile robot in CoppeliaSim via ROS
    control_robot(prediction)
    
    # Show in the terminal (or plots) the actual and predicted arrow
    print(f"Actual: {actual_label} ---- Predicted: {namesList[prediction]}")
    
    # Allow the robot to move for some seconds before loading the next arrow
    rospy.sleep(2)
    
    # Repeat the process until stopped by the user
    user_input = input("Press Enter to continue or 'q' to quit: ")
    if user_input.lower() == 'q':
        break
    
print('OK')
