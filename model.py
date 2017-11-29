
# coding: utf-8

# In[5]:


import tensorflow as tf
import numpy as np
import random
import csv
import cv2 
import json
import h5py

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model, model_from_json
from keras.regularizers import l2

WIDTH = 200
HEIGHT = 66

def get_csv_data(log_file):
    """
    Reads a csv file and returns two lists separated into examples and labels.
    :param log_file: The path of the log file to be read.
    """
    image_names, steering_angles = [], []
    # Steering offset used for left and right images
    steering_offset = 0.2
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for center_img, left_img, right_img, angle, _, _, _ in reader:
            angle = float(angle)
            image_names.append([center_img.strip(), left_img.strip(), right_img.strip()])
            steering_angles.append([angle, angle+steering_offset, angle-steering_offset])

    return image_names, steering_angles


def generate_batch(X_train, y_train, batch_size=64):
    """
    Return two numpy arrays containing images and their associated steering angles.
    :param X_train: A list of image names to be read in from data directory.
    :param y_train: A list of steering angles associated with each image.
    :param batch_size: The size of the numpy arrays to be return on each pass.
    """
    images = np.zeros((batch_size, HEIGHT, WIDTH, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    while True:
        straight_count = 0
        for i in range(batch_size):
            # Select a random index to use for data sample
            sample_index = random.randrange(len(X_train))
            image_index = random.randrange(len(X_train[0]))
            angle = y_train[sample_index][image_index]
            # Limit angles of less than absolute value of .1 to no more than 1/2 of data
            # to reduce bias of car driving straight
            if abs(angle) < .1:
                straight_count += 1
            if straight_count > (batch_size * .5):
                while abs(y_train[sample_index][image_index]) < .1:
                    sample_index = random.randrange(len(X_train))
            # Read image in from directory, process, and convert to numpy array
            image = cv2.imread('data/' + str(X_train[sample_index][image_index]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = process_image(image)
            image = np.array(image, dtype=np.float32)
            # Flip image and apply opposite angle 50% of the time
            if random.randrange(2) == 1:
                image = cv2.flip(image, 1)
                angle = -angle
            images[i] = image
            angles[i] = angle
        yield images, angles





def normalize(image):
    """
    Returns a normalized image with feature values from -1.0 to 1.0.
    :param image: Image represented as a numpy array.
    """
    return image / 127.5 - 1.


def process_image(image):
    """
    Returns an image after applying several preprocessing functions.
    :param image: Image represented as a numpy array.
    """
    #change image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    #remove the top 60 pixels (landscape) and the bottom 25 pixels (hood) from each image
    image = image[60:-25,:]
    
    #scale the image down to speed up training
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return image


def get_model():
    """
    Model based on Nvidia paper
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    
    model = Sequential()
    
    model.add(Lambda(normalize, input_shape=(HEIGHT, WIDTH, 3)))

    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    return model    


# In[6]:


# Get the training data from log file, shuffle, and split into train/validation datasets
X_train, y_train = get_csv_data('./data/driving_log.csv')
X_train, y_train = shuffle(X_train, y_train, random_state=14)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=14)


# In[ ]:


# Get model, print summary, and train using a generator
model = get_model()
model.summary()
model.fit_generator(generate_batch(X_train, y_train), samples_per_epoch=4800, nb_epoch=5, validation_data=generate_batch(X_validation, y_validation), nb_val_samples=64)


# In[6]:


print('Saving model weights and configuration file.')
# Save model weights
#print("Weights Saved")
json_string = model.to_json()
with open('model.json', 'w') as jsonfile:
    json.dump(json_string, jsonfile)
model.save("model.h5")
print("Model Saved")


# In[ ]:




