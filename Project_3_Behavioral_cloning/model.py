import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Input keras commands
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, Lambda, SpatialDropout2D
from keras.optimizers import Adam

DATA_PATH = "data"
LABEL_PATH = os.path.join(DATA_PATH, "driving_log.csv")
Batch_size = 64
Epochs = 50

# Define function to drop small steering data randomly
def drop_small_steering_data_ramdomly(data):
    index = data[abs(data['steer']) < .05].index.tolist()
    rows = [i for i in index if np.random.randint(10) < 8]
    data_drop = data.drop(data.index[rows])
    print("Dropped %s rows with low steering" % (len(rows)))
    return data_drop

# Cropping images
def cropping_image(img):
     return img[60:135, :]

# Cropping image from path folder
def Cropping_image_from_path(img_path):
     return cropping_image(plt.imread(img_path))

# Get batch size from data
def get_batch_size(data, batch_size):
     return data.sample(n=batch_size)

# Select left,center,right's images and steering angle randomly
def get_images_and_steering_angle_randomly(data, value, data_path):
    random = np.random.randint(4)
    if (random == 0):
        img_path = data['left'][value].strip()
        shift_angle = .25
    if (random == 1 or random == 3):
        img_path = data['center'][value].strip()
        shift_angle = 0.
    if (random == 2):
        img_path = data['right'][value].strip()
        shift_angle = -.25
    img = Cropping_image_from_path(os.path.join(data_path, img_path))
    steer_ang = float(data['steer'][value]) + shift_angle
    return img, steer_ang

# Get translated_image and it's steering angle
def translated_image_and_its_steering_angle(image, steer):
    translated_range = 100
    tr_x = translated_range * np.random.uniform() - translated_range / 2
    steer_ang = steer + tr_x / translated_range * 2 * .2
    tr_y = 0
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, M, (320, 75))
    return image_tr, steer_ang

# Get images for training
def images_for_training(data,batch_size,data_path):
    while 1:
        batch = get_batch_size(data, batch_size)
        features = np.empty([batch_size, 75, 320, 3])
        labels = np.empty([batch_size, 1])
        for i, value in enumerate(batch.index.values):
            img,steer_ang = get_images_and_steering_angle_randomly(data, value, data_path)
            img = img.reshape(img.shape[0], img.shape[1], 3)
            img, steer_ang = translated_image_and_its_steering_angle(img, steer_ang)
            random = np.random.randint(1)
            if (random == 0):
                img, steer_ang = np.fliplr(img), -steer_ang
            features[i] = img
            labels[i] = steer_ang
            yield np.array(features), np.array(labels)


# Get images for validation
def images_for_validation(data,data_path):
     while 1:
        for i in range(len(data)):
            img_path = data['center'][i].strip()
            img = Cropping_image_from_path(os.path.join(data_path, img_path))
            img = img.reshape(1, img.shape[0], img.shape[1], 3)
            steer_ang = data['steer'][i]
            steer_ang = np.array([[steer_ang]])
            yield img, steer_ang


# Resize images
def resize_images(img):
    return tf.image.resize_images(img, (66, 200))

# Based on the paper: "End to End Learning for Self-Driving Cars"
# Define the CNN structure by keras
def CNN_structure(input_shape):
    model = Sequential()
    model.add(Lambda(resize_images, input_shape=input_shape))
    model.add(Lambda(lambda x: x / 255. - 0.5))
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2, 2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2, 2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu"))
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
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    return model

# Read driving_log.csv file
csv_df = pd.read_csv(LABEL_PATH, index_col=False)
csv_df.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']

# Shuffle randomly
csv_df = csv_df.sample(n=len(csv_df))

# Training data
training_count = int(0.8 * len(csv_df))
training_data = csv_df[:training_count].reset_index()

# Validation data
validation_data = csv_df[training_count:].reset_index()

# Drop small steering data randomly
training_data =drop_small_steering_data_ramdomly(training_data)

# Cropping image
image_crop = Cropping_image_from_path(os.path.join(DATA_PATH, training_data['center'].iloc[909].strip()))

# Train the CNN
model = CNN_structure(image_crop.shape)
samples_per_epoch = int(len(training_data) / Batch_size) * Batch_size
nb_val_samples = len(validation_data)
values = model.fit_generator(images_for_training(training_data, Batch_size, DATA_PATH),
                             samples_per_epoch=samples_per_epoch, nb_epoch=Epochs,
                             validation_data= images_for_validation(validation_data, DATA_PATH),
                             nb_val_samples=len(validation_data))
model.save('model.h5')