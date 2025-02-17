#essential libraries for data loading and preprocessing
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os
# image dir paths
train_dir = 'FER2013/train'
test_dir = 'FER2013/test'
img_size =(48,48) #size of the image
batch_size = 32 #no of images to be loaded at once

#normalising pixel values
train_datagen = ImageDataGenerator(
    rescale=1./255,#normalising pixel values
    rotation_range=30,#rotating the image
    width_shift_range=0.2,#shifting the image
    height_shift_range=0.2,#shifting the image
    shear_range=0.2,#shearing the image
    zoom_range=0.2,#zooming the image
    horizontal_flip=True,#flipping the image
    brightness_range=[0.7,1.3],#changing the brightness of the image
    preprocessing_function= lambda x: x + np.random.normal(0,0.1,x.shape)#adding noise to the image
    )
test_datagen = ImageDataGenerator(
    rescale=1./255,
    )
# Loading the images from the directories using flow_from_directory method from ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,#size of the image
    batch_size=batch_size,#no of images to be loaded at once
    color_mode='grayscale', #converts the image to grayscale
    class_mode='categorical',#categorical labels;one hot encoding
    shuffle=True
    )
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
    )

# Checking dataset correctness
print("Class indices (should be 6 classes only):", train_generator.class_indices)
# Displaying a few images
images, labels = next(train_generator)
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i].reshape(48, 48), cmap='gray')
    plt.axis('off')
plt.show()