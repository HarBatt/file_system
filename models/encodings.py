import tensorflow as tf
import keras
import numpy as np
import os
import cv2
from skimage.transform import rescale, resize, downscale_local_mean
from pre_processing import Processing

image_path = 'C:/Users/Harsha/Desktop/President_Barack_Obama.jpg'
new_image_log = 'C:/Users/Harsha/Desktop/updated.jpg'
en = tf.keras.models.load_model('enc.h5')

image = Processing(image_path)
image.resize_save(new_image_log)
x = cv2.imread(image_path)
updated_image = cv2.resize(x, (128, 128),interpolation = cv2.INTER_NEAREST)
new_image = np.expand_dims(updated_image, axis=0)
print(en.predict(new_image)[0])
