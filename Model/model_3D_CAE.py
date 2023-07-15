import os
import cv2
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from tensorflow.data import Dataset
import time

# %tensorflow_version 2.x #If Colab
import tensorflow as tf
from keras.optimizers import RMSprop
from tensorflow.keras.layers import (Conv3D, Flatten, AveragePooling3D, Conv3DTranspose,
                                     BatchNormalization, Reshape, Dropout, UpSampling3D)
from keras.models import Sequential, Input
from tensorflow.keras.losses import MeanSquaredError
from keras import backend as K
from tensorflow.saved_model import load
import keras
import pandas as pd


'''
Architecture
Encoder + Decoder
'''


encoder = Sequential([
                      Input((182, 218, 182, 1)),

                      Conv3D(8, kernel_size= 7, padding= 'valid', activation= 'elu', data_format = "channels_last"),
                      Conv3D(8, kernel_size= 5, padding= 'valid', activation= 'elu'),
                      BatchNormalization(),
                      AveragePooling3D(2),

                      Conv3D(16, kernel_size = 3, padding = 'same', activation= 'elu'),
                      Conv3D(32, kernel_size = 3, padding = 'same', activation= 'elu'),
                      BatchNormalization(),
                      AveragePooling3D(2),

                      Conv3D(64, kernel_size= 3, padding= 'same', activation= 'elu'),
                      Conv3D(64, kernel_size= 3, padding= 'same', activation= 'elu'),
                      BatchNormalization(),
                    
                      Conv3D(128, kernel_size= 3, padding = 'same', activation = 'elu'),
                      Conv3D(128, kernel_size= 3, padding = 'same', activation= 'elu'),
                      BatchNormalization()
])

decoder = Sequential([
                      Input(shape = (43, 52, 43, 128)),

                      Conv3D(128, kernel_size= 3, padding = 'same', activation = 'elu'),
                      Conv3D(128, kernel_size= 3, padding = 'same', activation= 'elu'),
                      Conv3D(64, kernel_size= 1, padding = 'same', activation= 'elu'),
                      BatchNormalization(),

                      Conv3D(64, kernel_size= 3, padding = 'same', activation = 'elu'),
                      Conv3D(64, kernel_size= 1, padding = 'same', activation= 'elu'),
                      BatchNormalization(),
                      UpSampling3D(2),

                      Conv3D(32, kernel_size= 1, padding = 'same', activation= 'elu'),
                      Conv3D(32, kernel_size= 3, padding = 'same', activation = 'elu'),
                      BatchNormalization(),

                      Conv3D(16, kernel_size= 1, padding = 'same', activation= 'elu'),
                      Conv3D(16, kernel_size= 3, padding = 'same', activation= 'elu'),
                      BatchNormalization(),
                      UpSampling3D(2),
                      
                      Conv3D(16, kernel_size= 3, padding = 'same', activation = 'elu'),
                      Conv3D(1, kernel_size= 1, padding = 'same', activation = 'elu'),
                      Conv3DTranspose(1, kernel_size= 5, activation = 'elu', name = 'see'),
                      BatchNormalization(),

                      Conv3DTranspose(1, kernel_size= 7, padding = 'valid', activation = 'elu'),
                      Conv3D(1, kernel_size= 3, padding = 'same', activation = 'sigmoid'),
                      BatchNormalization(),

])

'''
Subclassing to define train_step, valid_step & predict. Also used for defining epoch behaviour.
'''

class AE(keras.Model):
  def __init__(self, ae, **kwargs):
    super(AE, self).__init__(**kwargs)
    self.ae_model = ae
    self.reconstruction_loss_tracker = keras.metrics.Mean(
              name="mse_loss"
          )
    self.recon_val_mse_loss_tracker = keras.metrics.Mean(
        name = 'val_mse_loss'
        )
    self.test_mse_loss_tracker = keras.metrics.Mean(
        name = 'test_mse_loss'
    )

  @property
  def metrics(self):
    return [self.reconstruction_loss_bc_tracker, self.reconstruction_loss_tracker]

  @tf.function
  def train_step(self, data):
    with tf.GradientTape() as tape:
      reconstruction = self.ae_model(data)
      reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction)
      
    grads = tape.gradient(reconstruction_loss, self.ae_model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.ae_model.trainable_variables))
    
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)

    return {"mse_loss" : self.reconstruction_loss_tracker.result()} 

  @tf.function
  def valid_step(self, data):
    reconstruction = self.ae_model(data)
    reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction)

    self.recon_val_mse_loss_tracker.update_state(reconstruction_loss)
    return {"val_mse_loss" : self.recon_val_mse_loss_tracker.result()} 
  
  @tf.function
  def predict(self, data):
    with tf.GradientTape() as tape:
      reconstruction = self.ae_model(data)
      reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction)
      self.test_mse_loss_tracker.update_state(reconstruction_loss)

    return (reconstruction, self.test_mse_loss_tracker.result())

'''
Functions to iter through the TFRecord datasets
'''
def _parse_function_labels(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        "label": tf.io.FixedLenFeature([], tf.int64)}
    
    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
    # Turn your saved image string into an array
    parsed_features['image'] = tf.io.decode_raw(
        parsed_features['image'], tf.float32)
    
    return parsed_features['image'], parsed_features["label"]

  
def create_dataset_labels(filepath):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function_labels, num_parallel_calls=8)
    dataset = dataset.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder = True)
    return dataset

'''
Training on 650 Samples
'''

# Saves model weights at every epoch and, returns history of mse_loss and val_mse_loss
def training(model, train_dataset = train_dataset, valid_dataset = valid_dataset, epochs = EPOCHS):
  start_time = time.time()
  history = pd.DataFrame(columns= ['mse_loss', 'val_mse_loss'])
  for epoch in range(EPOCHS):
    print(f'Epoch {epoch}')

    for step, img_batch in enumerate(train_dataset):
      imgs, _ = img_batch
      loss_metric = model.train_step(np.reshape(imgs, (-1, 182, 218, 182, 1)))

      if step % 5 == 0:
        print(f"\tStep: {step}\n\t\tmse_loss: {loss_metric['mse_loss'].numpy():.4f}")

    print(f'Time elapsed: {((time.time() - start_time) / 60):.4f}')

    for val_step, val_img_batch in enumerate(valid_dataset):
      val_imgs, _ = val_img_batch
      val_loss_metric = model.valid_step(np.reshape(val_imgs, (-1, 182, 218, 182, 1)))

    print(f"\tval_mse_loss: {val_loss_metric['val_mse_loss'].numpy():.4f}")

    history = history.append({'mse_loss': loss_metric['mse_loss'].numpy(), 
                              'val_mse_loss': val_loss_metric['val_mse_loss'].numpy()},
                              ignore_index= True)
    
    #Saving the model weights & its history after every epoch:
    model.save_weights("/content/drive/MyDrive/Prediction of Autism /Models/Autoencoder_TF/ae_varied", save_format= 'tf') #CHANGE FILE PATH
    history.to_csv("/content/drive/MyDrive/Prediction of Autism /Models/Autoencoder_TF/ae_diff.csv") #CHANGE FILE PATH

  end_time = time.time()
  print(f'Total time taken: {(end_time - start_time) / 60}')
  return history

'''
Prediction function: returns limited reconstructed images and error.
'''

def prediction(test_dataset = test_dataset):
  start_time = time.time()
  reconstructed_imgs = []

  for img_number, img_batch in enumerate(test_dataset):
    imgs, labels = img_batch  
    if img_number < 2:
      reconstruction, test_mse_loss = ae_different.predict(np.reshape(imgs, (-1, 182, 218, 182, 1)))
      reconstructed_imgs.append(reconstruction)
    else:
      reconstruction, test_mse_loss = ae_different.predict(np.reshape(imgs, (-1, 182, 218, 182, 1)))

  print(f"\t\n\t\ttest_mse_loss: {test_mse_loss.numpy():.4f}")

  print(f'Time elapsed: {((time.time() - start_time) / 60):.4f}')
  
  return (reconstructed_imgs, test_mse_loss)


SHUFFLE_BUFFER = 6
BATCH_SIZE = 4
EPOCHS = 10
TRAINING_SAMPLES = 680
VALID_SAMPLES = 168
TEST_SAMPLES = 102

if __name__ === "__main__":    
    #Creation of train and validation set using above described parse functions:
    train_dataset = create_dataset_labels("/content/drive/MyDrive/Prediction of Autism /Data/TFRecorder/Shuffled/Train_img_lab_shuffled.tfrecord")
    valid_dataset = create_dataset_labels("/content/drive/MyDrive/Prediction of Autism /Data/TFRecorder/Shuffled/Valid_img_lab_shuffled.tfrecord")
    
    autoencoder = Sequential([encoder, decoder])
    autoencoder = AE(autoencoder)
    autoencoder.compile(optimizer = RMSprop(1e-4))
    autoencoder.load_weights("/content/drive/MyDrive/Prediction of Autism /Models/Autoencoder_TF/ae_varied")


    with tf.device('/device:GPU:0'):
        history = training(autoencoder)




