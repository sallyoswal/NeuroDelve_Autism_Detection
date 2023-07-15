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
from tensorflow.keras.layers import (Conv3D, Flatten, AveragePooling3D, Dense,
                                     BatchNormalization, Reshape, Dropout,)
from keras.models import Sequential, Input
from tensorflow.keras.losses import binary_crossentropy
from keras import backend as K
from tensorflow.saved_model import load
import keras
import pandas as pd

from model_3D_CAE import *

def ensemble_model():
    ae_ = AE(autoencoder)
    ae_.compile(optimizer = RMSprop(1e-4))
    # ae_.load_weights("/content/drive/MyDrive/Prediction of Autism /Models/Autoencoder_TF/ae_varied")
    encoder = ae_.ae_model.layers[0]
    encoder.trainable = False

    classifier = Sequential([
                            Input(shape = (43, 52, 43, 128)),

                            Conv3D(128, kernel_size= 3, activation= 'relu', data_format = "channels_last"),
                            BatchNormalization(),
                            AveragePooling3D(2),

                            #  Conv3D(2, kernel_size= 1, activation= 'relu'),
                            Conv3D(256, kernel_size= 3, activation= 'relu'),
                            BatchNormalization(),
                            AveragePooling3D(2),

                            #  Conv3D(256, kernel_size= 1, activation= 'relu'),
                            Conv3D(256, kernel_size= 3, activation= 'relu'),
                            BatchNormalization(),
                            AveragePooling3D(2),

                            #  keras.layers.GlobalAveragePooling3D(),
                            Flatten(),

                            Dense(512, activation='relu', kernel_regularizer= keras.regularizers.l1_l2),
                            Dropout(0.5),

                            Dense(1024, activation= 'relu', kernel_regularizer= keras.regularizers.l1_l2),
                            Dropout(0.5),

                            Dense(1, activation='sigmoid')
    ])

    return Sequential([encoder, classifier])

class Ensemble_Model(keras.Model):
  def __init__(self, ensemble_model, **kwargs):
    super(Ensemble_Model, self).__init__(**kwargs)

    self.ensemble_model_ = ensemble_model
    self.binary_cross_entropy_loss_tracker = keras.metrics.Mean(
              name="binary_crossentropy_loss"
              )
    self.accuracy_tracker = keras.metrics.Mean(
        name ='accuracy'
    )
    self.val_binary_cross_entropy_loss_tracker = keras.metrics.Mean(
        name = 'val_binary_crossentropy_loss'
        )
    self.val_accuracy_tracker = keras.metrics.Mean(
        name = 'val_accuracy'
    )
    self.test_binary_cross_entropy_tracker = keras.metrics.Mean(
        name ='test_binary_crossentropy_loss'
    )
    self.test_accuracy_tracker = keras.metrics.Mean(
        name = 'test_accuracy'
    )

  @property
  def metrics(self):
    return [self.binary_cross_entropy_loss_tracker, self.accuracy_tracker]

  def train_step(self, imgs, labels):
    with tf.GradientTape() as tape:
      predicted_labels = self.ensemble_model_(imgs)
      # print(predicted_labels.numpy())
      binary_crossentropy_loss = tf.math.reduce_mean(keras.losses.binary_crossentropy(labels, predicted_labels), axis = [0])
      accuracy = tf.keras.metrics.Accuracy()
      
    grads = tape.gradient(binary_crossentropy_loss, self.ensemble_model_.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.ensemble_model_.trainable_weights))

    self.binary_cross_entropy_loss_tracker.update_state(binary_crossentropy_loss)
    self.accuracy_tracker.update_state(labels, predicted_labels)

    
    return {"binary_crossentropy_loss" : self.binary_cross_entropy_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()}

  def valid_step(self, imgs, labels):
    predicted_labels = self.ensemble_model_(imgs)
    val_binary_crossentropy_loss = tf.math.reduce_mean(keras.losses.binary_crossentropy(labels, predicted_labels, from_logits= True), axis = [0])
    val_accuracy = tf.keras.metrics.Accuracy()

    self.val_binary_cross_entropy_loss_tracker.update_state(val_binary_crossentropy_loss)
    self.val_accuracy_tracker.update_state(labels, predicted_labels)
    
    return {"val_binary_crossentropy_loss" : self.val_binary_cross_entropy_loss_tracker.result(),
            "val_accuracy": self.val_accuracy_tracker.result()}
  
  def predict(self, imgs, labels):
    predicted_labels = self.ensemble_model_(imgs)
    test_binary_crossentropy_loss = tf.math.reduce_mean(keras.losses.binary_crossentropy(labels, predicted_labels, from_logits= True), axis = [0])
    test_accuracy = tf.keras.metrics.Accuracy()

    self.test_binary_cross_entropy_tracker.update_state(test_binary_crossentropy_loss)
    self.test_accuracy_tracker.update_state(labels, predicted_labels)

    return (predicted_labels, self.test_binary_cross_entropy_tracker.result(), self.test_accuracy_tracker.result())



def training_ensemble(model, train_dataset, valid_dataset, epochs = 20):
  start_time = time.time()
  history = pd.DataFrame(columns= ['binary_crossentropy_loss', 'accuracy', 'val_binary_crossentropy_loss', 'val_accuracy'])
  for epoch in range(epochs):
    print(f'Epoch {epoch}')

    for step, img_label in enumerate(train_dataset):
      img_batch, label_batch = img_label
      loss_metric = model.train_step(imgs = np.reshape(img_batch, (-1, 182, 218, 182, 1)), labels = np.reshape(label_batch, (BATCH_SIZE_ENSEMBLE, 1)))

      if step % 5 == 0:
        print(f"\tStep: {step}\n\t\tbinary_crossentropy_loss: {loss_metric['binary_crossentropy_loss'].numpy():.4f}, accuracy: {loss_metric['accuracy'].numpy():.4f}")
    
    print(f'At the end of epoch {epoch}:')
    print(f"\tbinary_crossentropy_loss: {loss_metric['binary_crossentropy_loss'].numpy():.4f}, accuracy: {loss_metric['accuracy'].numpy():.4f}")

    print(f'Time elapsed: {((time.time() - start_time) / 60):.4f}\n')

    for val_step, val_img_batch in enumerate(valid_dataset):
      val_img_batch, val_label_batch = val_img_batch
      val_loss_metric = model.valid_step(imgs = np.reshape(val_img_batch, (-1, 182, 218, 182, 1)), labels = np.reshape(val_label_batch, (BATCH_SIZE_ENSEMBLE, 1)))

    print(f"\tval_binary_crossentropy_loss: {val_loss_metric['val_binary_crossentropy_loss'].numpy():.4f}, val_accuracy: {val_loss_metric['val_accuracy'].numpy():.4f}")

    history = history.append({'binary_crossentropy_loss': loss_metric['binary_crossentropy_loss'].numpy(), 'accuracy': loss_metric['accuracy'].numpy(),
                              'val_binary_crossentropy_loss': val_loss_metric['val_binary_crossentropy_loss'].numpy(), 'val_accuracy': val_loss_metric['val_accuracy'].numpy()},
                              ignore_index= True)
    model.save_weights("/content/drive/MyDrive/Prediction of Autism /Models/Autoencoder_TF/ensemble_04.csv", save_format= 'tf')
    history.to_csv("/content/drive/MyDrive/Prediction of Autism /Models/Autoencoder_TF/ensemble_04.csv")
  end_time = time.time()
  print(f'Total time taken: {end_time - start_time}')
  
  return history

def prediction_ensemble(test_dataset = test_dataset):
  start_time = time.time()
  predicted_labels = []

  for img_number, img_batch in enumerate(test_dataset):
    imgs, labels = img_batch  
    pred_labels, test_binary_crossentropy_loss, test_accuracy = ensemble_model_.predict(np.reshape(imgs, (-1, 182, 218, 182, 1)), np.reshape(labels, (1, 1)))
    predicted_labels.append(pred_labels)

  print(f"\t\n\t\ttest_binary_crossentropy_loss: {test_binary_crossentropy_loss.numpy():.4f}, test_accuracy: {test_accuracy.numpy():.4f}")

  print(f'Time elapsed: {((time.time() - start_time) / 60):.4f}')
  
  return (predicted_labels, test_binary_crossentropy_loss, test_accuracy)



BATCH_SIZE_ENSEMBLE = 8
SHUFFLE_BUFFER_ENSEMBLE = 10

if __name__ == '__main__':
    _ensemble_model_ = ensemble_model()
    ensemble_model_ = Ensemble_Model(_ensemble_model)
    ensemble_model_.compile(optimizer= RMSprop(1e-4), loss= 'binary_crossentropy', metrics = ['accuracy'])

    ensemble_model_.load_weights('/content/drive/MyDrive/Prediction of Autism /Models/Autoencoder_TF/ensemble_04.csv')

    #To train:
    # history = training_ensemble(ensemble_model_, train_dataset_ensemble, valid_dataset_ensemble)
        


