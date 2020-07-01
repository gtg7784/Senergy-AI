from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU

pd.options.mode.chained_assignment = None
EPOCHS = 1000

# dataset
# Enough,Hours,PhoneReach,PhoneTime,Tired,Breakfast
# Enough = Do you think that you get enough sleep?
# Hours = On average, how many hours of sleep do you get on a weeknight?
# PhoneReach = Do you sleep with your phone within arms reach?
# PhoneTime = Do you use your phone within 30 minutes of falling asleep?
# Tired = On a scale from 1 to 5, how tired are you throughout the day? (1 being not tired, 5 being very tired)
# Breakfast = Do you typically eat breakfast?

def pre_processing(dataset):  
  print(dataset.isna().sum())

  dataset = dataset.dropna()

  Enough = dataset.pop('Enough')
  phoneTime = dataset.pop('PhoneTime')
  phoneReach = dataset.pop('PhoneReach')
  breakfast = dataset.pop('Breakfast')
  dataset['Enough'] = Enough == 'Yes'
  dataset['PhoneTime'] = phoneTime == 'Yes'
  dataset['PhoneReach'] = phoneReach == 'Yes'
  dataset['Breakfast'] = breakfast == 'Yes'

  print(dataset.tail())

  return dataset

# min max normalization
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
  
def model():
  model = keras.Sequential()
  model.add(layers.Dense(64, input_shape=[len(train_dataset.keys())]))
  model.add(LeakyReLU(alpha=0.1))
  model.add(layers.Dense(64))
  model.add(LeakyReLU(alpha=0.1))
  model.add(layers.Dense(1))

  optimizer = keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
                
  return model

# Visualization with plt
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Tired]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Tired^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


def convert_nparray(data):
  return np.asarray(data).astype(np.float32)

if __name__ == "__main__":
  data = pd.read_csv('./data/SleepStudyData.csv')
  
  dataset = pre_processing(data)

  train_dataset = dataset.sample(frac=0.8,random_state=0)
  test_dataset = dataset.drop(train_dataset.index)

  # Visualization
  # sns.pairplot(train_dataset[["Hours", "PhoneReach", "PhoneTime", "Tired"]], diag_kind="kde")

  # Hours statistics
  train_stats = train_dataset.describe()
  train_stats.pop("Tired")
  train_stats = train_stats.transpose()
  print(train_stats)

  train_labels = train_dataset.pop('Tired')
  test_labels = test_dataset.pop('Tired')

  model = model()
  # check the model summary
  model.summary()

  # patience 매개변수는 성능 향상을 체크할 에포크 횟수입니다
  early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

  example_batch =  convert_nparray(train_dataset[:10])
  example_result = model.predict(example_batch)
  print(example_result)

  history = model.fit(
    convert_nparray(train_dataset), train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[early_stop])

  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  print(hist.tail())

  plot_history(history)

  loss, mae, mse = model.evaluate(convert_nparray(test_dataset), test_labels, verbose=2)

  print("테스트 세트의 평균 절대 오차: {:5.2f} Tired".format(mae))

  test_predictions = model.predict(convert_nparray(test_dataset)).flatten()

  plt.scatter(test_labels, test_predictions)
  plt.xlabel('True Values [Tired]')
  plt.ylabel('Predictions [Tired]')
  plt.axis('equal')
  plt.axis('square')
  plt.xlim([0,plt.xlim()[1]])
  plt.ylim([0,plt.ylim()[1]])
  _ = plt.plot([-100, 100], [-100, 100])
  plt.show()

  error = test_predictions - test_labels
  plt.hist(error, bins = 25)
  plt.xlabel("Prediction Error [Tired]")
  _ = plt.ylabel("Count")
  plt.show()