from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# dataset
# Enough,Hours,PhoneReach,PhoneTime,Tired,Breakfast
# Enough = Do you think that you get enough sleep?
# Hours = On average, how many hours of sleep do you get on a weeknight?
# PhoneReach = Do you sleep with your phone within arms reach?
# PhoneTime = Do you use your phone within 30 minutes of falling asleep?
# Tired = On a scale from 1 to 5, how tired are you throughout the day? (1 being not tired, 5 being very tired)
# Breakfast = Do you typically eat breakfast?

def preProcessing(dataset):  
  print(dataset.isna().sum())

  dataset = dataset.dropna()

  Enough = dataset.pop('Enough')
  phoneTime = dataset.pop('PhoneTime')
  phoneReach = dataset.pop('PhoneReach')
  Breakfast = dataset.pop('Breakfast')
  dataset['Enough'] = Enough == 'Yes'
  dataset['PhoneTime'] = phoneTime == 'Yes'
  dataset['phoneReach'] = phoneReach == 'Yes'
  dataset['Breakfast'] = Breakfast == 'Yes'
  
  return dataset
  
def model():
  model = keras.Sequential([
    layers.Dense(64, activation='leaky_relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
                
  return model

if __name__ == "__main__":
  data = pd.read_csv('./data/SleepStudyData.csv')
  
  dataset = preProcessing(data)

  train_dataset = dataset.sample(frac=0.8,random_state=0)
  test_dataset = dataset.drop(train_dataset.index)

  sns.pairplot(train_dataset[["Hours", "PhoneReach", "PhoneTime", "Tired"]], diag_kind="kde")
