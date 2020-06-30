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

  phoneTime = dataset.pop('PhoneTime')
  phoneReach = dataset.pop('PhoneReach')
  Breakfast = dataset.pop('Breakfast')
  dataset['PhoneTime'] = phoneTime == 'Yes'
  dataset['phoneReach'] = phoneReach == 'Yes'
  dataset['Breakfast'] = Breakfast == 'Yes'
  
  return dataset
  
def model():
  print('a')

if __name__ == "__main__":
  data = pd.read_csv('./data/SleepStudyData.csv')
  
  dataset = preProcessing(data)