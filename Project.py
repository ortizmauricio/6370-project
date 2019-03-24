#Frank Saldivar, Mauricio Ortiz
#CSCI 6370 - Machine Learning, Spring 2019
#Dr. kim
#
#Project -- Stock Price Movement prediction using Tensorflow
#Goal: Compare and contrast differing ML techniques with a dataset
#       of stock prices from 3-15-1996 to 3-14-2019, 
#       target prediction date: **3-15-2019**
#
#Notes: Training Set: 3-15-1996 -> 3-15-2016
#       Testing Set: 3-16-2016 -> 3-14-2019

#importing pkgs
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import numpy as np
import pandas as pd

from numpy import genfromtxt

#reading in testing and training data
trainingData = pd.read_csv('training_SPY_prices_1996_2016.csv', delimiter=',')
testingData = pd.read_csv('testing_SPY_prices_2016_2019.csv', delimiter=',')
