#Frank Saldivar, Mauricio Ortiz
#CSCI 6370 - Machine Learning, Spring 2019
#Dr. kim
#
#Project -- Stock Price Movement prediction using Tensorflow
#Goal: Compare and contrast differing ML techniques with a dataset
#       of stock prices from 3-15-1996 to 3-14-2019
#
#Notes: Training Set: 3-15-1996 -> 3-15-2016
#       Testing Set: 3-16-2016 -> 3-14-2019

#importing pkgs

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import genfromtxt


#reading in testing and training data
trainingData = pd.read_csv('training_SPY_prices_1996_2016.csv', delimiter=',')
testingData = pd.read_csv('testing_SPY_prices_2016_2019.csv', delimiter=',')

#drop irrelevant data, only keeping adjusted close to account for dividends/splits
#trainingData.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
#testingData.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
trainingData.drop(columns, axis=1, inplace=True)
testingData.drop(columns, axis=1, inplace=True)



#making a numpy array
trainingData = trainingData.values
testingData = testingData.values

#getting dimensions, preparing training and testing data
trainStart = 0
trainEnd = trainingData.shape[0] 
testStart = 0
testEnd = testingData.shape[0]

print("Size of training Data is: " + str(trainEnd) + "\nSize of testing data is: " + str(testEnd))


data_train_npArray = trainingData[np.arange(trainStart, trainEnd), :]
data_test_npArray = testingData[np.arange(testStart, testEnd), :]

plt.xlabel("Time")
plt.ylabel("price")
plt.plot(data_train_npArray)
plt.show()

#step 1 -- do regular linear regression, get accuracy and all that
X_train = data_train[:, 1:]
Y_train = data_train[:, 0]
X_test = data_test[:, 1:]
Y_test = data_test[:, 0]

weight = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

#hypothesis
h = x * w + b

#cost
cost = tf.reduce_mean(tf.square(h-y))




#step 2 -- scale down, try out LSTM or some sort of NN, rescale up, test accuracy

#scaling data down
#scaler = MinMaxScaler()
#data_train = scaler.fit_transform(data_train_npArray)
#data_test = scaler.transform(data_test_npArray)

#building our training/test of X & Y
#X_train = data_train[:, 1:]
#Y_train = data_train[:, 0]
print("done")


