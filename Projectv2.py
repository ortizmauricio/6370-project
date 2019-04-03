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
#------------------------------------------

#pandas for data containment
import pandas as pd
import tensorflow as tf
import numpy as np

#for plotting
import matplotlib.pyplot as plt
from math import sqrt
#for normalizing data
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
scaler = MinMaxScaler(feature_range=(0,1))

#to not get warnings
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#setting figure size
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 20,10

#------------done importing stuff-------------

#reading in testing and training data
dataSPY = pd.read_csv('SPY_data.csv', delimiter=',')

#print the head
print(dataSPY.head())

#setting index as date
dataSPY['Date'] = pd.to_datetime(dataSPY.Date,format='%Y-%m-%d')
dataSPY.index = dataSPY['Date']

#plotting data
plt.figure(figsize=(16,8))
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(dataSPY['Close'], label='Close Price History')
#plt.show()


#step 1 -- do regular linear regression, get accuracy and all that
#dataSPY['Date'] = pd.to_datetime(dataSPY.Date,format='%Y-%m-%d')
#dataSPY.index = dataSPY['Date']

#note to self: there are 5789 rows in data
#with 80/20 split
#training rows: 1 -> 4631 = 4631 entries
#testing rows: 4631 -> 5789 = 1158 entries

#partitioning data
trainingSet = dataSPY[:4631]
testingSet = dataSPY[4631:]

#x/y train split
x_train = trainingSet
x_train = x_train.drop(['Close', 'Date'],axis=1)
y_train = trainingSet['Close']

#x/y test split
x_test = testingSet
x_test = x_test.drop(['Close', 'Date'],axis=1)
y_test = testingSet['Close']


#convert to numpy arrays
x_train = x_train.values
y_train = y_train.values

print("x_train shape: " + str(x_train.shape) + "\ny_train shape: " + str(y_train.shape))

x_test = x_test.values
y_test = y_test.values

print("x_test shape: " + str(x_test.shape) + "\ny_test shape: " + str(y_test.shape))



#create linear model from sklearn
linearModel = LinearRegression()
linearModel.fit(x_train, y_train)

#make predictions and find rmse (root mean square error)
prediction = linearModel.predict(x_test)
print("prediction shape: " + str(prediction.shape))
rmse = sqrt(mean_squared_error(y_test, prediction))
print("Root Mean Square Error: " + str(rmse))

print("Sanity check\n\nShape of Prediction: " + str(prediction.shape) + "\nShape of y_test: " + str(y_test.shape))
#acc = accuracy_score(y_test,prediction)
#print("Accuracy: " + str())
#TODO: figure out how to plot actual prices vs prediction


#step 2 -- scale down, try out LSTM or some sort of NN, rescale up, test accuracy

#scaling data down
#scaler = MinMaxScaler()
#data_train = scaler.fit_transform(data_train_npArray)
#data_test = scaler.transform(data_test_npArray)

#building our training/test of X & Y
#X_train = data_train[:, 1:]
#Y_train = data_train[:, 0]
#X_test = data_test[:, 1:]
#Y_test = data_test[:, 0]
print("done")