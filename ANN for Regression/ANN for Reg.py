import numpy as np
import tensorflow as tf
import pandas as pd

#importing data
dataset = pd.read_excel('ML Projects\ANN for Regression\Folds5x2_pp.xlsx')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#initialize the ann
ann = tf.keras.models.Sequential()

#add input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#add another hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#add the output layer
ann.add(tf.keras.layers.Dense(units = 1))    #as we are doing regression here, there wont be an activation function for the output layer



#compile the ann
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

#training the ann
ann.fit(x_train, y_train, batch_size = 32,epochs = 100)


#predicting the result of the test set
y_pred = ann.predict(x_test)

#comparing the y_pred with the y_test
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))                 #converting y_pred into row x 1 matrics from 1 x columns matrics 
                







