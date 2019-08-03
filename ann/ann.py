# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('INSERT DATA HERE')
X = dataset.iloc[:,?, ?].values
y = dataset.iloc[:, ?].values

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]#remove dummy variables

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=?, init='uniform',activation='relu',input_dim=?))

# Adding the second hidden layer

classifier.add(Dense(output_dim=?, init='uniform',activation='relu'))

# Adding the output layer

classifier.add(Dense(output_dim=?, init='uniform',activation='sigmoid')) # use softmax if more than two categories

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_cross_entropy' metrics = ['accuracy'] ) # if more than 2 categories, loss = "categorical_cross_entropy"

# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
