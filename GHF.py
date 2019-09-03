# -*- coding: utf-8 -*-

# Using Naive Bayes Theorem
# Importing from the libraries

import numpy as np
import matplotlib.pyplot as py
import pandas as pd


# importing the dataset
dataset = pd.read_csv('salary.csv')
X = dataset.iloc[:, [1, 2]].values
y = dataset.iloc[:, [2, 501]].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
